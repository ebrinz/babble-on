"""Transformers adapter for DiffusionGemma with interpretability hooks.

Mirrors what `DiffusionGemmaGenerationMixin.generate` does around each
denoising pass (Transformers 5.18, `generation_diffusion_gemma.py`):

  prefill  : model.model.encoder(input_ids, attention_mask, past_key_values=DynamicCache(config), position_ids)
  per step : model.model.decoder(decoder_input_ids=canvas, past_key_values=cache,
                                 self_conditioning_logits=prev_processed_logits,
                                 decoder_attention_mask=pad(mask, canvas, True),
                                 decoder_position_ids=arange(cur_len, cur_len+L))
             logits = softcap(lm_head(last_hidden_state))          (as ForBlockDiffusion.forward)

The decoder reads the prefix cache read-only, so one prefill serves every
step and every sample for the same prompt. Going through the decoder rather
than the top-level forward is what exposes the recorded router logits.

Validated against a tiny random-weight model in `tests/test_hf_adapter.py`
and, step for step, against the reference `generate` in
`tests/test_reference_parity.py`. Not yet run against the real checkpoint.

Captured per step (CPU, float32 / int32):
  hidden            [n_captured, L, hidden]  residual stream after chosen decoder layers
  router_counts     [n_moe_layers, n_experts] top-k expert usage across the canvas
  router_entropy    [n_moe_layers]
  logit_lens_agree  [n_captured]              top-1 under softcap(lm_head(norm(h))) == final top-1
"""
from __future__ import annotations

from typing import Any

import numpy as np

MODEL_ID = "google/diffusiongemma-26B-A4B-it"
NVFP4_MODEL_ID = "nvidia/diffusiongemma-26B-A4B-it-NVFP4"


def load_processor(model_id: str = MODEL_ID):
    """The multimodal processor (needs Pillow); falls back to the text
    tokenizer, which carries the same chat template, for text-only use."""
    from transformers import AutoProcessor, AutoTokenizer

    try:
        return AutoProcessor.from_pretrained(model_id)
    except ImportError:  # Pillow missing: text-only is all the harness needs
        return AutoTokenizer.from_pretrained(model_id)


class DiffusionGemmaDenoiser:
    def __init__(self, model, processor=None, capture_layers: tuple[int, ...] | None = (6, 12, 18, 24, 30),
                 capture_router: bool = True, logit_lens: bool = True, system: str | None = None):
        import torch

        self.torch = torch
        self.model = model.eval()
        self.processor = processor
        self.cfg = model.config
        self.text_cfg = self.cfg.text_config
        self.canvas_length = int(self.cfg.canvas_length)
        self.vocab_size = int(self.text_cfg.vocab_size)
        self.softcap = float(getattr(self.text_cfg, "final_logit_softcapping", 0.0) or 0.0)
        self.capture_layers = tuple(capture_layers or ())
        self.capture_router = capture_router
        self.logit_lens = logit_lens
        self.system = system
        self.top_k = int(getattr(self.text_cfg, "top_k_experts", 0) or 0)
        self._cache = None
        self._dec_mask = None
        self._dec_pos = None
        self.prompt_text: str | None = None
        self.prompt_ids: list[int] = []

    # ---- construction ------------------------------------------------------
    @classmethod
    def from_pretrained(cls, model_id: str = MODEL_ID, quant: str | None = None, device_map: str = "auto",
                        dtype: str = "auto", **kw) -> "DiffusionGemmaDenoiser":
        """Load weights. ``quant``: ``None`` (as stored), ``"nvfp4"`` (NVIDIA's
        checkpoint, needs a GPU with FP4 kernels), ``"bnb4"`` (bitsandbytes
        NF4; CUDA only, and the MoE experts only quantise with Unsloth's
        per-expert Linear4bit swap — check ``quantized_fraction()`` after
        loading, a plain load leaves ~85 % of the parameters in bf16)."""
        from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

        load: dict[str, Any] = dict(dtype=dtype, device_map=device_map)
        if quant == "nvfp4" and model_id == MODEL_ID:
            model_id = NVFP4_MODEL_ID
        elif quant == "bnb4":
            import torch
            from transformers import BitsAndBytesConfig
            load["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                             bnb_4bit_compute_dtype=torch.bfloat16)
        elif quant not in (None, "none", "nvfp4"):
            raise ValueError(f"unknown quant '{quant}' (none | nvfp4 | bnb4)")
        processor = load_processor(model_id)
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(model_id, **load)
        return cls(model, processor, **kw)

    def quantized_fraction(self) -> float:
        """Share of parameters held in a quantised dtype (uint8/int8/fp4 storage)."""
        total = quant = 0
        for p in self.model.parameters():
            n = p.numel()
            total += n
            if p.dtype in (self.torch.uint8, self.torch.int8) or "fp4" in str(p.dtype) or "float4" in str(p.dtype):
                quant += n
        return quant / max(total, 1)

    # ---- prompt / cache --------------------------------------------------
    def tokenize_prompt(self, prompt: str | None):
        if self.processor is None:
            raise RuntimeError("no processor: use set_prompt_ids() with a tokenized prompt")
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt or ""})
        enc = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                 return_dict=True, return_tensors="pt")
        return enc["input_ids"][0].tolist()

    def set_prompt(self, prompt: str | None) -> None:
        self.set_prompt_ids(self.tokenize_prompt(prompt))
        self.prompt_text = prompt or ""

    def set_prompt_ids(self, ids: list[int]) -> None:
        """Encode the prompt once into a fresh KV cache (reference: step 1.a
        of `generate`, prefill)."""
        torch = self.torch
        from transformers import DynamicCache

        dev = self.model.device
        ids_t = torch.as_tensor([list(ids)], dtype=torch.long, device=dev)
        cur_len = ids_t.shape[1]
        attn = torch.ones((1, cur_len), dtype=torch.bool, device=dev)
        cache = DynamicCache(config=self.cfg.get_text_config(decoder=True))
        with torch.no_grad():
            enc = self.model.model.encoder(input_ids=ids_t, attention_mask=attn, past_key_values=cache,
                                           position_ids=torch.arange(cur_len, device=dev)[None])
        self._cache = enc.past_key_values
        self._dec_mask = torch.nn.functional.pad(attn, (0, self.canvas_length), value=True)
        self._dec_pos = torch.arange(cur_len, cur_len + self.canvas_length, dtype=torch.int32, device=dev)[None]
        self.prompt_ids = list(ids)
        self.prompt_text = None

    # ---- one denoising pass ---------------------------------------------
    def denoise(self, canvas: np.ndarray, self_cond: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any]]:
        if self._cache is None:
            raise RuntimeError("call set_prompt()/set_prompt_ids() before denoise()")
        torch = self.torch
        dev = self.model.device
        ids = torch.as_tensor(np.asarray(canvas, dtype=np.int64), device=dev)[None, :]
        sc = None
        if self_cond is not None:
            # reference: processed logits cast to the embedding dtype
            sc = torch.as_tensor(np.asarray(self_cond, dtype=np.float32), device=dev)[None]
            sc = sc.to(self.model.model.decoder.embed_tokens.weight.dtype)
        kwargs: dict[str, Any] = dict(decoder_input_ids=ids, past_key_values=self._cache, self_conditioning_logits=sc,
                                      decoder_attention_mask=self._dec_mask, decoder_position_ids=self._dec_pos)
        if self.capture_layers:
            kwargs["output_hidden_states"] = True
        if self.capture_router:
            kwargs["output_router_logits"] = True
        with torch.no_grad():
            dec = self.model.model.decoder(**kwargs)
            logits = self.head(dec.last_hidden_state)[0]  # [L, V] float32
            capture = self._capture(dec, logits)
        return logits.cpu().numpy(), capture

    def head(self, h):
        """`lm_head` + tanh softcapping, exactly as `DiffusionGemmaForBlockDiffusion.forward`."""
        torch = self.torch
        logits = self.model.lm_head(h).to(torch.float32)
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits

    def _capture(self, dec, logits) -> dict[str, Any]:
        torch = self.torch
        cap: dict[str, Any] = {}
        hs = getattr(dec, "hidden_states", None)
        if hs is not None and self.capture_layers:
            # hidden_states[0] is the embedding output, [i] the output of layer i (1-based).
            layers = [min(max(l, 0), len(hs) - 1) for l in self.capture_layers]
            cap["hidden"] = np.stack([hs[l][0].float().cpu().numpy() for l in layers]).astype(np.float32)
            cap["hidden_layers"] = np.asarray(layers, dtype=np.int32)
            if self.logit_lens:
                final_top = logits.argmax(-1)
                norm = getattr(self.model.model.decoder, "norm", None)
                agree = []
                for l in layers:
                    h = hs[l]
                    if norm is not None and not torch.equal(h, dec.last_hidden_state):
                        h = norm(h)
                    agree.append((self.head(h)[0].argmax(-1) == final_top).float().mean().item())
                cap["logit_lens_agree"] = np.asarray(agree, dtype=np.float32)
        rl = getattr(dec, "router_logits", None)
        if rl is not None and self.capture_router and self.top_k:
            counts, ents = [], []
            for layer_logits in rl:  # [L, E]
                x = layer_logits.reshape(-1, layer_logits.shape[-1]).float()
                top = x.topk(self.top_k, dim=-1).indices.flatten()
                c = torch.bincount(top, minlength=x.shape[-1]).cpu().numpy()
                p = c / max(c.sum(), 1)
                ents.append(float(-(p[p > 0] * np.log(p[p > 0])).sum()))
                counts.append(c)
            cap["router_counts"] = np.stack(counts).astype(np.int32)
            cap["router_entropy"] = np.asarray(ents, dtype=np.float32)
        return cap

    # ---- text ---------------------------------------------------------------
    def decode(self, ids: np.ndarray, skip_special: bool = False) -> str:
        if self.processor is None:
            return " ".join(str(int(i)) for i in ids)
        return self.processor.decode([int(i) for i in ids], skip_special_tokens=skip_special)

    def decode_each(self, ids: np.ndarray) -> list[str]:
        if self.processor is None:
            return [str(int(i)) for i in ids]
        return [self.processor.decode([int(i)], skip_special_tokens=False) for i in ids]

    def trim_after_eos(self, ids: np.ndarray) -> np.ndarray:
        # Reference `_finalize_canvas` pads after the first id in
        # generation_config.eos_token_id ([1, 106, 50] for the checkpoint).
        eos = None
        for c in (getattr(self.model, "generation_config", None), self.cfg, self.text_cfg):
            if c is None:
                continue
            try:
                eos = getattr(c, "eos_token_id", None)
            except AttributeError:  # heterogeneity configs raise instead of returning None
                eos = None
            if eos is not None:
                break
        if eos is None:
            return ids
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        for i, t in enumerate(ids):
            if int(t) in eos:
                return ids[:i]
        return ids
