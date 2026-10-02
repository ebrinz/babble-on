"""Transformers adapter for ``google/diffusiongemma-26B-A4B-it`` with
interpretability hooks.

Written against the Transformers 5.8 ``diffusion_gemma`` module
(``DiffusionGemmaForBlockDiffusion``: ``model.encoder`` builds the prompt KV
cache once; ``model.decoder(decoder_input_ids=canvas, past_key_values=cache,
self_conditioning_logits=prev)`` is one denoising pass; ``lm_head`` + tanh
softcapping gives logits). The denoising step is reproduced here instead of
calling ``generate`` so that every random draw comes from the entropy tape.

**Not yet validated on hardware** (this container has no GPU and no weights).
The two places most likely to need a touch when first run are marked
``# ADAPT``: the encoder call that primes the cache, and the kwarg that
enables router-logit recording. Everything else is hooks on standard modules.

Captured per step (all moved to CPU, float32 / int32):
  hidden        [n_layers_captured, L, hidden]  residual stream after chosen
                decoder layers (``output_hidden_states``)
  router_counts [n_moe_layers, n_experts]       how often each expert was in
                the top-k across the canvas (from router logits)
  router_entropy [n_moe_layers]                 entropy of the expert usage
  logit_lens_agree [n_layers_captured]          share of positions whose
                top-1 under lm_head(norm(h_layer)) equals the final top-1
"""
from __future__ import annotations

from typing import Any

import numpy as np

MODEL_ID = "google/diffusiongemma-26B-A4B-it"


class DiffusionGemmaDenoiser:
    def __init__(self, model_id: str = MODEL_ID, prompt: str | None = None, device_map: str = "auto",
                 dtype: str = "auto", capture_layers: tuple[int, ...] | None = (6, 12, 18, 24, 30),
                 capture_router: bool = True, logit_lens: bool = True, system: str | None = None):
        import torch
        from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = DiffusionGemmaForBlockDiffusion.from_pretrained(model_id, dtype=dtype, device_map=device_map)
        self.model.eval()
        self.cfg = self.model.config
        self.canvas_length = int(self.cfg.canvas_length)
        self.vocab_size = int(self.cfg.text_config.vocab_size)
        self.softcap = float(self.cfg.text_config.final_logit_softcapping or 0.0)
        self.capture_layers = tuple(capture_layers or ())
        self.capture_router = capture_router
        self.logit_lens = logit_lens
        self.system = system
        self._cache = None
        self._prompt_ids = None
        self.set_prompt(prompt)

    # ---- prompt / cache --------------------------------------------------
    def set_prompt(self, prompt: str | None) -> None:
        """Encode the prompt once and cache its KV; later denoise() calls read it."""
        torch = self.torch
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt or ""})
        enc = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                 return_dict=True, return_tensors="pt")
        self._prompt_ids = enc["input_ids"].to(self.model.device)
        attn = enc.get("attention_mask")
        from transformers import DynamicCache
        self._cache = DynamicCache()
        with torch.no_grad():
            # ADAPT: Transformers' generate() calls an `encoder_forward` that
            # fills `past_key_values` from the prompt; the public forward with
            # only `input_ids` + a fresh cache does the same (it also runs one
            # throw-away decoder pass on a random canvas).
            out = self.model(input_ids=self._prompt_ids,
                             attention_mask=attn.to(self.model.device) if attn is not None else None,
                             past_key_values=self._cache, use_cache=True)
            self._cache = out.past_key_values
        self.prompt_text = prompt or ""

    # ---- one denoising pass ---------------------------------------------
    def denoise(self, canvas: np.ndarray, self_cond: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any]]:
        torch = self.torch
        dev = self.model.device
        ids = torch.as_tensor(np.asarray(canvas, dtype=np.int64), device=dev)[None, :]
        sc = None
        if self_cond is not None:
            sc = torch.as_tensor(np.asarray(self_cond, dtype=np.float32), device=dev)[None, :, :]
        kwargs: dict[str, Any] = dict(decoder_input_ids=ids, past_key_values=self._cache,
                                      self_conditioning_logits=sc,
                                      output_hidden_states=bool(self.capture_layers))
        if self.capture_router:
            kwargs["output_router_logits"] = True  # ADAPT if the recorder kwarg differs
        with torch.no_grad():
            dec = self.model.model.decoder(**kwargs)
            h_last = dec.last_hidden_state  # [1, L, hidden]
            logits = self._head(h_last)[0]  # [L, V]
        capture = self._capture(dec, logits)
        return logits.float().cpu().numpy(), capture

    def _head(self, h):
        torch = self.torch
        logits = self.model.lm_head(h)
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits.float()

    def _capture(self, dec, logits) -> dict[str, Any]:
        torch = self.torch
        cap: dict[str, Any] = {}
        hs = getattr(dec, "hidden_states", None)
        if hs is not None and self.capture_layers:
            layers = [min(l, len(hs) - 1) for l in self.capture_layers]
            cap["hidden"] = np.stack([hs[l][0].float().cpu().numpy() for l in layers]).astype(np.float32)
            if self.logit_lens:
                final_top = logits.argmax(-1)
                norm = getattr(self.model.model.decoder, "norm", None)
                agree = []
                with torch.no_grad():
                    for l in layers:
                        h = hs[l]
                        if norm is not None and l != len(hs) - 1:
                            h = norm(h)
                        agree.append((self._head(h)[0].argmax(-1) == final_top).float().mean().item())
                cap["logit_lens_agree"] = np.asarray(agree, dtype=np.float32)
        rl = getattr(dec, "router_logits", None)
        if rl is not None and self.capture_router:
            k = int(getattr(self.cfg.text_config, "top_k_experts", 8))
            counts, ents = [], []
            for layer_logits in rl:  # each [L, n_experts] (or [1, L, E])
                x = layer_logits.reshape(-1, layer_logits.shape[-1]).float()
                top = x.topk(k, dim=-1).indices.flatten()
                c = torch.bincount(top, minlength=x.shape[-1]).cpu().numpy()
                p = c / max(c.sum(), 1)
                ents.append(float(-(p[p > 0] * np.log(p[p > 0])).sum()))
                counts.append(c)
            cap["router_counts"] = np.stack(counts).astype(np.int32)
            cap["router_entropy"] = np.asarray(ents, dtype=np.float32)
        return cap

    # ---- text ---------------------------------------------------------------
    def decode(self, ids: np.ndarray, skip_special: bool = False) -> str:
        return self.processor.decode([int(i) for i in ids], skip_special_tokens=skip_special)

    def decode_each(self, ids: np.ndarray) -> list[str]:
        return [self.processor.decode([int(i)], skip_special_tokens=False) for i in ids]

    def trim_after_eos(self, ids: np.ndarray) -> np.ndarray:
        eos = self.cfg.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        for i, t in enumerate(ids):
            if int(t) in eos:
                return ids[:i]
        return ids
