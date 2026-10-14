"""MLX adapter for DiffusionGemma (Apple Silicon), mirroring
`diffusion_gemma.py` behind the same `Denoiser` protocol.

Wraps mlx-vlm's model (pinned 0.7.6). mlx-vlm's private decoder internals
(`_embed_canvas`, `_make_decoder_masks`, `_cache_offset`) are used in one
place, `_forward`; if mlx-vlm drifts, that is the one place to fix, and
`tests/test_mlx_adapter.py` is the gate.

Per step, as the Transformers adapter: embed canvas (+ self-conditioning
logits cast to the embedding dtype) -> decoder layers over the read-only
prompt cache -> norm -> float32 softcap(tied head). Captures have the same
names, shapes and dtypes; the last captured layer is post-norm, matching
Transformers' `hidden_states[-1]`.
"""
from __future__ import annotations

import contextlib
import warnings
from typing import Any

import numpy as np

MLX_MODEL_ID = "mlx-community/diffusiongemma-26B-A4B-it-4bit"
MLX_VLM_PIN = "0.7.6"


def _check_version() -> None:
    import mlx_vlm
    if mlx_vlm.__version__ != MLX_VLM_PIN:
        warnings.warn(f"mlx-vlm {mlx_vlm.__version__} differs from the pinned {MLX_VLM_PIN}; "
                      "run tests/test_mlx_adapter.py before trusting results", stacklevel=2)


@contextlib.contextmanager
def record_router(enabled: bool):
    """Record each MoE router's top-k indices during our forward only."""
    from mlx_vlm.models.diffusion_gemma import language

    calls: list = []
    if not enabled:
        yield calls
        return
    orig = language.Router.__call__

    def wrapped(self, x):
        idx, w = orig(self, x)
        calls.append(idx)
        return idx, w

    language.Router.__call__ = wrapped
    try:
        yield calls
    finally:
        language.Router.__call__ = orig


def _quant_summary(model_id: str | None) -> dict | None:
    """The checkpoint's declared quantisation: default bits / group size /
    mode, and how many modules override it (the 8-bit attention, dense MLP,
    router and embeddings of the 4-bit build)."""
    if not model_id:
        return None
    import json
    from pathlib import Path

    from mlx_vlm.utils import get_model_path

    q = json.loads((Path(get_model_path(model_id)) / "config.json").read_text()).get("quantization")
    if not q:
        return None
    return {"default": {"bits": q.get("bits"), "group_size": q.get("group_size"), "mode": q.get("mode", "affine")},
            "overrides": sum(1 for v in q.values() if isinstance(v, dict))}


class MLXGemmaDenoiser:
    def __init__(self, model, processor=None, capture_layers: tuple[int, ...] | None = (6, 12, 18, 24, 30),
                 capture_router: bool = True, logit_lens: bool = True, system: str | None = None,
                 model_id: str | None = None):
        import mlx.core as mx
        import mlx.nn as nn

        self.mx = mx
        self.model = model
        self.processor = processor
        self.model_id = model_id
        tc = model.config.text_config
        self.canvas_length = int(model.config.canvas_length)
        self.vocab_size = int(tc.vocab_size)
        self.softcap = float(tc.final_logit_softcapping or 0.0)
        self.top_k = int(tc.top_k_experts or 0)
        self.n_layers = int(tc.num_hidden_layers)
        self.capture_layers = tuple(capture_layers or ())
        self.capture_router = capture_router
        self.logit_lens = logit_lens
        self.system = system
        emb = model.model.decoder.embed_tokens
        # reference: self-conditioning logits are stored in the embedding dtype
        self._sc_dtype = emb.scales.dtype if isinstance(emb, nn.QuantizedEmbedding) else emb.weight.dtype
        self._cache = None
        self.prompt_text: str | None = None
        self.prompt_ids: list[int] = []

    @classmethod
    def from_pretrained(cls, model_id: str = MLX_MODEL_ID, **kw) -> "MLXGemmaDenoiser":
        _check_version()
        from mlx_vlm import load

        model, processor = load(model_id)
        return cls(model, processor, model_id=model_id, **kw)

    def describe(self) -> dict:
        import mlx.core as mx
        import mlx_vlm
        return {"backend": "mlx", "model_id": self.model_id, "mlx": mx.__version__,
                "mlx_vlm": mlx_vlm.__version__, "quantization": _quant_summary(self.model_id)}

    # ---- prompt / cache ---------------------------------------------------
    def tokenize_prompt(self, prompt: str | None) -> list[int]:
        if self.processor is None:
            raise RuntimeError("no processor: use set_prompt_ids() with a tokenized prompt")
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt or ""})
        tok = getattr(self.processor, "tokenizer", self.processor)
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if isinstance(ids, dict) or hasattr(ids, "keys"):
            ids = ids["input_ids"]
        return [int(i) for i in ids]

    def set_prompt(self, prompt: str | None) -> None:
        self.set_prompt_ids(self.tokenize_prompt(prompt))
        self.prompt_text = prompt or ""

    def set_prompt_ids(self, ids: list[int]) -> None:
        """Prefill a fresh cache once per prompt; the decoder only reads it."""
        mx = self.mx
        cache = self.model.make_cache()
        cache = self.model.diffusion_prefill_cache(mx.array([list(ids)], dtype=mx.int32), cache=cache)
        mx.eval([c.state for c in cache])
        self._cache = cache
        self.prompt_ids = [int(i) for i in ids]
        self.prompt_text = None

    def cache_offset(self) -> int:
        from mlx_vlm.models.diffusion_gemma.language import _cache_offset
        return _cache_offset(self._cache[0]) if self._cache else 0

    # ---- one denoising pass ---------------------------------------------
    def _head(self, h):
        mx = self.mx
        logits = self.model.model.decoder.embed_tokens.as_linear(h).astype(mx.float32)
        if self.softcap:
            logits = mx.tanh(logits / self.softcap) * self.softcap
        return logits

    def _forward(self, canvas: np.ndarray, self_cond: np.ndarray | None):
        """The single place that touches mlx-vlm internals."""
        from mlx_vlm.models.diffusion_gemma.language import _cache_offset

        mx = self.mx
        dec = self.model.model.decoder
        ids = mx.array(np.asarray(canvas, dtype=np.int32))[None]
        sc = None
        if self_cond is not None:
            sc = mx.array(np.asarray(self_cond, dtype=np.float32))[None].astype(self._sc_dtype)
        wanted = {min(max(l, 0), self.n_layers) for l in self.capture_layers}
        h = dec._embed_canvas(ids, sc, None)
        masks = dec._make_decoder_masks(h, self._cache, None)
        offset = _cache_offset(self._cache[0])
        kept = {0: h} if 0 in wanted else {}
        for i, (layer, c) in enumerate(zip(dec.layers, self._cache)):
            h = layer(h, masks.get(layer.layer_type), c, decoder=True, offset=offset)
            if i + 1 in wanted:
                kept[i + 1] = h
        h = dec.norm(h)
        if self.n_layers in kept:
            kept[self.n_layers] = h  # last layer post-norm, as Transformers
        return self._head(h), kept

    def denoise(self, canvas: np.ndarray, self_cond: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any]]:
        if self._cache is None:
            raise RuntimeError("call set_prompt()/set_prompt_ids() before denoise()")
        mx = self.mx
        dec = self.model.model.decoder
        with record_router(self.capture_router and bool(self.top_k)) as calls:
            logits, kept = self._forward(canvas, self_cond)
            layers = sorted(kept)
            lens = []
            if self.logit_lens and layers:
                final_top = mx.argmax(logits, axis=-1)
                for l in layers:
                    hl = kept[l] if l == self.n_layers else dec.norm(kept[l])
                    lens.append(mx.mean((mx.argmax(self._head(hl), axis=-1) == final_top).astype(mx.float32)))
            mx.eval(logits, *[kept[l] for l in layers], *lens, *calls)
        cap: dict[str, Any] = {}
        if layers:
            cap["hidden"] = np.stack([np.array(kept[l][0].astype(mx.float32)) for l in layers]).astype(np.float32)
            cap["hidden_layers"] = np.asarray(layers, dtype=np.int32)
            if lens:
                cap["logit_lens_agree"] = np.asarray([float(x.item()) for x in lens], dtype=np.float32)
        if calls:
            n_exp = int(self.model.config.text_config.num_experts)
            counts, ents = [], []
            for idx in calls:
                c = np.bincount(np.array(idx).reshape(-1), minlength=n_exp)
                p = c / max(c.sum(), 1)
                ents.append(float(-(p[p > 0] * np.log(p[p > 0])).sum()))
                counts.append(c)
            cap["router_counts"] = np.stack(counts).astype(np.int32)
            cap["router_entropy"] = np.asarray(ents, dtype=np.float32)
        return np.array(logits[0]), cap

    # ---- text ---------------------------------------------------------------
    def decode(self, ids: np.ndarray, skip_special: bool = False) -> str:
        if self.processor is None:
            return " ".join(str(int(i)) for i in ids)
        tok = getattr(self.processor, "tokenizer", self.processor)
        return tok.decode([int(i) for i in ids], skip_special_tokens=skip_special)

    def decode_each(self, ids: np.ndarray) -> list[str]:
        if self.processor is None:
            return [str(int(i)) for i in ids]
        tok = getattr(self.processor, "tokenizer", self.processor)
        return [tok.decode([int(i)], skip_special_tokens=False) for i in ids]

    def trim_after_eos(self, ids: np.ndarray) -> np.ndarray:
        cfg = self.model.config
        eos = (cfg.generation_config or {}).get("eos_token_id") if isinstance(cfg.generation_config, dict) else None
        eos = eos if eos is not None else cfg.eos_token_id
        eos = eos if eos is not None else cfg.text_config.eos_token_id
        if eos is None:
            return ids
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        for i, t in enumerate(ids):
            if int(t) in eos:
                return ids[:i]
        return ids
