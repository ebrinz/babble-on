"""Denoiser implementations: ``stub`` (headless, deterministic) and
``diffusion_gemma`` (Transformers adapter with interpretability hooks),
``mlx`` (the same on Apple Silicon via mlx-vlm), ``tiny`` / ``mlx_tiny`` (random weights)."""
from __future__ import annotations


def load_denoiser(name: str, **kw):
    if name == "stub":
        from .stub import StubDenoiser
        return StubDenoiser(**kw)
    if name in ("diffusion_gemma", "gemma", "hf"):
        from .diffusion_gemma import DiffusionGemmaDenoiser
        if "model" in kw:
            return DiffusionGemmaDenoiser(**kw)
        return DiffusionGemmaDenoiser.from_pretrained(**kw)
    if name == "tiny":
        # the real adapter on a tiny random-weight DiffusionGemma: exercises every
        # Transformers code path without weights (no tokenizer: ids are the "text")
        from .diffusion_gemma import DiffusionGemmaDenoiser
        from .tiny import tiny_model
        d = DiffusionGemmaDenoiser(tiny_model(kw.pop("seed", 0)), capture_layers=kw.pop("capture_layers", (1, 2, 3)), **kw)
        d.set_prompt_ids = _ids_prompt(d)
        return d
    if name == "mlx":
        _require_mlx()
        from .mlx_gemma import MLXGemmaDenoiser
        if "model" in kw:
            return MLXGemmaDenoiser(**kw)
        return MLXGemmaDenoiser.from_pretrained(**kw)
    if name == "mlx_tiny":
        # the MLX adapter on the same tiny random-weight model as `tiny`
        _require_mlx()
        from .mlx_gemma import MLXGemmaDenoiser
        from .tiny import tiny_mlx_model
        d = MLXGemmaDenoiser(tiny_mlx_model(kw.pop("seed", 0)), capture_layers=kw.pop("capture_layers", (1, 2, 3)), **kw)
        d.set_prompt_ids = _ids_prompt(d)
        return d
    raise ValueError(f"unknown model '{name}' (stub | tiny | diffusion_gemma | mlx | mlx_tiny)")


def _require_mlx():
    try:
        import mlx_vlm  # noqa: F401
    except ImportError as e:
        raise ImportError("the mlx backend needs mlx-vlm: pip install -e '.[mlx]' (Apple Silicon only)") from e


def _ids_prompt(d):
    """Without a tokenizer, `set_prompt(text)` maps each character to a token id."""
    orig = d.set_prompt_ids

    def set_prompt(prompt):
        orig([2] + [ord(c) % d.vocab_size for c in (prompt or "")])
        d.prompt_text = prompt or ""

    d.set_prompt = set_prompt
    return orig
