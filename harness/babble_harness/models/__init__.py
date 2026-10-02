"""Denoiser implementations: ``stub`` (headless, deterministic) and
``diffusion_gemma`` (Transformers adapter with interpretability hooks)."""
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
    raise ValueError(f"unknown model '{name}' (stub | tiny | diffusion_gemma)")


def _ids_prompt(d):
    """Without a tokenizer, `set_prompt(text)` maps each character to a token id."""
    orig = d.set_prompt_ids

    def set_prompt(prompt):
        orig([2] + [ord(c) % d.vocab_size for c in (prompt or "")])
        d.prompt_text = prompt or ""

    d.set_prompt = set_prompt
    return orig
