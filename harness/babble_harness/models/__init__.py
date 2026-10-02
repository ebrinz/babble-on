"""Denoiser implementations: ``stub`` (headless, deterministic) and
``diffusion_gemma`` (Transformers adapter with interpretability hooks)."""
from __future__ import annotations


def load_denoiser(name: str, **kw):
    if name == "stub":
        from .stub import StubDenoiser
        return StubDenoiser(**kw)
    if name in ("diffusion_gemma", "gemma", "hf"):
        from .diffusion_gemma import DiffusionGemmaDenoiser
        return DiffusionGemmaDenoiser(**kw)
    raise ValueError(f"unknown model '{name}' (stub | diffusion_gemma)")
