"""A tiny random-weight DiffusionGemma (``--model tiny``) (same classes as the 26B checkpoint)
so the adapter and the reference sampler can be exercised without weights."""
from __future__ import annotations

VOCAB, CANVAS, HIDDEN, LAYERS, EXPERTS, TOPK = 256, 16, 64, 3, 4, 2


def tiny_model(seed: int = 0, vocab_size: int = VOCAB):
    import torch
    from transformers import (DiffusionGemmaConfig, DiffusionGemmaForBlockDiffusion, DiffusionGemmaTextConfig,
                              Gemma4VisionConfig)

    torch.manual_seed(seed)
    tc = DiffusionGemmaTextConfig(
        vocab_size=vocab_size, hidden_size=HIDDEN, intermediate_size=128, num_hidden_layers=LAYERS,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32, global_head_dim=32, num_global_key_value_heads=1,
        num_experts=EXPERTS, top_k_experts=TOPK, moe_intermediate_size=32, sliding_window=64,
        max_position_embeddings=1024, pad_token_id=0, eos_token_id=1, bos_token_id=2,
    )
    vc = Gemma4VisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                            num_key_value_heads=2, head_dim=16, patch_size=16, position_embedding_size=16)
    cfg = DiffusionGemmaConfig(text_config=tc, vision_config=vc, canvas_length=CANVAS,
                               boi_token_id=3, eoi_token_id=4, image_token_id=5)
    return DiffusionGemmaForBlockDiffusion(cfg).eval()


# Transformers serialises these as None; mlx-vlm would then fall back to its
# 26B defaults (512 / 2) and silently build the wrong model.
_FILL_FROM = {"global_head_dim": "head_dim", "num_global_key_value_heads": "num_key_value_heads"}


def translate_config(hf_config) -> dict:
    """Transformers DiffusionGemmaConfig -> mlx-vlm ModelConfig dict (text only).
    Fills the known None fields and refuses a None in any field mlx-vlm types
    as non-optional (it would substitute its own 26B default)."""
    import dataclasses

    from mlx_vlm.models.diffusion_gemma.config import TextConfig

    d = hf_config.to_dict()
    tc = dict(d["text_config"])
    for k, src in _FILL_FROM.items():
        if tc.get(k) is None:
            tc[k] = tc[src]
    for f in dataclasses.fields(TextConfig):
        optional = "Optional" in str(f.type) or "None" in str(f.type)
        if f.name in tc and tc[f.name] is None and not optional:
            raise ValueError(f"text_config.{f.name} is None; mlx-vlm would use its default {f.default!r}")
    return {"text_config": tc, "vision_config": None, "model_type": "diffusion_gemma",
            "canvas_length": d["canvas_length"], "boi_token_id": d.get("boi_token_id"),
            "eoi_token_id": d.get("eoi_token_id"), "image_token_id": d.get("image_token_id")}


def tiny_mlx_model(seed: int = 0, vocab_size: int = VOCAB):
    """`tiny_model(seed)` rebuilt in mlx-vlm with the same weights."""
    import mlx.core as mx
    from mlx_vlm.models.diffusion_gemma.config import ModelConfig
    from mlx_vlm.models.diffusion_gemma.diffusion_gemma import Model

    hf = tiny_model(seed, vocab_size)
    model = Model(ModelConfig.from_dict(translate_config(hf.config)))
    weights = {k: mx.array(v.detach().float().numpy()) for k, v in hf.state_dict().items()}
    model.load_weights(list(model.sanitize(weights).items()), strict=True)
    mx.eval(model.parameters())
    return model
