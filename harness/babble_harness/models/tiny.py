"""A tiny random-weight DiffusionGemma (``--model tiny``) (same classes as the 26B checkpoint)
so the adapter and the reference sampler can be exercised without weights."""
from __future__ import annotations

VOCAB, CANVAS, HIDDEN, LAYERS, EXPERTS, TOPK = 256, 16, 64, 3, 4, 2


def tiny_model(seed: int = 0):
    import torch
    from transformers import (DiffusionGemmaConfig, DiffusionGemmaForBlockDiffusion, DiffusionGemmaTextConfig,
                              Gemma4VisionConfig)

    torch.manual_seed(seed)
    tc = DiffusionGemmaTextConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=128, num_hidden_layers=LAYERS,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32, global_head_dim=32, num_global_key_value_heads=1,
        num_experts=EXPERTS, top_k_experts=TOPK, moe_intermediate_size=32, sliding_window=64,
        max_position_embeddings=1024, pad_token_id=0, eos_token_id=1, bos_token_id=2,
    )
    vc = Gemma4VisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                            num_key_value_heads=2, head_dim=16, patch_size=16, position_embedding_size=16)
    cfg = DiffusionGemmaConfig(text_config=tc, vision_config=vc, canvas_length=CANVAS,
                               boi_token_id=3, eoi_token_id=4, image_token_id=5)
    return DiffusionGemmaForBlockDiffusion(cfg).eval()
