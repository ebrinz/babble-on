"""Mac/MPS compatibility shim for the Plaid repo.

The Plaid model is a standard rotary transformer, but the repo wires it to
CUDA-only kernels (NVIDIA Apex FusedRMSNorm, FlashAttention FusedMLP + attention
+ rotary). This module registers pure-torch stand-ins in ``sys.modules`` so the
unmodified model code imports and runs on Apple Silicon (MPS) / CPU.

Import this module BEFORE importing ``lib.models``:

    import compat  # noqa: F401  (registers the shims)
    import lib.models

The parameter names/shapes of the stand-ins match the originals, so the
published Plaid-1B weights load unchanged.
"""
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- apex.normalization.FusedRMSNorm -> pure RMSNorm (param: `weight` [dim]) ---
class _RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


_apex = types.ModuleType("apex")
_apex_norm = types.ModuleType("apex.normalization")
_apex_norm.FusedRMSNorm = _RMSNorm
_apex.normalization = _apex_norm
sys.modules["apex"] = _apex
sys.modules["apex.normalization"] = _apex_norm


# --- flash_attn.ops.fused_dense.FusedMLP -> Linear/gelu/Linear (fc1, fc2) ---
class _FusedMLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None,
                 bias1=True, bias2=True, activation="gelu_approx", **kwargs):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias1)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias2)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


# --- flash_attn unpadded packed attention -> F.scaled_dot_product_attention ---
def _flash_attn_unpadded_qkvpacked_func(qkv, cu_seqlens, max_seqlen,
                                        dropout_p=0.0, softmax_scale=None,
                                        causal=False, **kwargs):
    # qkv: [total, 3, h, d] where total == batch * max_seqlen (fixed-length here)
    total, _three, h, d = qkv.shape
    b = total // max_seqlen
    qkv = qkv.view(b, max_seqlen, 3, h, d)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]      # [b, s, h, d]
    q = q.transpose(1, 2)                                    # [b, h, s, d]
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(
        q, k, v, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale
    )
    return out.transpose(1, 2).reshape(total, h, d)          # [total, h, d]


_fa = types.ModuleType("flash_attn")
_fa_iface = types.ModuleType("flash_attn.flash_attn_interface")
_fa_iface.flash_attn_unpadded_qkvpacked_func = _flash_attn_unpadded_qkvpacked_func
_fa_ops = types.ModuleType("flash_attn.ops")
_fa_fused = types.ModuleType("flash_attn.ops.fused_dense")
_fa_fused.FusedMLP = _FusedMLP
_fa.flash_attn_interface = _fa_iface
_fa.ops = _fa_ops
_fa_ops.fused_dense = _fa_fused
sys.modules["flash_attn"] = _fa
sys.modules["flash_attn.flash_attn_interface"] = _fa_iface
sys.modules["flash_attn.ops"] = _fa_ops
sys.modules["flash_attn.ops.fused_dense"] = _fa_fused
# Intentionally NOT registering flash_attn.layers.rotary, so lib/rotary.py's
# `try: import flash_attn.layers.rotary` fails and it uses its pure-torch
# torchscript rotary fallback.
