"""A tiny deterministic "denoiser" so the whole pipeline runs without weights.

It behaves like a model that is confident about a fixed target sentence and
gets more confident the more of the canvas already matches it (so early
stopping and the entropy bound exercise their real code paths), and it emits a
small fake activation capture shaped like the real adapter's.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class StubDenoiser:
    def __init__(self, vocab_size: int = 64, canvas_length: int = 16, hidden: int = 8,
                 n_layers: int = 3, n_experts: int = 4, sharpness: float = 2.0, seed: int = 0):
        self.V, self.L, self.hidden = vocab_size, canvas_length, hidden
        self.vocab_size, self.canvas_length = vocab_size, canvas_length
        self.n_layers, self.n_experts, self.sharpness = n_layers, n_experts, sharpness
        rng = np.random.default_rng(seed)
        self.target = rng.integers(0, vocab_size, size=canvas_length)
        self.proj = rng.standard_normal((vocab_size, hidden)).astype(np.float32)
        self.calls = 0

    def denoise(self, canvas: np.ndarray, self_cond: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any]]:
        self.calls += 1
        match = (canvas == self.target).mean()
        conf = self.sharpness * (1.0 + 4.0 * match)  # sharper as the canvas converges
        logits = np.zeros((self.L, self.V), dtype=np.float32)
        logits[np.arange(self.L), self.target] = conf
        # positions already right get a bonus (committed tokens are more certain)
        logits[np.arange(self.L), self.target] += 3.0 * (canvas == self.target)
        if self_cond is not None:
            logits += 0.1 * self_cond.astype(np.float32)
        hidden = self.proj[canvas] * (1.0 + match)  # [L, hidden]
        counts = np.bincount(canvas % self.n_experts, minlength=self.n_experts)[None, :].repeat(self.n_layers, 0)
        p = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=1)
        capture = {
            "hidden": np.stack([hidden * (l + 1) for l in range(self.n_layers)]),  # [layers, L, hidden]
            "router_counts": counts.astype(np.int32),
            "router_entropy": ent.astype(np.float32),
            "logit_lens_agree": np.linspace(0.2, 1.0, self.n_layers, dtype=np.float32),
        }
        return logits, capture
