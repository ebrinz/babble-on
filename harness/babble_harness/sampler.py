"""Entropy-driven mirror of DiffusionGemma's entropy-bounded sampler.

Reference: DeepMind ``gemma/diffusion/_sampler.py`` (``SampleFromPredictions``,
``AnnealingTemperatureShaper``) and Transformers
``generation_diffusion_gemma.py`` (``EntropyBoundSampler``, ``_denoising_step``).
Every random draw the reference makes with ``torch.randint`` /
``torch.multinomial`` (``jax.random.randint`` / ``jax.random.categorical``) is
made here from the ``EntropyTape`` instead, one uniform per draw:

    canvas  <- tape.randint(V, L)                         (initial canvas)
    for step in S..1:
        logits   <- denoiser(canvas, self_cond_logits)
        T        <- t_min + (t_max - t_min) * step / S
        probs    <- softmax(logits / T)
        H_i      <- entropy(probs_i)                      per position
        accept   <- entropy-bound selection on H (all on the last step)
        sampled  <- tape.categorical(probs)               (one uniform / position)
        renoise  <- tape.randint(V, L)                    (one uniform / position)
        canvas   <- where(accept, sampled, renoise)
        stop when mean(H) <= confidence_threshold and the argmax canvas has not
        changed for `stability_threshold` consecutive steps

The denoiser is abstract (``Denoiser`` protocol) so the loop runs headless on
a stub and on the real model through ``models/diffusion_gemma.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .noise import EntropyTape


@dataclass
class SamplerConfig:
    canvas_length: int = 256
    vocab_size: int = 262_144
    max_denoising_steps: int = 48
    t_max: float = 0.8
    t_min: float = 0.4
    entropy_bound: float = 0.1
    confidence_threshold: float = 0.005  # mean entropy for early stop
    stability_threshold: int = 1  # consecutive unchanged-argmax steps
    early_stop: bool = True


class Denoiser(Protocol):
    """One denoising forward. ``canvas``: int64 ``[L]``. ``self_cond``: the
    previous step's float32 logits ``[L, V]`` or ``None`` on the first step.
    Returns ``(logits [L, V] float32, capture)`` where ``capture`` is any dict
    of per-step observations (hidden states, routing...) the adapter made."""

    def denoise(self, canvas: np.ndarray, self_cond: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any]]: ...


def temperature(step: int, cfg: SamplerConfig) -> float:
    """Transformers: ``t_min + (t_max - t_min) * cur_step / max_denoising_steps``
    with ``cur_step`` counting down from ``max_denoising_steps`` to 1."""
    return cfg.t_min + (cfg.t_max - cfg.t_min) * (step / cfg.max_denoising_steps)


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    x = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def token_entropy(probs: np.ndarray) -> np.ndarray:
    """Per-row Shannon entropy in nats, ``0·log0 = 0``."""
    p = np.asarray(probs, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.where(p > 0, np.log(p), 0.0)
    return -(p * lp).sum(axis=-1)


def entropy_bound_selection(entropy: np.ndarray, bound: float) -> np.ndarray:
    """Reference rule: sort ascending; accept position ``j`` (in sorted order)
    while ``cumsum[j] - entropy[j] <= bound``, i.e. the entropy of everything
    *more confident* than it fits under the bound. The most confident position
    is always accepted."""
    order = np.argsort(entropy, kind="stable")
    s = entropy[order]
    cum = np.cumsum(s)
    sorted_mask = (cum - s) <= bound
    mask = np.zeros_like(sorted_mask)
    mask[order] = sorted_mask
    return mask


@dataclass
class StepTrace:
    step: int  # countdown value (S..1)
    index: int  # 0-based iteration
    temperature: float
    canvas_in: np.ndarray
    canvas_out: np.ndarray
    argmax: np.ndarray
    accepted: np.ndarray
    entropy: np.ndarray
    sampled: np.ndarray
    tape_offset_before: int
    tape_offset_after: int
    capture: dict[str, Any] = field(default_factory=dict)

    @property
    def n_accepted(self) -> int:
        return int(self.accepted.sum())

    @property
    def mean_entropy(self) -> float:
        return float(self.entropy.mean())


@dataclass
class SampleResult:
    final_canvas: np.ndarray  # argmax canvas after the last executed step
    steps: list[StepTrace]
    stopped_early: bool
    initial_canvas: np.ndarray
    tape_label: str
    tape_consumed: int
    provenance: list[dict]

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def steps_to_commit(self) -> np.ndarray:
        """Per position: the iteration index at which its final token was first
        accepted and never changed again (``-1`` if it never stabilised)."""
        L = self.final_canvas.shape[0]
        out = np.full(L, -1, dtype=np.int64)
        for i in range(L):
            tok = self.final_canvas[i]
            since = None
            for st in self.steps:
                if st.accepted[i] and st.canvas_out[i] == tok:
                    if since is None:
                        since = st.index
                else:
                    since = None
            out[i] = -1 if since is None else since
        return out


def sample_canvas(denoiser: Denoiser, tape: EntropyTape, cfg: SamplerConfig,
                  keep_capture: bool = True, on_step=None) -> SampleResult:
    L, V, S = cfg.canvas_length, cfg.vocab_size, cfg.max_denoising_steps
    tape.step = None
    canvas = tape.randint(V, L, "initial_canvas")
    initial = canvas.copy()
    self_cond: np.ndarray | None = None
    prev_argmax: np.ndarray | None = None
    stable = 0
    traces: list[StepTrace] = []
    stopped_early = False
    argmax = canvas

    for idx, step in enumerate(range(S, 0, -1)):
        tape.step = idx
        before = tape.consumed
        logits, capture = denoiser.denoise(canvas, self_cond)
        logits = np.asarray(logits, dtype=np.float32)
        T = temperature(step, cfg)
        probs = softmax(logits.astype(np.float64) / T)
        H = token_entropy(probs)
        argmax = logits.argmax(axis=-1)
        if step == 1:
            accepted = np.ones(L, dtype=bool)
        else:
            accepted = entropy_bound_selection(H, cfg.entropy_bound)
        sampled = tape.categorical(probs, "categorical")
        noise = tape.randint(V, L, "renoise")
        new_canvas = np.where(accepted, sampled, noise)

        tr = StepTrace(step, idx, T, canvas.copy(), new_canvas.copy(), argmax.copy(), accepted,
                       H, sampled, before, tape.consumed, capture if keep_capture else {})
        traces.append(tr)
        if on_step:
            on_step(tr)

        self_cond = logits
        canvas = new_canvas

        if cfg.early_stop and step > 1:
            if prev_argmax is not None and np.array_equal(argmax, prev_argmax):
                stable += 1
            else:
                stable = 0
            prev_argmax = argmax
            if H.mean() <= cfg.confidence_threshold and stable >= cfg.stability_threshold:
                stopped_early = True
                break

    return SampleResult(
        final_canvas=argmax.copy(), steps=traces, stopped_early=stopped_early,
        initial_canvas=initial, tape_label=tape.label, tape_consumed=tape.consumed,
        provenance=tape.provenance(),
    )
