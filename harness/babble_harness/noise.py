"""The entropy contract (mirror of ``bbrec/src/noise.rs``) and the tape.

Every random draw the sampler makes is a uniform in (0, 1) pulled from an
``EntropyTape``: 4 bytes little-endian -> u32 -> ``(u + 0.5) / 2**32``.
Integers come from ``floor(u * n)`` clamped to ``n - 1``; categorical draws
are inverse-CDF with one uniform per draw. Pinned by
``docs/contract/noise_vectors.json`` on both sides.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TWO32 = 4_294_967_296.0


def bytes_to_uniforms(raw: bytes | np.ndarray, n: int) -> np.ndarray:
    """``4*n`` raw bytes -> ``n`` float64 uniforms strictly inside (0, 1)."""
    buf = np.frombuffer(bytes(raw), dtype=np.uint8)
    need = n * 4
    if buf.size < need:
        raise ValueError(f"need {need} bytes for {n} uniforms, got {buf.size}")
    u32 = buf[:need].view("<u4").astype(np.float64)
    return (u32 + 0.5) / TWO32


def uniform_to_index(u: np.ndarray | float, n: int) -> np.ndarray:
    """Uniform (0,1) -> integer in ``[0, n)``: ``min(floor(u*n), n-1)``."""
    if n <= 0:
        raise ValueError("n must be positive")
    i = np.floor(np.asarray(u, dtype=np.float64) * n).astype(np.int64)
    return np.clip(i, 0, n - 1)


def categorical_inverse_cdf(probs: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Exact categorical sampling with one uniform per row.

    ``probs``: ``[rows, V]`` (any float dtype, rows need not sum to exactly 1);
    ``u``: ``[rows]``. Returns the first index whose cumulative mass exceeds
    ``u * total``; scaling by the row total makes a slightly unnormalised row
    (bf16 softmax) behave as if normalised. Clamped to ``V-1``.
    """
    p = np.asarray(probs, dtype=np.float64)
    cdf = np.cumsum(p, axis=-1)
    total = cdf[:, -1:]
    target = np.asarray(u, dtype=np.float64)[:, None] * total
    idx = (cdf <= target).sum(axis=-1)
    return np.minimum(idx, p.shape[-1] - 1)


def budget_bytes(canvas_length: int, max_steps: int) -> int:
    """Worst-case entropy one canvas can consume: the random initial canvas,
    then per step one categorical draw and one renoise draw per position."""
    return 4 * canvas_length * (1 + 2 * max_steps)


@dataclass
class Draw:
    purpose: str
    offset: int
    nbytes: int
    step: int | None = None


@dataclass
class EntropyTape:
    """Sequential reader over a byte buffer with a consumption log.

    Raises ``EntropyExhausted`` rather than topping up: a seed labelled
    *out_band* must stay out-of-band bytes only.
    """

    data: bytes
    label: str = "unlabelled"
    meta: dict = field(default_factory=dict)
    pos: int = 0
    log: list[Draw] = field(default_factory=list)
    step: int | None = None

    @classmethod
    def from_seed(cls, stem: str | Path) -> "EntropyTape":
        """Load an app-exported bundle (``<stem>.seed.bin`` + ``.seed.json``)."""
        stem = str(stem)
        if stem.endswith(".seed.bin") or stem.endswith(".seed.json"):
            stem = stem.rsplit(".seed.", 1)[0]
        meta = json.loads(Path(stem + ".seed.json").read_text())
        data = Path(stem + ".seed.bin").read_bytes()
        if len(data) != meta["bytes"]:
            raise ValueError(f"{stem}.seed.bin holds {len(data)} bytes, json says {meta['bytes']}")
        return cls(data=data, label=meta.get("label", "unlabelled"), meta=meta)

    @property
    def remaining(self) -> int:
        return len(self.data) - self.pos

    @property
    def consumed(self) -> int:
        return self.pos

    def uniforms(self, n: int, purpose: str = "uniform") -> np.ndarray:
        need = 4 * n
        if self.remaining < need:
            raise EntropyExhausted(
                f"tape '{self.label}': {purpose} needs {need} bytes, {self.remaining} left "
                f"(consumed {self.pos} of {len(self.data)})"
            )
        u = bytes_to_uniforms(self.data[self.pos:self.pos + need], n)
        self.log.append(Draw(purpose, self.pos, need, self.step))
        self.pos += need
        return u

    def randint(self, n: int, size: int, purpose: str = "randint") -> np.ndarray:
        return uniform_to_index(self.uniforms(size, purpose), n)

    def categorical(self, probs: np.ndarray, purpose: str = "categorical") -> np.ndarray:
        rows = probs.shape[0]
        return categorical_inverse_cdf(probs, self.uniforms(rows, purpose))

    def provenance(self) -> list[dict]:
        return [d.__dict__ for d in self.log]


class EntropyExhausted(RuntimeError):
    pass


def prng_tape(seed: int, nbytes: int, label: str = "prng") -> EntropyTape:
    """Control arm: PCG64 bytes. Same contract, no physics."""
    rng = np.random.default_rng(seed)
    return EntropyTape(data=rng.bytes(nbytes), label=label, meta={"prng_seed": seed})
