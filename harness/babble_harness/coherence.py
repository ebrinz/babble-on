"""Offline replay of the observatory's coherence walk (mirror of
``src-tauri/src/stats.rs``: ``Stats::push`` + ``Stats::tick_trials``).

The app finalises a *trial* on an engine tick when at least ``trial_interval``
has elapsed since the last trial **and** at least ``trial_min_bits`` have
accumulated. Each trial contributes a monobit z-score
``z = (ones - n/2) / sqrt(n/4)`` to the cumulative walk ``C_k``; the normalised
deviation is ``sigma = C_k / sqrt(k)`` and the band is classified at
1.96 / 2.576 / 3.291 (95 / 99 / 99.9 %).

A recording frame is one engine tick's bytes, so replaying "push frame, then
tick" labels each frame with the band in force right after it arrived — which
is exactly when the app's anomaly bank decides whether to deposit it. The
header carries the walk state at the moment recording started (cumulative
sum, trial count, the partial trial, time since the last trial), so the replay
continues the app's walk rather than starting from zero; files written before
those fields existed replay from zero, which is only exact if the recording
started on a fresh session.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .recording import Frame, RecordingHeader

Z95 = 1.959_963_98
Z99 = 2.575_829_30
Z999 = 3.290_526_73

BANDS = ("in-band", "95%", "99%", "99.9%")


def band_of(sigma: float) -> str:
    a = abs(sigma)
    if a >= Z999:
        return "99.9%"
    if a >= Z99:
        return "99%"
    if a >= Z95:
        return "95%"
    return "in-band"


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint64)


def popcount(data: bytes) -> int:
    return int(_POPCOUNT[np.frombuffer(data, dtype=np.uint8)].sum()) if data else 0


@dataclass
class Trial:
    k: int
    t_ns: int
    z: float
    cum: float
    sigma: float
    band: str


@dataclass
class LabelledFrame:
    t_ns: int
    offset: int  # byte offset into the concatenated stream
    length: int
    band: str  # band in force after this frame was pushed
    sigma: float
    k: int


@dataclass
class Walk:
    """Incremental replica of the app's trial walk."""

    trial_interval_ns: int = 100_000_000
    trial_min_bits: int = 2048
    trial_ones: int = 0
    trial_bits: int = 0
    trial_last_ns: int = 0
    cum: float = 0.0
    k: int = 0
    sigma: float = 0.0
    band: str = "in-band"
    trials: list[Trial] = field(default_factory=list)

    @classmethod
    def from_header(cls, h: RecordingHeader) -> "Walk":
        w = cls(trial_interval_ns=h.trial_interval_ms * 1_000_000, trial_min_bits=h.trial_min_bits,
                trial_ones=h.trial_ones, trial_bits=h.trial_bits, cum=h.walk_cum, k=h.walk_k,
                # frame clocks start at 0 when recording starts; the last trial was this long before that
                trial_last_ns=-int(h.since_last_trial_ns))
        if w.k > 0:
            w.sigma = w.cum / math.sqrt(w.k)
            w.band = band_of(w.sigma)
        return w

    def push(self, data: bytes) -> None:
        self.trial_ones += popcount(data)
        self.trial_bits += len(data) * 8

    def tick(self, now_ns: int) -> Trial | None:
        if now_ns - self.trial_last_ns < self.trial_interval_ns:
            return None
        if self.trial_bits < self.trial_min_bits:
            return None
        n = float(self.trial_bits)
        z = (self.trial_ones - n / 2.0) / math.sqrt(n / 4.0)
        self.cum += z
        self.k += 1
        self.trial_ones = 0
        self.trial_bits = 0
        self.trial_last_ns = now_ns
        self.sigma = self.cum / math.sqrt(self.k)
        self.band = band_of(self.sigma)
        t = Trial(self.k, now_ns, z, self.cum, self.sigma, self.band)
        self.trials.append(t)
        return t


def label_frames(frames: list[Frame], header: RecordingHeader | None = None) -> tuple[list[LabelledFrame], Walk]:
    """Replay a recording: every frame gets the band in force after it."""
    walk = Walk.from_header(header) if header else Walk()
    out: list[LabelledFrame] = []
    offset = 0
    for f in frames:
        walk.push(f.data)
        walk.tick(f.t_ns)
        out.append(LabelledFrame(f.t_ns, offset, len(f.data), walk.band, walk.sigma, walk.k))
        offset += len(f.data)
    return out, walk


def duty_cycle(labelled: list[LabelledFrame]) -> dict[str, float]:
    """Fraction of bytes per band — the "how much did the bank harvest" number."""
    total = sum(f.length for f in labelled) or 1
    out = {b: 0 for b in BANDS}
    for f in labelled:
        out[f.band] += f.length
    return {b: n / total for b, n in out.items()}
