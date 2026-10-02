"""Simulated recordings (no device): a healthy PRNG stream and a biased
one, framed per 50 ms tick like the app's engine, so the whole workflow can
be dry-run and the classifier/seed-cutting exercised on known inputs.

    python -m babble_harness.cli record-sim out.bbrec --seconds 60 [--bias 0.55] [--rate 50000]
"""
from __future__ import annotations

import time

import numpy as np

from ..recording import RecordingHeader, RecordingWriter


def record_sim(path: str, seconds: float, rate_bytes_per_s: float = 50_000.0, bias: float | None = None,
               tick_s: float = 0.05, seed: int = 0, bias_from_s: float = 0.0) -> int:
    """``bias`` is the probability of a 1-bit (None/0.5 = healthy). With
    ``bias_from_s`` the stream is healthy until that time, then biased."""
    rng = np.random.default_rng(seed)
    label = "simulate" if bias in (None, 0.5) else f"bad(p1={bias})"
    header = RecordingHeader(source=label, started_at_ms=int(time.time() * 1000),
                             notes=f"simulated, rate {rate_bytes_per_s:.0f} B/s, tick {tick_s}s, seed {seed}")
    per_tick = max(1, int(rate_bytes_per_s * tick_s))
    total = 0
    t = 0.0
    with RecordingWriter.create(path, header) as w:
        while t < seconds - 1e-9:
            t += tick_s
            if bias in (None, 0.5) or t < bias_from_s:
                data = rng.bytes(per_tick)
            else:
                bits = (rng.random((per_tick, 8)) < bias).astype(np.uint8)
                data = np.packbits(bits, axis=1).reshape(-1).tobytes()
            w.frame(int(round(t * 1e9)), data)
            total += len(data)
    return total
