"""Offline port of the app's coherence walk (src-tauri/src/stats.rs).

The app finalises a trial every 100 ms with at least 2048 bits; offline there
is no clock, so a trial is a fixed TRIAL_BYTES (what 100 ms delivers at the
TrueRNG's ~50 KB/s). z-score, walk, sigma and the 95% threshold are verbatim.
"""
import numpy as np

Z95 = 1.959_963_98
TRIAL_BYTES = 5120


def trial_z(trial: bytes) -> float:
    """Monobit z for one trial: mean n/2, variance n/4."""
    n = len(trial) * 8
    ones = int(np.unpackbits(np.frombuffer(trial, dtype=np.uint8)).sum())
    return (ones - n / 2) / (n / 4) ** 0.5


def classify(data: bytes, trial_bytes: int = TRIAL_BYTES, reset_every: int | None = None):
    """Split `data` into (anomaly, ordinary, sigmas).

    Trial i's bytes are anomalous when |sigma| >= Z95 after trial i, where
    sigma = C_k / sqrt(k) and C_k is the running sum of z-scores. A trailing
    partial trial is dropped. `reset_every` restarts the walk (C_k = 0, k = 0)
    after that many trials, like a fresh app session: a single long walk
    stays inside the envelope almost all the time because sigma is correlated
    over a span proportional to k.
    """
    anomaly, ordinary, sigmas = bytearray(), bytearray(), []
    cum = 0.0
    k = 0
    for start in range(0, len(data) - trial_bytes + 1, trial_bytes):
        trial = data[start : start + trial_bytes]
        if reset_every and k == reset_every:
            cum, k = 0.0, 0
        cum += trial_z(trial)
        k += 1
        sigma = cum / k ** 0.5
        sigmas.append(sigma)
        (anomaly if abs(sigma) >= Z95 else ordinary).extend(trial)
    return bytes(anomaly), bytes(ordinary), sigmas
