"""Power analysis: how many seeds per condition does the experiment need?

Two detectors are simulated under a Gaussian model:
  scalar  Mann-Whitney on one per-sample scalar with standardised effect d
  probe   ridge probe on D-dimensional activations whose class means differ
          by a vector of norm delta (isotropic unit noise), permutation p-value
Both report the smallest n per group reaching the target power, and the
stream the out-band arm would cost at the observatory's ~5 % out-band duty
cycle.
"""
from __future__ import annotations

import numpy as np

from .analysis import cv_probe_accuracy, mann_whitney
from .noise import budget_bytes

OUT_BAND_DUTY = 0.05


def mw_power(d: float, n: int, alpha: float = 0.05, sims: int = 400, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(sims):
        a = rng.normal(0, 1, n)
        b = rng.normal(d, 1, n)
        hits += mann_whitney(a, b).p < alpha
    return hits / sims


def probe_power(delta: float, n: int, dim: int = 64, alpha: float = 0.05, sims: int = 60, n_perm: int = 60,
                seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    shift = np.zeros(dim)
    shift[0] = delta
    hits = 0
    for _ in range(sims):
        X = np.vstack([rng.normal(0, 1, (n, dim)), rng.normal(0, 1, (n, dim)) + shift])
        y = np.array([1] * n + [0] * n)
        acc = cv_probe_accuracy(X, y)
        null = np.array([cv_probe_accuracy(X, rng.permutation(y)) for _ in range(n_perm)])
        p = (np.sum(null >= acc) + 1) / (n_perm + 1)
        hits += p < alpha
    return hits / sims


def seeds_needed(power_fn, effect: float, target: float = 0.8, grid=(6, 8, 10, 15, 20, 30, 40, 60, 80, 120, 160, 240),
                 **kw) -> tuple[int | None, dict[int, float]]:
    curve = {}
    for n in grid:
        p = power_fn(effect, n, **kw)
        curve[n] = p
        if p >= target:
            return n, curve
    return None, curve


def stream_cost(n_seeds: int, canvas: int = 256, steps: int = 48, duty: float = OUT_BAND_DUTY,
                rate_bytes_per_s: float = 50_000.0) -> dict[str, float]:
    seed = budget_bytes(canvas, steps)
    raw = n_seeds * seed / duty
    return {"seed_bytes": seed, "stream_bytes": raw, "stream_mib": raw / 2**20, "minutes_at_rate": raw / rate_bytes_per_s / 60}


def power_table(effects=(0.3, 0.5, 0.8, 1.2), target: float = 0.8, dim: int = 64, rate: float = 50_000.0,
                canvas: int = 256, steps: int = 48, quick: bool = False) -> list[dict]:
    rows = []
    for d in effects:
        n_mw, _ = seeds_needed(mw_power, d, target, sims=150 if quick else 400)
        n_pr, _ = seeds_needed(probe_power, d, target, grid=(6, 8, 10, 15, 20, 30, 40, 60),
                               dim=dim, sims=20 if quick else 60, n_perm=30 if quick else 60)
        n = max([x for x in (n_mw, n_pr) if x is not None], default=None)
        cost = stream_cost(n, canvas, steps, rate_bytes_per_s=rate) if n else None
        rows.append({"effect": d, "n_mann_whitney": n_mw, "n_probe": n_pr, "n_recommended": n, "cost": cost})
    return rows


def format_table(rows: list[dict], rate: float) -> str:
    out = ["| effect size | n (Mann-Whitney) | n (probe) | out-band seeds needed | stream to record | at device rate |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        c = r["cost"]
        if c:
            out.append(f"| {r['effect']:.1f} | {r['n_mann_whitney'] or '>240'} | {r['n_probe'] or '>60'} | "
                       f"{r['n_recommended']} | {c['stream_mib']:.0f} MiB | {c['minutes_at_rate']:.0f} min |")
        else:
            out.append(f"| {r['effect']:.1f} | {r['n_mann_whitney'] or '>240'} | {r['n_probe'] or '>60'} | — | — | — |")
    out.append("")
    out.append(f"effect size: standardised mean shift (d) of a per-sample scalar, or the norm of the activation mean-difference "
               f"in unit-noise units; power target 0.8 at alpha 0.05; stream cost assumes {OUT_BAND_DUTY:.0%} out-band duty "
               f"and {rate/1000:.0f} KB/s (TrueRNG V3 ≈ 50, TrueRNGpro V2 ≈ 400).")
    return "\n".join(out)
