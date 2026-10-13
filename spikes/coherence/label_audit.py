"""Audit the anomaly label: do the trials the coherence walk flags differ from
ordinary trials on anything the walk does not select on?

The walk selects on the running mean bit, so `mean_bit` is reported only as a
reference for the size of that selection effect. Every other feature is held
out. Comparisons: anomalous vs all ordinary; anomalous vs the in-band trials
adjacent to each excursion (time-matched control); adjacent vs far ordinary
(lead-in / trail-out); a biased-LCG positive control; an even/odd null split.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from batch_build import bad_rng
from walk import TRIAL_BYTES, Z95, classify

FEATURES = [
    "mean_bit",        # reference only: the walk selects on this
    "ac_bit_1", "ac_bit_2", "ac_bit_8", "ac_bit_64",
    "ac_byte_1",
    "byte_chi2",
    "runs_z",
    "longest_run_1",
    "dft_z",
]
HELD_OUT = FEATURES[1:]
N_PERM = 20_000


def _autocorr(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom == 0.0:
        return 0.0
    return float((x[:-lag] * x[lag:]).sum() / denom)


def trial_features(trial: bytes) -> np.ndarray:
    b = np.frombuffer(trial, dtype=np.uint8)
    bits = np.unpackbits(b).astype(np.float64)
    n = bits.size
    pi = bits.mean()

    counts = np.bincount(b, minlength=256)
    chi2 = float(((counts - b.size / 256) ** 2 / (b.size / 256)).sum())

    # NIST runs test statistic, conditioned on the observed proportion
    changes = float((bits[1:] != bits[:-1]).sum()) + 1.0
    runs_z = (changes - 2 * n * pi * (1 - pi)) / (2 * np.sqrt(2 * n) * pi * (1 - pi)) if 0 < pi < 1 else 0.0

    # longest run of ones
    padded = np.concatenate([[0], bits, [0]])
    edges = np.flatnonzero(np.diff(padded))
    longest = int((edges[1::2] - edges[0::2]).max()) if edges.size else 0

    # NIST discrete Fourier transform test statistic
    x = 2 * bits - 1
    mags = np.abs(np.fft.rfft(x))[: n // 2]
    thresh = np.sqrt(np.log(1 / 0.05) * n)
    n1 = float((mags < thresh).sum())
    n0 = 0.95 * n / 2
    dft_z = (n1 - n0) / np.sqrt(n * 0.95 * 0.05 / 4)

    return np.array([
        pi,
        _autocorr(bits, 1), _autocorr(bits, 2), _autocorr(bits, 8), _autocorr(bits, 64),
        _autocorr(b.astype(np.float64), 1),
        chi2,
        runs_z,
        longest,
        dft_z,
    ])


def adjacent_ordinary(anomalous: np.ndarray) -> np.ndarray:
    """Indices of ordinary trials that sit right before or after an anomalous one."""
    before = np.zeros_like(anomalous)
    after = np.zeros_like(anomalous)
    before[:-1] = anomalous[1:]
    after[1:] = anomalous[:-1]
    return np.flatnonzero((before | after) & ~anomalous)


def cohen_d_cols(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(0, ddof=1) + (nb - 1) * b.var(0, ddof=1)) / (na + nb - 2))
    # a column of identical values can still show a pooled sd of ~1e-16 from
    # rounding in the mean; treat that as zero variance
    degenerate = pooled <= 1e-9 * np.maximum(np.abs(pooled_mean := np.concatenate([a, b]).mean(0)), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (a.mean(0) - b.mean(0)) / pooled
    return np.where(degenerate, 0.0, d)


def perm_matrix(a: np.ndarray, b: np.ndarray, n_perm: int = N_PERM, seed: int = 0) -> np.ndarray:
    """Two-sided permutation p-value of the mean difference, one per column."""
    rng = np.random.default_rng(seed)
    observed = np.abs(a.mean(0) - b.mean(0))
    pooled = np.concatenate([a, b])
    na = len(a)
    total = pooled.sum(0)
    hits = np.zeros(a.shape[1])
    for _ in range(n_perm):
        idx = rng.permutation(len(pooled))
        sa = pooled[idx[:na]].sum(0)
        diff = np.abs(sa / na - (total - sa) / (len(pooled) - na))
        hits += diff >= observed
    degenerate = pooled.std(0) <= 1e-9 * np.maximum(np.abs(pooled.mean(0)), 1.0)
    return np.where(degenerate, 1.0, (hits + 1) / (n_perm + 1))


def compare(a: np.ndarray, b: np.ndarray, n_perm: int = N_PERM) -> dict:
    d = cohen_d_cols(a, b)
    p = perm_matrix(a, b, n_perm)
    return {f: {"d": float(d[i]), "p": float(p[i]), "mean_a": float(a[:, i].mean()), "mean_b": float(b[:, i].mean())}
            for i, f in enumerate(FEATURES)}


def audit(data: bytes, reset_every: int | None, n_perm: int = N_PERM, seed: int = 0) -> dict:
    _, _, sigmas = classify(data, reset_every=reset_every)
    n_trials = len(sigmas)
    anomalous = np.abs(np.array(sigmas)) >= Z95
    feats = np.stack([trial_features(data[i * TRIAL_BYTES:(i + 1) * TRIAL_BYTES]) for i in range(n_trials)])

    anom_idx = np.flatnonzero(anomalous)
    ord_idx = np.flatnonzero(~anomalous)
    adj_idx = adjacent_ordinary(anomalous)
    far_idx = np.setdiff1d(ord_idx, adj_idx)
    biased = bad_rng(len(anom_idx) * TRIAL_BYTES, seed=seed)
    biased_feats = np.stack([trial_features(biased[i * TRIAL_BYTES:(i + 1) * TRIAL_BYTES]) for i in range(len(anom_idx))])

    comparisons = {
        "anomalous_vs_ordinary": compare(feats[anom_idx], feats[ord_idx], n_perm),
        "anomalous_vs_adjacent": compare(feats[anom_idx], feats[adj_idx], n_perm),
        "adjacent_vs_far_ordinary": compare(feats[adj_idx], feats[far_idx], n_perm),
        "biased_vs_ordinary": compare(biased_feats, feats[ord_idx], n_perm),
        "null_ordinary_even_vs_odd": compare(feats[ord_idx[0::2]], feats[ord_idx[1::2]], n_perm),
    }
    return {
        "trials": n_trials, "anomalous": int(len(anom_idx)), "ordinary": int(len(ord_idx)),
        "adjacent": int(len(adj_idx)), "far": int(len(far_idx)), "reset_every": reset_every,
        "n_perm": n_perm, "alpha_bonferroni": 0.01 / len(HELD_OUT), "features": FEATURES,
        "comparisons": comparisons,
    }


def report(result: dict) -> str:
    alpha = result["alpha_bonferroni"]
    lines = [f"trials {result['trials']}: anomalous {result['anomalous']}, ordinary {result['ordinary']} "
             f"(adjacent {result['adjacent']}, far {result['far']}); Bonferroni alpha {alpha:.4f}", ""]
    for name, rows in result["comparisons"].items():
        hits = [f for f in HELD_OUT if rows[f]["p"] < alpha]
        lines.append(f"## {name}  (held-out features below alpha: {', '.join(hits) or 'none'})")
        lines.append("| feature | d | p |")
        lines.append("|---|---|---|")
        for f in FEATURES:
            tag = " (selected on)" if f == "mean_bit" else ""
            lines.append(f"| {f}{tag} | {rows[f]['d']:+.3f} | {rows[f]['p']:.4f} |")
        lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("capture")
    p.add_argument("out_json")
    p.add_argument("--reset-every", type=int, default=1000)
    p.add_argument("--n-perm", type=int, default=N_PERM)
    args = p.parse_args()
    result = audit(Path(args.capture).read_bytes(), args.reset_every, args.n_perm)
    Path(args.out_json).write_text(json.dumps(result, indent=1))
    print(report(result))


if __name__ == "__main__":
    main()
