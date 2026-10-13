"""Pre-registered comparisons (see the spec) over scored.jsonl, with a
permutation test, Cohen's d, and the decision rule."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ("gpt2_nll", "steps", "distinct_ratio")
COMPARISONS = (
    ("positive_control", "biased", "csprng"),
    ("null_calibration", "csprng_even", "csprng_odd"),
    ("primary", "anomaly", "ordinary"),
    ("anomaly_vs_csprng", "anomaly", "csprng"),
)
N_PERM = 20_000


def cohen_d(a, b) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def perm_test(a, b, n_perm: int = N_PERM, seed: int = 0) -> float:
    """Two-sided permutation p-value for the difference of means."""
    if len(a) == 0 or len(b) == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    observed = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    na = len(a)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if abs(pooled[:na].mean() - pooled[na:].mean()) >= observed:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def groups(scored, manifest, metric):
    """Metric values per condition (plus csprng split by rank parity) and the
    number of samples excluded for a null metric."""
    condition = {s["index"]: s["condition"] for s in manifest["samples"]}
    g = defaultdict(list)
    excluded = 0
    for line in sorted(scored, key=lambda l: l["index"]):
        v = line.get(metric)
        if v is None:
            excluded += 1
            continue
        g[condition[line["index"]]].append(float(v))
    cs = g.get("csprng", [])
    g["csprng_even"], g["csprng_odd"] = cs[0::2], cs[1::2]
    return {k: np.array(v) for k, v in g.items()}, excluded


def compare(scored, manifest, n_perm: int = N_PERM):
    results = []
    for metric in METRICS:
        g, _ = groups(scored, manifest, metric)
        empty = np.array([])
        for name, x, y in COMPARISONS:
            a, b = g.get(x, empty), g.get(y, empty)
            results.append({
                "comparison": name, "metric": metric, "n1": len(a), "n2": len(b),
                "mean1": float(a.mean()) if len(a) else float("nan"),
                "mean2": float(b.mean()) if len(b) else float("nan"),
                "d": cohen_d(a, b), "p": perm_test(a, b, n_perm=n_perm),
            })
    return results


def verdict(results) -> str:
    p = {(r["comparison"], r["metric"]): r["p"] for r in results}
    pc, nc, pr = p[("positive_control", "gpt2_nll")], p[("null_calibration", "gpt2_nll")], p[("primary", "gpt2_nll")]
    if pc >= 0.001:
        return (f"INCONCLUSIVE: positive control failed (biased vs csprng gpt2_nll p={pc:.3f}); "
                "the metric cannot detect gross bias")
    note = "" if nc > 0.05 else f" (null calibration FAILED, p={nc:.3f}; treat with suspicion)"
    if pr < 0.01:
        return (f"CANDIDATE EFFECT: anomaly vs ordinary gpt2_nll p={pr:.4f}; "
                f"needs replication on a second capture{note}")
    return f"NO DETECTABLE DIFFERENCE: anomaly vs ordinary gpt2_nll p={pr:.3f}{note}"


def report(results, verdict_line: str, excluded: int) -> str:
    lines = [
        "# Coherence spike report", "",
        f"Samples excluded for a missing primary metric: {excluded}", "",
        "| comparison | metric | n1 | n2 | mean1 | mean2 | d | p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['comparison']} | {r['metric']} | {r['n1']} | {r['n2']} | "
                     f"{r['mean1']:.4f} | {r['mean2']:.4f} | {r['d']:+.3f} | {r['p']:.4f} |")
    lines += ["", f"**Verdict:** {verdict_line}", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scored")
    p.add_argument("manifest")
    p.add_argument("out")
    args = p.parse_args()
    scored = [json.loads(l) for l in Path(args.scored).read_text().splitlines() if l.strip()]
    manifest = json.loads(Path(args.manifest).read_text())
    results = compare(scored, manifest)
    _, excluded = groups(scored, manifest, "gpt2_nll")
    text = report(results, verdict(results), excluded)
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
