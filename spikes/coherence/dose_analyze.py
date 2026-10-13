"""Effect size per dose against the zero-dose arm, and the smallest dose the
primary metric can see."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze import METRICS, N_PERM, cohen_d, perm_test


def _by_condition(scored, manifest, metric):
    cond = {s["index"]: s["condition"] for s in manifest["samples"]}
    g = defaultdict(list)
    for line in sorted(scored, key=lambda l: l["index"]):
        v = line.get(metric)
        if v is not None:
            g[cond[line["index"]]].append(float(v))
    return {k: np.array(v) for k, v in g.items()}


def compare_doses(scored, manifest, n_perm: int = N_PERM):
    delta = {}
    for s in manifest["samples"]:
        delta.setdefault(s["condition"], s["delta"])
    order = sorted((c for c in delta if c != "d0"), key=lambda c: (delta[c] is None, delta[c] or 0.0))
    rows = []
    for metric in METRICS:
        g = _by_condition(scored, manifest, metric)
        base = g.get("d0", np.array([]))
        for c in order:
            a = g.get(c, np.array([]))
            rows.append({
                "condition": c, "delta": delta[c], "metric": metric, "n": len(a), "n0": len(base),
                "mean": float(a.mean()) if len(a) else float("nan"),
                "mean0": float(base.mean()) if len(base) else float("nan"),
                "d": cohen_d(a, base), "p": perm_test(a, base, n_perm=n_perm),
            })
    return rows


def threshold(rows, metric: str, alpha: float = 0.01):
    """Smallest numeric dose whose `metric` differs from d0 at `alpha`, else None."""
    hits = [r["delta"] for r in rows if r["metric"] == metric and r["delta"] is not None and r["p"] < alpha]
    return min(hits) if hits else None


def report(rows, alpha: float = 0.01) -> str:
    lines = []
    for metric in METRICS:
        t = threshold(rows, metric, alpha)
        lines.append(f"## {metric}  (smallest dose with p < {alpha}: {t if t is not None else 'none'})")
        lines.append("| dose | n | mean | d vs d0 | p |")
        lines.append("|---|---|---|---|---|")
        for r in rows:
            if r["metric"] != metric:
                continue
            dose = f"{r['delta']:g}" if r["delta"] is not None else "lcg"
            lines.append(f"| {dose} | {r['n']} | {r['mean']:.3f} | {r['d']:+.2f} | {r['p']:.4f} |")
        lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scored_jsonl")
    p.add_argument("manifest")
    p.add_argument("out_md")
    p.add_argument("--n-perm", type=int, default=N_PERM)
    args = p.parse_args()
    scored = [json.loads(l) for l in Path(args.scored_jsonl).read_text().splitlines() if l.strip()]
    manifest = json.loads(Path(args.manifest).read_text())
    rows = compare_doses(scored, manifest, args.n_perm)
    text = report(rows)
    Path(args.out_md).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
