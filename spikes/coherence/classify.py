"""Split a raw TrueRNG capture into anomaly and ordinary byte pools."""
import argparse
import json
import sys
from pathlib import Path

from walk import TRIAL_BYTES, Z95, classify


def check_pools(anomaly_len: int, ordinary_len: int, min_bytes: int) -> list[str]:
    """Names of the pools shorter than `min_bytes` (empty list means OK)."""
    pools = (("anomaly", anomaly_len), ("ordinary", ordinary_len))
    return [name for name, length in pools if length < min_bytes]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("capture")
    p.add_argument("outdir")
    p.add_argument("--min-bytes", type=int, default=100 * 51_200)
    p.add_argument("--reset-every", type=int, default=None,
                   help="restart the walk after this many trials (fresh app session)")
    args = p.parse_args()

    data = Path(args.capture).read_bytes()
    anomaly, ordinary, sigmas = classify(data, reset_every=args.reset_every)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "anomaly.bin").write_bytes(anomaly)
    (out / "ordinary.bin").write_bytes(ordinary)

    n_anom = sum(abs(s) >= Z95 for s in sigmas)
    summary = {
        "capture_bytes": len(data),
        "trial_bytes": TRIAL_BYTES,
        "reset_every": args.reset_every,
        "trials": len(sigmas),
        "anomalous_trials": n_anom,
        "anomalous_fraction": n_anom / len(sigmas) if sigmas else 0.0,
        "anomaly_bytes": len(anomaly),
        "ordinary_bytes": len(ordinary),
        "final_sigma": sigmas[-1] if sigmas else 0.0,
        "sigma_trace": [round(s, 3) for s in sigmas],
    }
    (out / "classify.json").write_text(json.dumps(summary, indent=1))
    print(f"{len(sigmas)} trials, {n_anom} anomalous ({summary['anomalous_fraction']:.1%}), "
          f"final sigma {summary['final_sigma']:+.2f}")
    print(f"anomaly {len(anomaly)} B, ordinary {len(ordinary)} B")

    short = check_pools(len(anomaly), len(ordinary), args.min_bytes)
    if short:
        print(f"ERROR: pool(s) below {args.min_bytes} bytes: {', '.join(short)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
