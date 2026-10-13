"""Lay out a dose-response batch: the same TrueRNG carrier bytes with a known
bit bias injected at each dose, plus the app's biased LCG as the ceiling.

Dose delta means P(bit = 1) = 0.5 + delta: each bit is forced to 1 with
probability 2*delta by an independent seeded PRNG, otherwise left alone.
Conditions are interleaved round-robin so every arm spans the same stretch
of the carrier. The output is read by generate.py / score.py unchanged.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from batch_build import PROMPT, SAMPLE_BYTES, bad_rng

DOSES = (0.0, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1)


def bias_bits(data: bytes, delta: float, rng: np.random.Generator) -> bytes:
    if delta == 0.0:
        return data
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    force = rng.random(bits.size) < 2 * delta
    return np.packbits(bits | force).tobytes()


def build(carrier: bytes, per_condition: int, sample_bytes: int = SAMPLE_BYTES, seed: int = 0):
    """Round-robin d0, d1e-4, …, lcg, d0, …; the carrier is consumed front to back."""
    rng = np.random.default_rng(seed)
    conditions = [(f"d{d:g}", d) for d in DOSES] + [("lcg", None)]
    lcg = bad_rng(per_condition * sample_bytes)
    out = bytearray()
    samples = []
    offset = 0
    for i in range(per_condition):
        for name, delta in conditions:
            if delta is None:
                chunk = lcg[i * sample_bytes : (i + 1) * sample_bytes]
            else:
                raw = carrier[offset : offset + sample_bytes]
                if len(raw) < sample_bytes:
                    raise ValueError(f"carrier exhausted at sample {len(samples)} ({name})")
                offset += sample_bytes
                chunk = bias_bits(raw, delta, rng)
            out += chunk
            samples.append({"index": len(samples), "condition": name, "delta": delta})
    return bytes(out), samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("carrier", help="ordinary TrueRNG bytes, e.g. runs/2026-10-12/ordinary.bin")
    p.add_argument("outdir")
    p.add_argument("--per-condition", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    batch, samples = build(Path(args.carrier).read_bytes(), args.per_condition, seed=args.seed)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "batch.bin").write_bytes(batch)
    manifest = {
        "sample_bytes": SAMPLE_BYTES, "prompt": PROMPT, "temperature": 0.0, "max_tokens": 256,
        "carrier": args.carrier, "doses": list(DOSES), "seed": args.seed, "samples": samples,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(samples)} samples, {len(batch)} bytes -> {out / 'batch.bin'}")


if __name__ == "__main__":
    main()
