"""Lay out the four conditions' entropy into one round-robin batch file.

Each sample owns a fixed SAMPLE_BYTES reservoir (worst case for 48 denoising
steps is 50 180 bytes). The generator reads only batch.bin + sample_bytes;
condition labels live in manifest.json.
"""
import argparse
import json
import os
from pathlib import Path

SAMPLE_BYTES = 51_200
CONDITIONS = ("anomaly", "ordinary", "csprng", "biased")
PROMPT = "Write a short passage of prose."


def bad_rng(n: int, seed: int = 0x12345678) -> bytes:
    """The app's deliberately biased source (src-tauri/src/source.rs run_software)."""
    x = seed
    out = bytearray(n)
    for i in range(n):
        x = (x * 1103515245 + 12345) & 0xFFFFFFFF
        out[i] = ((x >> 16) & 0xFF) & 0x3F
    return bytes(out)


def build(pools: dict, per_condition: int, sample_bytes: int = SAMPLE_BYTES):
    """Round-robin A, B, C, D, A, B, … Each pool is consumed front to back."""
    offsets = {c: 0 for c in CONDITIONS}
    out = bytearray()
    samples = []
    for i in range(per_condition):
        for c in CONDITIONS:
            start = offsets[c]
            chunk = pools[c][start : start + sample_bytes]
            if len(chunk) < sample_bytes:
                raise ValueError(
                    f"{c} pool exhausted at sample {i} (need {sample_bytes}, have {len(chunk)})"
                )
            offsets[c] = start + sample_bytes
            out += chunk
            samples.append({"index": len(samples), "condition": c})
    return bytes(out), samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("anomaly")
    p.add_argument("ordinary")
    p.add_argument("outdir")
    p.add_argument("--per-condition", type=int, default=100)
    args = p.parse_args()

    n = args.per_condition * SAMPLE_BYTES
    pools = {
        "anomaly": Path(args.anomaly).read_bytes(),
        "ordinary": Path(args.ordinary).read_bytes(),
        "csprng": os.urandom(n),
        "biased": bad_rng(n),
    }
    batch, samples = build(pools, args.per_condition)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "batch.bin").write_bytes(batch)
    manifest = {
        "sample_bytes": SAMPLE_BYTES,
        "prompt": PROMPT,
        "temperature": 0.0,
        "max_tokens": 256,
        "samples": samples,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(samples)} samples, {len(batch)} bytes -> {out / 'batch.bin'}")


if __name__ == "__main__":
    main()
