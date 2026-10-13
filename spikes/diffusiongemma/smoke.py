"""DiffusionGemma smoke test: does it run on this Mac, and does entropy steer it?

Two sources of randomness feed mlx-vlm's DiffusionGemma sampler:
  1. the initial canvas of random tokens, which we can supply via
     `decoder_input_ids` (this is where TrueRNG bytes would go), and
  2. a software re-noise of not-yet-accepted positions on every step
     (`mx.random`, controlled by `seed`).

Four runs of one prompt separate the two:
  A  canvas E1, seed 0
  B  canvas E1, seed 0   -> same as A?  (determinism)
  C  canvas E2, seed 0   -> differs from A?  (the canvas steers the output)
  D  canvas E1, seed 1   -> differs from A?  (per-step PRNG noise alone)

Entropy comes from os.urandom here; swapping in TrueRNG bytes is a later step.
"""
import argparse
import difflib
import os
import time

import mlx.core as mx
import numpy as np
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

MODEL = "mlx-community/diffusiongemma-26B-A4B-it-4bit"
CANVAS = 256  # DiffusionGemma's canvas length


def bytes_to_canvas(raw, vocab_size, length=CANVAS):
    """4 bytes per token -> token id uniform over the vocab (u32 * V >> 32)."""
    u32 = np.frombuffer(raw[: length * 4], dtype="<u4").astype(np.uint64)
    ids = (u32 * np.uint64(vocab_size)) >> np.uint64(32)
    return mx.array(ids.astype(np.int32)[None, :])


def run(model, processor, prompt, canvas, seed, args):
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    result = generate(
        model,
        processor,
        prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        decoder_input_ids=canvas,
        seed=seed,
    )
    return result.text, time.perf_counter() - t0, mx.get_peak_memory() / 1e9


def similarity(a, b):
    return difflib.SequenceMatcher(None, a.split(), b.split()).ratio()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="Write a short passage of prose.")
    p.add_argument("--max-tokens", type=int, default=CANVAS)
    p.add_argument("--temperature", type=float, default=0.0)
    args = p.parse_args()

    t0 = time.perf_counter()
    model, processor = load(MODEL)
    config = load_config(MODEL)
    print(f"load: {time.perf_counter() - t0:.1f}s")

    vocab_size = int(model.config.text_config.vocab_size)
    prompt = apply_chat_template(processor, config, args.prompt, num_images=0)
    e1 = bytes_to_canvas(os.urandom(CANVAS * 4), vocab_size)
    e2 = bytes_to_canvas(os.urandom(CANVAS * 4), vocab_size)

    # Throwaway run so compile/warm-up cost doesn't land on run A's timing.
    run(model, processor, prompt, e1, 123, args)

    runs = {
        "A": (e1, 0),
        "B": (e1, 0),
        "C": (e2, 0),
        "D": (e1, 1),
    }
    texts = {}
    for name, (canvas, seed) in runs.items():
        text, secs, peak = run(model, processor, prompt, canvas, seed, args)
        texts[name] = text
        print(f"\n--- {name}  ({secs:.1f}s, peak {peak:.1f} GB) ---\n{text}")

    print("\nword-level similarity to A:")
    print(f"  B (same canvas, same seed): {similarity(texts['A'], texts['B']):.2f}  expect 1.00")
    print(f"  C (new canvas,  same seed): {similarity(texts['A'], texts['C']):.2f}")
    print(f"  D (same canvas, new seed):  {similarity(texts['A'], texts['D']):.2f}")


if __name__ == "__main__":
    main()
