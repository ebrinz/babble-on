"""Step 1: is the output a pure function of TrueRNG bytes?

  A  live TrueRNG pool P1 (record every byte consumed)
  B  live TrueRNG pool P2             -> differs from A  (entropy steers)
  C  replay of A's recorded bytes     -> identical to A, even though B has
                                         since advanced mlx's global PRNG
Repeated at temperature 0 and 0.7 (the latter also exercises the sampler PRNG).
"""
import argparse
import difflib
import time

import mlx.core as mx
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

from entropy import EntropyPool, TrueRNG, entropy_driven

MODEL = "mlx-community/diffusiongemma-26B-A4B-it-4bit"


def run(model, processor, prompt, pool, temperature, max_tokens):
    t0 = time.perf_counter()
    with entropy_driven(pool):
        result = generate(model, processor, prompt, max_tokens=max_tokens, temperature=temperature)
    return result.text, time.perf_counter() - t0


def similarity(a, b):
    return difflib.SequenceMatcher(None, a.split(), b.split()).ratio()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="Write a short passage of prose.")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperatures", default="0,0.7")
    args = p.parse_args()

    model, processor = load(MODEL)
    config = load_config(MODEL)
    prompt = apply_chat_template(processor, config, args.prompt, num_images=0)
    trng = TrueRNG()

    run(model, processor, prompt, EntropyPool(trng), 0.0, args.max_tokens)  # warm-up

    ok = True
    for temp in [float(t) for t in args.temperatures.split(",")]:
        print(f"\n===== temperature {temp} =====")
        pa = EntropyPool(trng)
        a, ta = run(model, processor, prompt, pa, temp, args.max_tokens)
        print(f"--- A ({ta:.1f}s, {len(pa.consumed)} TrueRNG bytes) ---\n{a}")
        pb = EntropyPool(trng)
        b, tb = run(model, processor, prompt, pb, temp, args.max_tokens)
        print(f"--- B ({tb:.1f}s) ---\n{b}")
        mx.random.seed(424242)  # perturb global PRNG state before the replay
        mx.eval(mx.random.normal((64,)))
        c, tc = run(model, processor, prompt, EntropyPool(replay=bytes(pa.consumed)), temp, args.max_tokens)
        sab, sac = similarity(a, b), similarity(a, c)
        print(f"\nsimilarity to A:  B (fresh entropy) {sab:.2f}   C (replayed bytes) {sac:.2f}")
        if c != a:
            ok = False
            print("FAIL: replay did not reproduce A")
    print("\nPURE FUNCTION OF ENTROPY:", "yes" if ok else "NO")


if __name__ == "__main__":
    main()
