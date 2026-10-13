"""Step 2: can entropy move the *subject*, not just the wording?

Sweeps prompt x temperature, generating N times per cell with fresh TrueRNG
bytes each time. Reports mean pairwise word similarity within each cell and
dumps every text to a JSON file for reading.
"""
import argparse
import difflib
import itertools
import json
import time

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

from entropy import EntropyPool, TrueRNG, entropy_driven

MODEL = "mlx-community/diffusiongemma-26B-A4B-it-4bit"

PROMPTS = {
    "passage": ("Write a short passage of prose.", True),
    "minimal": ("Write.", True),
    "empty": ("", True),
    "raw-empty": ("", False),  # no chat template at all
    # hand-over prompts: can the prompt delegate the subject choice to entropy?
    "handover": ("Pick any subject you like and write a short passage about it.", True),
    "handover-wild": ("Choose a surprising subject at random and write a short passage of prose about it.", True),
}


def similarity(a, b):
    return difflib.SequenceMatcher(None, a.split(), b.split()).ratio()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--temperatures", default="0,0.7,1.0")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--out", default="sweep_results.json")
    p.add_argument("--prompts", default=",".join(PROMPTS), help="comma-separated subset of prompt names")
    args = p.parse_args()

    model, processor = load(MODEL)
    config = load_config(MODEL)
    trng = TrueRNG()
    temps = [float(t) for t in args.temperatures.split(",")]

    results = []
    selected = [(n, PROMPTS[n]) for n in args.prompts.split(",")]
    for (name, (text, templated)), temp in itertools.product(selected, temps):
        prompt = apply_chat_template(processor, config, text, num_images=0) if templated else text
        texts = []
        for i in range(args.runs):
            t0 = time.perf_counter()
            try:
                with entropy_driven(EntropyPool(trng)):
                    out = generate(model, processor, prompt, max_tokens=args.max_tokens, temperature=temp).text
            except Exception as e:  # keep sweeping; report the failure
                out = f"<<ERROR: {e!r}>>"
            texts.append(out)
            print(f"\n--- {name} T={temp} #{i} ({time.perf_counter() - t0:.1f}s) ---\n{out}", flush=True)
        sims = [similarity(a, b) for a, b in itertools.combinations(texts, 2)]
        mean = sum(sims) / len(sims) if sims else float("nan")
        results.append({"prompt": name, "temperature": temp, "texts": texts, "mean_similarity": mean})
        print(f"\n### {name} T={temp}: mean pairwise similarity {mean:.2f}", flush=True)
        json.dump(results, open(args.out, "w"), indent=1)

    print("\n\nprompt       T     sim")
    for r in results:
        print(f"{r['prompt']:<12} {r['temperature']:<5} {r['mean_similarity']:.2f}")


if __name__ == "__main__":
    main()
