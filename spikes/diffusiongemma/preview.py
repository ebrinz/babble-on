"""Streaming preview test: does mlx-vlm hand us the canvas after every
denoising step, live, so the app's crystallizing "boil" can survive the swap?

The public `generate`/`stream_generate` batch the per-step drafts and release
them only after generation finishes, so a sidecar must drive the generic
`stream_diffusion_generate` generator directly (which is what the model's own
generate() wraps). This script does that and prints the timeline.
"""
import argparse
import importlib
import time

from mlx_vlm import load
from mlx_vlm.generate.dispatch import _prepare_generation_inputs
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

from entropy import EntropyPool, TrueRNG, entropy_driven

dg = importlib.import_module("mlx_vlm.generate.diffusion")
MODEL = "mlx-community/diffusiongemma-26B-A4B-it-4bit"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="Write a short passage of prose.")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    args = p.parse_args()

    model, processor = load(MODEL)
    config = load_config(MODEL)
    prompt = apply_chat_template(processor, config, args.prompt, num_images=0)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    input_ids, pixel_values, mask, _ = _prepare_generation_inputs(model, processor, prompt, None, None, None, {})

    events = []
    t0 = time.perf_counter()
    with entropy_driven(EntropyPool(TrueRNG())):
        gen = dg.stream_diffusion_generate(
            model, processor, tokenizer, input_ids, pixel_values, mask,
            max_tokens=args.max_tokens, skip_special_token_ids=[],
            temperature=args.temperature, diffusion_show_unmasking=True,
            diffusion_unmasking_interval=1, diffusion_unmasking_width=0,
        )
        for r in gen:
            events.append((time.perf_counter() - t0, r.is_draft, r.diffusion_step,
                           r.diffusion_total_steps, r.draft_text, r.text, r.finish_reason))

    drafts = [e for e in events if e[1]]
    finals = [e for e in events if not e[1]]
    print(f"\n{len(events)} events in {events[-1][0]:.1f}s: {len(drafts)} drafts, {len(finals)} token/final results")
    if len(drafts) > 1:
        gaps = [b[0] - a[0] for a, b in zip(drafts, drafts[1:])]
        print(f"draft cadence: first at {drafts[0][0]:.1f}s, mean gap {sum(gaps)/len(gaps):.2f}s, max gap {max(gaps):.2f}s")
    print("\ndraft timeline (step/total, revealed chars, text so far):")
    for t, _, step, total, draft, _, _ in drafts:
        print(f"  {t:5.1f}s  step {step:>2}/{total}  {len(draft.strip()):>4} chars  {draft.strip()[:90]!r}")
    print("\nfinal text:\n" + "".join(e[5] for e in finals))


if __name__ == "__main__":
    main()
