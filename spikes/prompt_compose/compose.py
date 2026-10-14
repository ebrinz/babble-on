"""Spike: the device writes the brief. TrueRNG bytes fill four slots of a
prompt (subject, setting, mood, form) from lists ordered roughly from
ordinary to strange, then DiffusionGemma writes the passage through the
harness sampler (every draw a tape uniform). Uses the harness's
bytes -> uniform -> index contract, so a byte stream skewed high picks from
the strange end of every list.

Run with the harness venv:
    ../../harness/.venv/bin/python compose.py <carrier.bin> <out.jsonl> --n 10
"""
import argparse
import json
import time
from pathlib import Path

SLOTS = {
    "subject": [
        "a cup of tea", "a dog", "a garden", "a bus stop", "an old chair", "a kitchen", "a bicycle", "a letter",
        "a market", "a schoolteacher", "a fisherman", "a bakery", "a violin", "a lighthouse keeper",
        "a beekeeper", "a night nurse", "a clockmaker", "a cartographer", "a beached whale", "a ghost train",
        "an abandoned observatory", "a swarm of moths", "a sentient fog", "a lighthouse with no sea",
        "a library of unwritten books", "a man made of keys", "a staircase that counts", "a moon that hums",
        "a city of glass bees", "a borrowed shadow", "the last word of a dead language", "a door into weather",
    ],
    "setting": [
        "on a quiet morning", "in a small town", "in a busy café", "at a train station", "in a suburban house",
        "on a farm", "in a hospital corridor", "at the seaside", "in a snowed-in cabin", "in a flooded basement",
        "on a night ferry", "in a closed museum", "at the edge of a desert", "inside a lighthouse",
        "on an oil rig", "in a cathedral at dawn", "beneath a frozen lake", "in a city after the evacuation",
        "on a generation ship", "inside a whale", "in the space between two clocks", "in a house that rearranges itself",
        "on the far side of a mirror", "in a forest of radio towers", "inside a dream someone else is having",
        "on a planet with three suns", "at the bottom of a well of stars", "in a museum of lost sounds",
        "inside a single raindrop", "where the map runs out", "in the year that never happened", "nowhere at all",
    ],
    "mood": [
        "calm", "warm", "cheerful", "nostalgic", "wistful", "tender", "restless", "hopeful", "bittersweet",
        "lonely", "uneasy", "playful", "solemn", "melancholy", "tense", "eerie", "feverish", "giddy",
        "grief-stricken", "furious", "dreamlike", "uncanny", "ecstatic", "menacing", "hallucinatory",
        "absurd", "sacred", "unhinged", "weightless", "apocalyptic", "rapturous", "unspeakable",
    ],
    "form": [
        "a short scene", "a diary entry", "a letter", "a short story opening", "a description", "a memory",
        "a postcard", "a recipe", "a weather report", "a eulogy", "a confession", "a lullaby in prose",
        "a field guide entry", "a ship's log", "a museum placard", "an instruction manual", "a prayer",
        "a classified advertisement", "a police report", "a dream journal", "a list of rules", "a radio broadcast",
        "an encyclopedia entry", "a ransom note", "a translation of birdsong", "a map legend",
        "a message in a bottle", "a horoscope", "a court transcript", "a spell", "a footnote to nothing",
        "the last page of a novel",
    ],
}
BYTES_PER_PROMPT = 4 * len(SLOTS)


def _uniform_to_index(u: float, n: int) -> int:
    from babble_harness.noise import uniform_to_index
    return int(uniform_to_index(u, n))


def compose(raw: bytes) -> tuple[str, dict]:
    if len(raw) < BYTES_PER_PROMPT:
        raise ValueError(f"need {BYTES_PER_PROMPT} bytes, got {len(raw)}")
    from babble_harness.noise import bytes_to_uniforms
    us = bytes_to_uniforms(raw[:BYTES_PER_PROMPT], len(SLOTS))
    picks = {name: items[_uniform_to_index(u, len(items))] for (name, items), u in zip(SLOTS.items(), us)}
    prompt = (f"Write {picks['form']} about {picks['subject']}, {picks['setting']}. "
              f"The mood is {picks['mood']}. Keep it under 200 words.")
    return prompt, picks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("carrier")
    p.add_argument("out")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--offset", type=int, default=0, help="byte offset into the carrier")
    args = p.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coherence"))
    from babble_harness.models import load_denoiser
    from babble_harness.noise import EntropyTape, budget_bytes
    from babble_harness.sampler import SamplerConfig, sample_canvas
    from harness_dose import text_of

    den = load_denoiser("mlx", capture_layers=(), capture_router=False, logit_lens=False)
    cfg = SamplerConfig(canvas_length=den.canvas_length, vocab_size=den.vocab_size)
    per = BYTES_PER_PROMPT + budget_bytes(cfg.canvas_length, cfg.max_denoising_steps)
    data = Path(args.carrier).read_bytes()
    with open(args.out, "w") as f:
        for i in range(args.n):
            chunk = data[args.offset + i * per: args.offset + (i + 1) * per]
            prompt, picks = compose(chunk)
            den.set_prompt(prompt)
            t0 = time.time()
            r = sample_canvas(den, EntropyTape(data=chunk[BYTES_PER_PROMPT:], label="compose"), cfg, keep_capture=False)
            line = {"index": i, "prompt": prompt, "picks": picks, "text": text_of(den, r.final_canvas),
                    "steps": r.n_steps, "elapsed_s": round(time.time() - t0, 1)}
            f.write(json.dumps(line) + "\n")
            f.flush()
            print(f"{i}: {line['elapsed_s']}s · {prompt}", flush=True)


if __name__ == "__main__":
    main()
