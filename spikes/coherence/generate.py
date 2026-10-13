"""Generate one DiffusionGemma text per sample, each a pure function of its
reservoir bytes, appending to a JSONL file so a crash costs only a restart."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusiongemma"))
from entropy import EntropyPool, entropy_driven  # noqa: E402

MODEL = "mlx-community/diffusiongemma-26B-A4B-it-4bit"


def slice_for(index: int, sample_bytes: int, data: bytes) -> bytes:
    s = data[index * sample_bytes : (index + 1) * sample_bytes]
    if len(s) != sample_bytes:
        raise ValueError(f"sample {index}: batch file holds {len(s)} bytes, need {sample_bytes}")
    return s


def done_indices(jsonl_path) -> set:
    path = Path(jsonl_path)
    if not path.exists():
        return set()
    return {json.loads(line)["index"] for line in path.read_text().splitlines() if line.strip()}


def run_sample(slice_bytes: bytes, generate_text) -> dict:
    """Run `generate_text(pool)` with every random draw coming from `slice_bytes`.

    An exhausted reservoir (or any RuntimeError from the model) is recorded as
    an error; the sample is never silently re-run on software noise.
    """
    pool = EntropyPool(replay=slice_bytes)
    t0 = time.perf_counter()
    try:
        with entropy_driven(pool):
            text = generate_text(pool)
        error = None
    except RuntimeError as e:
        text, error = "", str(e)
    return {
        "text": text,
        "consumed": len(pool.consumed),
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "error": error,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("batch")
    p.add_argument("manifest")
    p.add_argument("out")
    p.add_argument("--limit", type=int, default=None, help="stop after N new samples")
    args = p.parse_args()

    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    manifest = json.loads(Path(args.manifest).read_text())
    data = Path(args.batch).read_bytes()
    done = done_indices(args.out)
    todo = [s for s in manifest["samples"] if s["index"] not in done]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"{len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return

    model, processor = load(MODEL)
    config = load_config(MODEL)
    prompt = apply_chat_template(processor, config, manifest["prompt"], num_images=0)

    def generate_text(_pool):
        return generate(
            model, processor, prompt,
            max_tokens=manifest["max_tokens"], temperature=manifest["temperature"],
        ).text

    with open(args.out, "a") as out:
        for s in todo:
            r = run_sample(slice_for(s["index"], manifest["sample_bytes"], data), generate_text)
            out.write(json.dumps({"index": s["index"], **r}) + "\n")
            out.flush()
            status = f"ERROR {r['error']}" if r["error"] else f"{r['consumed']} B, {r['elapsed_s']}s"
            print(f"sample {s['index']}: {status}", flush=True)


if __name__ == "__main__":
    main()
