"""Dose-response through the harness sampler (MLX backend).

Same batch layout as generate.py (batch.bin + manifest.json from
dose_build.py), but every random draw goes through the harness's
`sample_canvas`: the initial canvas, one categorical draw per position per
step at the Transformers temperature schedule (0.8 -> 0.4), and the renoise,
each a tape uniform. Run with the harness venv:

    ../../harness/.venv/bin/python harness_dose.py runs/<id>/batch.bin runs/<id>/manifest.json runs/<id>/batch.jsonl

Resumable: samples already written without an error are skipped; failed
lines are dropped and retried, so rerunning the command finishes the batch.
Output lines carry what score.py reads: index, text, consumed, steps,
elapsed_s, error.
"""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path


def _compact(out: Path) -> set:
    """Drop failed lines; return the indices already done."""
    if not out.exists():
        return set()
    ok = [l for l in out.read_text().splitlines() if l.strip() and json.loads(l).get("error") is None]
    out.write_text("".join(l + "\n" for l in ok))
    return {json.loads(l)["index"] for l in ok}


def text_of(den, ids) -> str:
    if hasattr(den, "trim_after_eos"):
        ids = den.trim_after_eos(ids)
    if not hasattr(den, "decode"):
        return ""
    text = den.decode(ids, skip_special=False)  # keep the markers so clean_output can find them
    proc = getattr(den, "processor", None)
    if proc is not None and hasattr(proc, "clean_output"):
        text = proc.clean_output(text)  # strips the empty thought-channel scaffolding
    return text.strip()


def run(den, cfg, data: bytes, manifest: dict, out, limit: int | None = None, log=print) -> None:
    from babble_harness.noise import EntropyExhausted, EntropyTape, budget_bytes
    from babble_harness.sampler import sample_canvas

    sb = manifest["sample_bytes"]
    need = budget_bytes(cfg.canvas_length, cfg.max_denoising_steps)
    if sb < need:
        raise ValueError(f"sample_bytes {sb} is below the sampler's worst-case budget {need}")
    out = Path(out)
    done = _compact(out)
    todo = [s for s in manifest["samples"] if s["index"] not in done][:limit]
    log(f"{len(done)} done, {len(todo)} to go")
    with open(out, "a") as f:
        for s in todo:
            i = s["index"]
            chunk = data[i * sb:(i + 1) * sb]
            if len(chunk) != sb:
                raise ValueError(f"sample {i}: batch holds {len(chunk)} bytes, need {sb}")
            tape = EntropyTape(data=chunk, label=s["condition"])
            t0 = time.time()
            try:
                r = sample_canvas(den, tape, cfg, keep_capture=False)
                line = {"index": i, "text": text_of(den, r.final_canvas), "consumed": r.tape_consumed,
                        "steps": r.n_steps, "error": None}
            except (RuntimeError, EntropyExhausted) as e:
                line = {"index": i, "text": "", "consumed": tape.consumed, "steps": None, "error": str(e)}
            line["elapsed_s"] = round(time.time() - t0, 1)
            f.write(json.dumps(line) + "\n")
            f.flush()
            status = f"ERROR {line['error']}" if line["error"] else f"{line['steps']} steps, {line['consumed']} B"
            log(f"sample {i} ({s['condition']}): {status}, {line['elapsed_s']}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("batch")
    p.add_argument("manifest")
    p.add_argument("out")
    p.add_argument("--model", default="mlx")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    from babble_harness.models import load_denoiser
    from babble_harness.sampler import SamplerConfig

    manifest = json.loads(Path(args.manifest).read_text())
    den = load_denoiser(args.model, capture_layers=(), capture_router=False, logit_lens=False)
    cfg = SamplerConfig(canvas_length=den.canvas_length, vocab_size=den.vocab_size)
    den.set_prompt(manifest["prompt"])
    meta = {"model": args.model, "backend": den.describe() if hasattr(den, "describe") else None,
            "sampler": asdict(cfg), "prompt_ids": den.prompt_ids}
    (Path(args.out).parent / "harness.json").write_text(json.dumps(meta, indent=1))
    run(den, cfg, Path(args.batch).read_bytes(), manifest, args.out, args.limit, log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
