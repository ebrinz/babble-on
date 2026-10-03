"""``python -m babble_harness.cli <command>`` — the harness entry point."""
from __future__ import annotations

import argparse
import json
import sys

from .sampler import SamplerConfig


def _cfg(a) -> SamplerConfig:
    return SamplerConfig(canvas_length=a.canvas, vocab_size=a.vocab, max_denoising_steps=a.steps,
                         t_max=a.t_max, t_min=a.t_min, entropy_bound=a.entropy_bound,
                         confidence_threshold=a.confidence, stability_threshold=a.stability,
                         early_stop=not a.no_early_stop)


def _add_sampler_args(p):
    p.add_argument("--steps", type=int, default=48)
    p.add_argument("--canvas", type=int, default=256)
    p.add_argument("--vocab", type=int, default=262_144)
    p.add_argument("--t-max", type=float, default=0.8)
    p.add_argument("--t-min", type=float, default=0.4)
    p.add_argument("--entropy-bound", type=float, default=0.1)
    p.add_argument("--confidence", type=float, default=0.005)
    p.add_argument("--stability", type=int, default=1)
    p.add_argument("--no-early-stop", action="store_true")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="babble-harness")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("label", help="replay the coherence walk over a .bbrec and print band duty cycle + seed counts")
    p.add_argument("recording")
    p.add_argument("--seed-bytes", type=int, default=None, help="default: budget for --steps/--canvas")
    _add_sampler_args(p)

    p = sub.add_parser("run", help="run conditions × seeds × prompts")
    p.add_argument("out")
    p.add_argument("--model", default="stub", help="stub | tiny | diffusion_gemma")
    p.add_argument("--recording", action="append", default=[], help=".bbrec from the app or record-serial")
    p.add_argument("--remote", action="append", default=[], help=".bbrec from fetch-anu (remote control arm)")
    p.add_argument("--seed-dir", action="append", default=[], help="directory of app-exported *.seed.json bundles")
    p.add_argument("--prng", type=int, default=0, help="number of PRNG control seeds")
    p.add_argument("--max-per-group", type=int, default=None)
    p.add_argument("--allow-concat", action="store_true", help="join run tails to cut more seeds (weaker provenance)")
    p.add_argument("--prompt", action="append", default=[], help="repeatable; default: one empty prompt")
    p.add_argument("--no-activations", action="store_true")
    p.add_argument("--layers", default="6,12,18,24,30", help="decoder layers to capture (diffusion_gemma)")
    p.add_argument("--model-id", default=None)
    p.add_argument("--quant", default=None, help="none | nvfp4 | bnb4 (diffusion_gemma)")
    _add_sampler_args(p)

    p = sub.add_parser("analyze", help="write report.md/.json (+ SVG figures) for a run and append to the experiment log")
    p.add_argument("run")
    p.add_argument("--pair", nargs=2, default=("in_band", "out_band"))
    p.add_argument("--perm", type=int, default=200)
    p.add_argument("--log-file", default=None, help="experiment log to append to (default: docs/experiments/LOG.md; BABBLE_EXPERIMENT_LOG=0 disables)")

    p = sub.add_parser("fetch-anu", help="bulk-fetch ANU QRNG bytes into a .bbrec (needs ANU_API_KEY)")
    p.add_argument("out")
    p.add_argument("--bytes", type=int, required=True)
    p.add_argument("--pause", type=float, default=0.0)

    p = sub.add_parser("record-serial", help="capture a TrueRNG serial device to .bbrec without the app")
    p.add_argument("device"); p.add_argument("out")
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("--baud", type=int, default=9600)

    p = sub.add_parser("record-sim", help="write a simulated .bbrec (healthy or biased) for dry runs")
    p.add_argument("out")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--rate", type=float, default=50_000.0, help="bytes/s (TrueRNG V3 ≈ 50000)")
    p.add_argument("--bias", type=float, default=None, help="P(bit=1); omit for a healthy stream")
    p.add_argument("--bias-from", type=float, default=0.0, help="seconds of healthy stream before the bias kicks in")
    p.add_argument("--sim-seed", type=int, default=0)

    p = sub.add_parser("power", help="how many seeds per condition the tests need, and the stream that costs")
    p.add_argument("--effects", default="0.3,0.5,0.8,1.2")
    p.add_argument("--dim", type=int, default=64, help="pooled activation dimension for the probe simulation")
    p.add_argument("--rate", type=float, default=50_000.0, help="device bytes/s for the time estimate")
    p.add_argument("--quick", action="store_true")
    _add_sampler_args(p)

    p = sub.add_parser("serve", help="sidecar mode (newline JSON on stdio) for the app")
    p.add_argument("--model", default="diffusion_gemma")
    p.add_argument("--model-id", default=None)

    a = ap.parse_args(argv)
    log = lambda *x: print(*x, file=sys.stderr, flush=True)  # noqa: E731

    if a.cmd == "label":
        from .coherence import duty_cycle, label_frames
        from .noise import budget_bytes
        from .recording import read_recording
        from .segments import contiguous_runs
        header, frames = read_recording(a.recording)
        labelled, walk = label_frames(frames, header)
        need = a.seed_bytes or budget_bytes(a.canvas, a.steps)
        runs = contiguous_runs(labelled)
        total = sum(f.length for f in labelled)
        seeds = {g: sum(r.length // need for r in runs if r.band_group == g) for g in ("in_band", "out_band")}
        print(json.dumps({"source": header.source, "bytes": total, "frames": len(frames), "trials": walk.k,
                          "final_sigma": walk.sigma, "final_band": walk.band, "duty_cycle": duty_cycle(labelled),
                          "runs": len(runs), "seed_bytes": need, "seeds_available": seeds}, indent=1))
        return 0

    if a.cmd == "run":
        from .models import load_denoiser
        from .noise import budget_bytes
        from .runner import analyze_run, build_conditions, run_experiment
        cfg = _cfg(a)
        kw = {}
        if a.model == "stub":
            kw = dict(vocab_size=cfg.vocab_size, canvas_length=cfg.canvas_length)
        elif a.model == "tiny":
            kw = {}
        else:
            kw = dict(capture_layers=tuple(int(x) for x in a.layers.split(",") if x))
            if a.model_id:
                kw["model_id"] = a.model_id
            if a.quant:
                kw["quant"] = a.quant
        den = load_denoiser(a.model, **kw)
        if hasattr(den, "quantized_fraction"):
            log(f"quantized parameter share: {den.quantized_fraction():.2f}")
        if hasattr(den, "canvas_length"):
            cfg.canvas_length, cfg.vocab_size = den.canvas_length, den.vocab_size
        conds, recs = build_conditions(cfg, a.recording, a.remote, a.seed_dir, a.prng, a.max_per_group, a.allow_concat, log=log)
        if not conds:
            log("no seeds: pass --recording/--remote/--seed-dir and/or --prng N (a recording shorter than one "
                f"seed budget of {budget_bytes(cfg.canvas_length, cfg.max_denoising_steps)} bytes yields none)"); return 2
        out = run_experiment(a.out, den, cfg, conds, a.prompt or [""], a.model, not a.no_activations, log=log, recordings=recs)
        analyze_run(out, log=log)
        return 0

    if a.cmd == "analyze":
        from .runner import analyze_run
        analyze_run(a.run, tuple(a.pair), a.perm, log=log, log_file=a.log_file)
        return 0

    if a.cmd == "fetch-anu":
        from .sources.anu import fetch_to_recording
        fetch_to_recording(a.out, a.bytes, pause_s=a.pause, log=log)
        return 0

    if a.cmd == "record-serial":
        from .sources.serial_device import record_serial
        record_serial(a.out, a.device, a.seconds, a.baud, log=log)
        return 0

    if a.cmd == "record-sim":
        from .sources.simulate import record_sim
        n = record_sim(a.out, a.seconds, a.rate, a.bias, seed=a.sim_seed, bias_from_s=a.bias_from)
        log(f"wrote {n} bytes to {a.out}")
        return 0

    if a.cmd == "power":
        from .power import format_table, power_table
        rows = power_table([float(x) for x in a.effects.split(",")], dim=a.dim, rate=a.rate,
                           canvas=a.canvas, steps=a.steps, quick=a.quick)
        print(format_table(rows, a.rate))
        return 0

    if a.cmd == "serve":
        from .serve import serve
        serve(a.model, **({"model_id": a.model_id} if a.model_id else {}))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
