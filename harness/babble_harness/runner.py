"""Experiment runner: conditions × seeds × prompts → a run directory.

    runs/<name>/
      manifest.json         config, model, conditions, counts, timings
      samples.jsonl         one line per sample: condition, prompt, text,
                            token ids, scalar features, tape provenance
      activations/<id>.npz  per-step captures (hidden, router, logit lens)
      report.md / .json     written by `analyze_run`
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .analysis import (
    mann_whitney, pooled_activation, probe_with_permutation_null, sample_scalars, text_scalars,
)
from .noise import EntropyExhausted, EntropyTape, budget_bytes
from .recording import read_recording
from .sampler import SamplerConfig, sample_canvas
from .segments import prng_seeds, seeds_from_recording


def build_conditions(cfg: SamplerConfig, recordings: list[str], remote: list[str], seed_dirs: list[str],
                     n_prng: int, max_per_group: int | None, allow_concat: bool, prng_base: int = 0,
                     log=print) -> dict[str, list[EntropyTape]]:
    need = budget_bytes(cfg.canvas_length, cfg.max_denoising_steps)
    conds: dict[str, list[EntropyTape]] = {"in_band": [], "out_band": [], "prng": [], "remote": []}
    for path in recordings:
        header, frames = read_recording(path)
        got = seeds_from_recording(header, frames, need, max_per_group, allow_concat)
        for g, tapes in got.items():
            for t in tapes:
                t.meta["recording"] = path
            conds[g].extend(tapes)
        log(f"{path}: {sum(len(f.data) for f in frames)} bytes → in_band {len(got['in_band'])}, out_band {len(got['out_band'])} seeds")
    for path in remote:
        header, frames = read_recording(path)
        got = seeds_from_recording(header, frames, need, max_per_group, allow_concat, label_prefix="remote_")
        tapes = got["in_band"] + got["out_band"]
        for t in tapes:
            t.label = "remote"
            t.meta["recording"] = path
        conds["remote"].extend(tapes[:max_per_group] if max_per_group else tapes)
        log(f"{path}: remote → {len(conds['remote'])} seeds")
    for d in seed_dirs:
        for j in sorted(Path(d).glob("*.seed.json")):
            t = EntropyTape.from_seed(j)
            if len(t.data) < need:
                log(f"skip {j.name}: {len(t.data)} bytes < budget {need}")
                continue
            key = {"out_band": "out_band", "live": "in_band", "in_band": "in_band"}.get(t.label, "mixed")
            conds.setdefault(key, []).append(t)
    if n_prng:
        conds["prng"] = prng_seeds(n_prng, need, prng_base)
    return {k: v for k, v in conds.items() if v}


def run_experiment(out_dir: str | Path, denoiser, cfg: SamplerConfig, conditions: dict[str, list[EntropyTape]],
                   prompts: list[str], model_name: str, save_activations: bool = True, log=print) -> Path:
    out = Path(out_dir)
    (out / "activations").mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": model_name, "sampler": asdict(cfg), "prompts": prompts,
        "budget_bytes": budget_bytes(cfg.canvas_length, cfg.max_denoising_steps),
        "conditions": {k: len(v) for k, v in conditions.items()}, "started_at": time.time(), "samples": 0, "skipped": 0,
    }
    n = 0
    with open(out / "samples.jsonl", "w") as f:
        for prompt in prompts:
            if hasattr(denoiser, "set_prompt"):
                denoiser.set_prompt(prompt)
            for cond, tapes in conditions.items():
                for ti, tape in enumerate(tapes):
                    tape.pos = 0; tape.log.clear()
                    sid = f"{cond}-{ti:03d}-{abs(hash(prompt)) % 10_000:04d}"
                    t0 = time.time()
                    try:
                        res = sample_canvas(denoiser, tape, cfg, keep_capture=True)
                    except EntropyExhausted as e:
                        manifest["skipped"] += 1
                        log(f"skip {sid}: {e}")
                        continue
                    dt = time.time() - t0
                    ids = res.final_canvas
                    text = denoiser.decode(denoiser.trim_after_eos(ids)) if hasattr(denoiser, "decode") else ""
                    row = {
                        "id": sid, "condition": cond, "prompt": prompt, "tape_label": tape.label, "tape_meta": tape.meta,
                        "text": text, "ids": ids.tolist(), "initial_ids": res.initial_canvas.tolist(),
                        "n_steps": res.n_steps, "stopped_early": res.stopped_early, "elapsed_s": round(dt, 3),
                        "scalars": {**sample_scalars(res), **text_scalars(ids)},
                        "per_step": {"accepted": [s.n_accepted for s in res.steps],
                                     "mean_entropy": [s.mean_entropy for s in res.steps],
                                     "temperature": [s.temperature for s in res.steps]},
                        "steps_to_commit": res.steps_to_commit().tolist(),
                        "provenance": res.provenance,
                    }
                    f.write(json.dumps(row) + "\n")
                    if save_activations:
                        arrays = {}
                        for s in res.steps:
                            for k, v in s.capture.items():
                                arrays[f"step{s.index:03d}_{k}"] = np.asarray(v)
                        pooled = pooled_activation(res, -1, 0)
                        if pooled is not None:
                            arrays["pooled_first"] = pooled
                            arrays["pooled_last"] = pooled_activation(res, -1, -1)
                        np.savez_compressed(out / "activations" / f"{sid}.npz", **arrays)
                    n += 1
                    log(f"{sid}: {res.n_steps} steps, {dt:.1f}s, {len(text)} chars")
    manifest["samples"] = n
    manifest["finished_at"] = time.time()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return out


def analyze_run(run_dir: str | Path, pair: tuple[str, str] = ("in_band", "out_band"), n_perm: int = 200,
                log=print) -> dict:
    run = Path(run_dir)
    rows = [json.loads(l) for l in open(run / "samples.jsonl") if l.strip()]
    conds = sorted({r["condition"] for r in rows})
    keys = sorted({k for r in rows for k in r["scalars"]})
    by = {c: [r for r in rows if r["condition"] == c] for c in conds}
    report: dict = {"run": str(run), "n": {c: len(v) for c, v in by.items()}, "scalars": {}, "tests": {}, "probe": None}
    lines = [f"# babble-on run report — `{run.name}`", "", f"samples: " + ", ".join(f"{c}={len(v)}" for c, v in by.items()), ""]
    lines += ["## Per-condition medians", "", "| scalar | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    for k in keys:
        meds = []
        for c in conds:
            vals = [r["scalars"][k] for r in by[c] if r["scalars"].get(k) is not None and not np.isnan(r["scalars"][k])]
            m = float(np.median(vals)) if vals else float("nan")
            report["scalars"].setdefault(k, {})[c] = m
            meds.append(f"{m:.4g}")
        lines.append(f"| {k} | " + " | ".join(meds) + " |")
    a, b = pair
    if a in by and b in by:
        lines += ["", f"## {a} vs {b} (Mann–Whitney, two-sided)", "", "| scalar | median " + a + " | median " + b + " | z | p |", "|---|---|---|---|---|"]
        for k in keys:
            r = mann_whitney([x["scalars"].get(k, float("nan")) for x in by[a]], [x["scalars"].get(k, float("nan")) for x in by[b]])
            report["tests"][k] = {"z": r.z, "p": r.p, "median_a": r.median1, "median_b": r.median2, "n": [r.n1, r.n2]}
            flag = " **" if r.p < 0.01 else (" *" if r.p < 0.05 else "")
            lines.append(f"| {k} | {r.median1:.4g} | {r.median2:.4g} | {r.z:+.2f} | {r.p:.3g}{flag} |")
        X, y = [], []
        for c, lab in ((a, 1), (b, 0)):
            for r in by[c]:
                p = run / "activations" / f"{r['id']}.npz"
                if p.exists():
                    z = np.load(p)
                    if "pooled_first" in z:
                        X.append(z["pooled_first"]); y.append(lab)
        if len(X) >= 6 and len(set(y)) == 2:
            pr = probe_with_permutation_null(np.stack(X), np.array(y), n_perm=n_perm)
            report["probe"] = asdict(pr)
            lines += ["", "## Linear probe on pooled first-step activations", "",
                      f"cross-validated accuracy **{pr.accuracy:.3f}** vs permutation null {pr.null_mean:.3f} ± {pr.null_sd:.3f} "
                      f"(p = {pr.p_value:.3g}, n = {pr.n}, dim = {pr.dim}, |mean diff| = {pr.mean_diff_norm:.3g})"]
        else:
            lines += ["", "_probe skipped: need ≥ 6 samples with activations across both conditions_"]
    lines += ["", "## Reading the numbers", "",
              "- `accepted_step0`, `commit_step_mean`: how fast the canvas crystallises — the direct analogue of the Plaid heat map.",
              "- `entropy_auc`, `n_steps`: how much uncertainty the model carried before early stopping.",
              "- `router_entropy_mean`: how spread expert routing was; `logit_lens_depth`: how deep the final answer appeared.",
              "- The probe answers: can anything linear in the residual stream tell the conditions apart? Compare against `prng` and `remote` before believing a hit.",
              "- Multiple comparisons: with ~15 scalars, expect ~1 false positive at p < 0.05 by chance."]
    (run / "report.md").write_text("\n".join(lines) + "\n")
    (run / "report.json").write_text(json.dumps(report, indent=1, default=float))
    log("\n".join(lines))
    return report
