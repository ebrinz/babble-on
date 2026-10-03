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
from .figures import duty_figure, write_probe_figures, write_run_figures
from .noise import EntropyExhausted, EntropyTape, budget_bytes
from .recording import read_recording
from .sampler import SamplerConfig, sample_canvas
from .segments import prng_seeds, seeds_from_recording


def build_conditions(cfg: SamplerConfig, recordings: list[str], remote: list[str], seed_dirs: list[str],
                     n_prng: int, max_per_group: int | None, allow_concat: bool, prng_base: int = 0,
                     log=print) -> tuple[dict[str, list[EntropyTape]], list[str]]:
    """Returns (conditions with at least one seed, the recordings consulted)."""
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
    return {k: v for k, v in conds.items() if v}, list(recordings) + list(remote)


def run_experiment(out_dir: str | Path, denoiser, cfg: SamplerConfig, conditions: dict[str, list[EntropyTape]],
                   prompts: list[str], model_name: str, save_activations: bool = True, log=print,
                   recordings: list[str] | None = None) -> Path:
    out = Path(out_dir)
    (out / "activations").mkdir(parents=True, exist_ok=True)
    recordings = recordings or []
    manifest = {
        "model": model_name, "sampler": asdict(cfg), "prompts": prompts,
        "budget_bytes": budget_bytes(cfg.canvas_length, cfg.max_denoising_steps),
        "conditions": {k: len(v) for k, v in conditions.items()}, "recordings": list(recordings),
        "started_at": time.time(), "samples": 0, "skipped": 0,
    }
    n = 0
    with open(out / "samples.jsonl", "w") as f:
        for pi, prompt in enumerate(prompts):
            if hasattr(denoiser, "set_prompt"):
                denoiser.set_prompt(prompt)
            for cond, tapes in conditions.items():
                for ti, tape in enumerate(tapes):
                    tape.pos = 0; tape.log.clear()
                    sid = f"{cond}-{ti:03d}-p{pi}"  # reproducible, collision-free
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
                                     "accepted_mask": ["".join("1" if a else "0" for a in s.accepted) for s in res.steps],
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
                log=print, log_file: str | Path | None = None) -> dict:
    run = Path(run_dir)
    rows = [json.loads(l) for l in open(run / "samples.jsonl") if l.strip()]
    manifest = json.loads((run / "manifest.json").read_text()) if (run / "manifest.json").exists() else {}
    conds = sorted({r["condition"] for r in rows})
    keys = sorted({k for r in rows for k in r["scalars"]})
    by = {c: [r for r in rows if r["condition"] == c] for c in conds}
    report: dict = {"run": str(run), "model": manifest.get("model"), "n": {c: len(v) for c, v in by.items()},
                    "scalars": {}, "tests": {}, "probe": None}
    probe_nulls: dict = {}
    model = manifest.get("model", "?")
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(manifest.get("finished_at", time.time())))
    figs = write_run_figures(run, rows, title="babble-on", subtitle=f"{run.name} · {model} · {when}")
    activations: dict[str, dict] = {}  # id → loaded npz (read once, used by every probe)

    def load_act(r):
        if r["id"] not in activations:
            p = run / "activations" / f"{r['id']}.npz"
            activations[r["id"]] = dict(np.load(p)) if p.exists() else {}
        return activations[r["id"]]
    lines = [f"![{run.name}]({figs['header']})", "", f"# Run report — `{run.name}`", "",
             f"model `{model}` · {when} · samples: " + ", ".join(f"{c}={len(v)}" for c, v in by.items()), ""]
    if manifest.get("sampler"):
        sp = manifest["sampler"]
        lines += [f"sampler: canvas {sp.get('canvas_length')} · ≤ {sp.get('max_denoising_steps')} steps · "
                  f"T {sp.get('t_max')}→{sp.get('t_min')} · entropy bound {sp.get('entropy_bound')} · "
                  f"early stop {'on' if sp.get('early_stop') else 'off'} · budget {manifest.get('budget_bytes')} bytes/sample", ""]
    for path in manifest.get("recordings", []):
        try:
            from .coherence import duty_cycle, label_frames
            header, frames = read_recording(path)
            labelled, walk = label_frames(frames, header)
            d = duty_cycle(labelled)
            name = Path(path).stem
            (run / "report-assets" / f"duty-{name}.svg").write_text(duty_figure(d, header.source))
            lines += [f"![duty cycle {name}](report-assets/duty-{name}.svg)", "",
                      f"`{path}`: {sum(f.length for f in labelled):,} bytes, {walk.k} trials, final σ {walk.sigma:+.2f} ({walk.band})", ""]
        except (OSError, ValueError, KeyError) as e:  # recording moved or unreadable since the run
            lines += [f"_recording `{path}` not readable for the duty-cycle figure: {e}_", ""]
    if rows:
        lines += ["## Crystallisation", "", f"![entropy]({figs['entropy']})", "", f"![accepted]({figs['accepted']})", ""]
        for c in conds:
            if f"raster-{c}" in figs:
                lines += [f"![acceptance raster {c}]({figs[f'raster-{c}']})", ""]
        for c in conds:
            if f"commit-{c}" in figs:
                lines += [f"![commit order {c}]({figs[f'commit-{c}']})", ""]
        for k in sorted(k for k in figs if k.startswith("steps-")):
            lines += [f"![steps]({figs[k]})", ""]
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
        # Probe 1: the scalar feature vector (low-dimensional, well powered).
        Xs, ys = [], []
        for c, lab in ((a, 1), (b, 0)):
            for r in by[c]:
                Xs.append([r["scalars"].get(k, float("nan")) for k in keys]); ys.append(lab)
        Xs = np.array(Xs, dtype=np.float64)
        if Xs.size:
            Xs = Xs[:, ~np.isnan(Xs).any(axis=0)]  # drop scalars undefined for some sample (e.g. commit stats)
            Xs = Xs[:, Xs.std(axis=0) > 0]  # and constants
        report["probes"] = {}
        lines += ["", "## Linear probes (mass-mean, cross-validated, permutation null)", ""]
        if len(Xs) >= 6 and len(set(ys)) == 2 and Xs.shape[1] > 0:
            pr = probe_with_permutation_null(Xs, np.array(ys), n_perm=n_perm)
            report["probes"]["scalars"] = pr.summary()
            probe_nulls["scalars"] = (np.array(pr.null), pr.accuracy)
            lines.append(f"- **scalar features** ({pr.dim} dims): accuracy **{pr.accuracy:.3f}** vs null {pr.null_mean:.3f} ± {pr.null_sd:.3f} "
                         f"(p = {pr.p_value:.3g}, n = {pr.n})")
        # Probe 2: pooled residual-stream activations, first and last step.
        for key, label in (("pooled_first", "first-step pooled activations"), ("pooled_last", "last-step pooled activations")):
            X, y = [], []
            for c, lab in ((a, 1), (b, 0)):
                for r in by[c]:
                    z = load_act(r)
                    if key in z:
                        X.append(z[key]); y.append(lab)
            if len(X) >= 6 and len(set(y)) == 2:
                pr = probe_with_permutation_null(np.stack(X), np.array(y), n_perm=n_perm)
                report["probes"][key] = pr.summary()
                probe_nulls[key] = (np.array(pr.null), pr.accuracy)
                lines.append(f"- **{label}** ({pr.dim} dims): accuracy **{pr.accuracy:.3f}** vs null {pr.null_mean:.3f} ± {pr.null_sd:.3f} "
                             f"(p = {pr.p_value:.3g}, n = {pr.n}, |mean diff| = {pr.mean_diff_norm:.3g})")
            else:
                lines.append(f"- {label}: skipped (need ≥ 6 samples with activations across both conditions)")
        report["probe"] = report["probes"].get("pooled_first")
        if probe_nulls:
            pf = write_probe_figures(run, probe_nulls)
            lines.append("")
            for name in probe_nulls:
                lines += [f"![probe null {name}]({pf[f'probe-{name}']})", ""]
    lines += ["", "## Reading the numbers", "",
              "- `accepted_step0`, `commit_step_mean`: how fast the canvas crystallises — the direct analogue of the Plaid heat map.",
              "- `entropy_auc`, `n_steps`: how much uncertainty the model carried before early stopping.",
              "- `router_entropy_mean`: how spread expert routing was; `logit_lens_depth`: how deep the final answer appeared.",
              "- The probe answers: can anything linear in the residual stream tell the conditions apart? Compare against `prng` and `remote` before believing a hit.",
              "- Multiple comparisons: with ~15 scalars, expect ~1 false positive at p < 0.05 by chance."]
    (run / "report.md").write_text("\n".join(lines) + "\n")
    (run / "report.json").write_text(json.dumps(report, indent=1, default=float))
    entry = log_entry(run, report, pair, when)
    target = find_log_file(run, log_file)
    if target:
        with open(target, "a") as f:
            f.write(entry)
        log(f"logged to {target}")
    log("\n".join(lines))
    return report


def find_log_file(run: Path, explicit: str | Path | None = None) -> Path | None:
    """`BABBLE_EXPERIMENT_LOG`, an explicit path, or the repo's
    docs/experiments/LOG.md found by walking up from the run directory (and
    from this package). `BABBLE_EXPERIMENT_LOG=0` disables logging."""
    import os
    env = os.environ.get("BABBLE_EXPERIMENT_LOG")
    if env == "0":
        return None
    if explicit:
        return Path(explicit)
    if env:
        return Path(env)
    for start in (run.resolve(), Path(__file__).resolve()):
        for d in [start, *start.parents]:
            cand = d / "docs" / "experiments" / "LOG.md"
            if cand.exists():
                return cand
    return None


def log_entry(run: Path, report: dict, pair: tuple[str, str], when: str) -> str:
    a, b = pair
    sig = [(k, v) for k, v in report.get("tests", {}).items() if v.get("p") is not None and v["p"] < 0.05]
    sig.sort(key=lambda kv: kv[1]["p"])
    counts = ", ".join(f"{c}={n}" for c, n in report.get("n", {}).items())
    lines = [f"\n## {when} · `{run.name}` · model `{report.get('model', '?')}`", "",
             f"- conditions: {counts}; compared `{a}` vs `{b}`"]
    if sig:
        lines.append("- scalars that moved (p < 0.05): " + "; ".join(
            f"`{k}` {v['median_a']:.3g} → {v['median_b']:.3g} (p={v['p']:.2g})" for k, v in sig[:6]))
    else:
        lines.append("- no scalar differed at p < 0.05")
    for name, pr in (report.get("probes") or {}).items():
        lines.append(f"- probe `{name}`: accuracy {pr['accuracy']:.2f} vs null {pr['null_mean']:.2f}±{pr['null_sd']:.2f} (p={pr['p_value']:.2g}, n={pr['n']})")
    lines.append(f"- report: `{run}/report.md` (copy to docs/experiments/runs/ to keep it)")
    lines.append("- outcome: _(fill in: what this means for the hypothesis, and what to try next)_")
    return "\n".join(lines) + "\n"
