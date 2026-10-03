"""Build the report's SVG figures from a run directory (samples.jsonl +
activations/*.npz) and from a labelled recording."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import svg


def _by_condition(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["condition"], []).append(r)
    return dict(sorted(out.items()))


def _mean_curve(rows: list[dict], key: str, max_len: int) -> tuple[list[float], list[float], list[float]]:
    """Mean/min/max of a per-step series across samples, padded with NaN."""
    arr = np.full((len(rows), max_len), np.nan)
    for i, r in enumerate(rows):
        v = r["per_step"][key]
        arr[i, :len(v)] = v
    with np.errstate(all="ignore"):
        return (np.nanmean(arr, 0).tolist(), np.nanmin(arr, 0).tolist(), np.nanmax(arr, 0).tolist())


def entropy_figure(rows: list[dict]) -> str:
    by = _by_condition(rows)
    n = max(len(r["per_step"]["mean_entropy"]) for r in rows)
    series, bands = {}, {}
    for c, rs in by.items():
        m, lo, hi = _mean_curve(rs, "mean_entropy", n)
        series[c], bands[c] = m, (lo, hi)
    return svg.line_chart(series, "Mean token entropy per denoising step", "mean across samples; wash = min–max",
                          y_label="entropy (nats)", bands=bands, y_min=0.0)


def accepted_figure(rows: list[dict]) -> str:
    by = _by_condition(rows)
    n = max(len(r["per_step"]["accepted"]) for r in rows)
    series = {c: _mean_curve(rs, "accepted", n)[0] for c, rs in by.items()}
    return svg.line_chart(series, "Tokens accepted per step", "entropy-bound selection; mean across samples",
                          y_label="accepted positions", y_min=0.0)


def commit_heatmap(rows: list[dict], condition: str, max_samples: int = 40) -> str:
    rs = [r for r in rows if r["condition"] == condition][:max_samples]
    if not rs:
        return svg.heatmap([], f"Steps to commit — {condition}")
    mat = [[(float(v) if v >= 0 else None) for v in r["steps_to_commit"]] for r in rs]
    order = [[(int(v) if v >= 0 else 0) for v in r["steps_to_commit"]] for r in rs]
    vmax = max((v for row in mat for v in row if v is not None), default=1.0)
    return svg.heatmap(mat, f"Steps to commit — {condition}", f"{len(rs)} samples (rows) × canvas positions (columns); "
                       "cells appear in the order the model committed them", x_label="canvas position",
                       y_label="sample", vmin=0.0, vmax=max(vmax, 1.0), legend=("early", "late"), commit_order=order,
                       empty_note="no position kept its final token from an accepted step onward (the model never stabilised)")


def acceptance_raster(rows: list[dict], condition: str, sample_index: int = 0) -> str:
    """One sample's accepted mask per step (rows = steps, columns = positions):
    the canvas crystallising. Needs `per_step.accepted_mask` in the row."""
    rs = [r for r in rows if r["condition"] == condition]
    if not rs or "accepted_mask" not in rs[0]["per_step"]:
        return svg.heatmap([], f"Acceptance raster — {condition}")
    r = rs[min(sample_index, len(rs) - 1)]
    masks = r["per_step"]["accepted_mask"]
    mat = [[1.0 if ch == "1" else 0.15 for ch in m] for m in masks]
    order = [[i for _ in m] for i, m in enumerate(masks)]
    return svg.heatmap(mat, f"Acceptance per step — {condition}", f"sample `{r['id']}`: rows = denoising steps, columns = "
                       "canvas positions; bright = accepted by the entropy bound, dim = renoised",
                       x_label="canvas position", y_label="step", vmin=0.0, vmax=1.0, legend=("renoised", "accepted"),
                       commit_order=order)


def steps_figures(rows: list[dict]) -> list[str]:
    by = _by_condition(rows)
    cats = sorted({r["n_steps"] for r in rows})
    series = {}
    for c, rs in by.items():
        counts = {k: 0 for k in cats}
        for r in rs:
            counts[r["n_steps"]] += 1
        series[c] = [counts[k] / len(rs) for k in cats]
    # one small columns chart per condition (grouped lines over the step axis would mislead)
    return [svg.columns([str(k) for k in cats], v, f"Steps run before stopping — {c}", "share of samples",
                        color=svg.color_for(c, i), x_label="steps", y_label="share", h=220)
            for i, (c, v) in enumerate(series.items())]


def probe_null_figure(null: np.ndarray, observed: float, title: str) -> str:
    edges = np.linspace(0, 1, 21)
    counts, _ = np.histogram(null, bins=edges)
    mids = (edges[:-1] + edges[1:]) / 2
    return svg.columns([f"{m:.2f}" for m in mids], counts.tolist(), title,
                       "permutation null of cross-validated accuracy; gold line = observed",
                       marker=(observed, f"observed {observed:.2f}"), cat_positions=mids.tolist(),
                       x_label="accuracy", y_label="permutations")


def duty_figure(duty: dict[str, float], source: str) -> str:
    return svg.band_meter(duty, "Share of recorded bytes by coherence band", f"source: {source}")


def _write(run_dir: str | Path, files: dict[str, str]) -> dict[str, str]:
    assets = Path(run_dir) / "report-assets"
    assets.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, content in files.items():
        (assets / f"{name}.svg").write_text(content)
        out[name] = f"report-assets/{name}.svg"
    return out


def write_run_figures(run_dir: str | Path, rows: list[dict], title: str = "babble-on", subtitle: str = "") -> dict[str, str]:
    """Write the header and the sample figures into `<run>/report-assets/`;
    returns name → relative path."""
    files = {"header": svg.header(title, subtitle)}
    if rows:
        files["entropy"] = entropy_figure(rows)
        files["accepted"] = accepted_figure(rows)
        for c in sorted({r["condition"] for r in rows}):
            files[f"commit-{c}"] = commit_heatmap(rows, c)
            files[f"raster-{c}"] = acceptance_raster(rows, c)
        for i, fig in enumerate(steps_figures(rows)):
            files[f"steps-{i}"] = fig
    return _write(run_dir, files)


def write_probe_figures(run_dir: str | Path, probe_nulls: dict[str, tuple[np.ndarray, float]]) -> dict[str, str]:
    return _write(run_dir, {f"probe-{name}": probe_null_figure(null, obs, f"Probe null — {name}")
                            for name, (null, obs) in probe_nulls.items()})
