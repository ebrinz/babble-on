import numpy as np

from dose_analyze import compare_doses, threshold


def _fake(deltas_per_cond, n, shift):
    """scored lines + manifest where gpt2_nll rises by shift*delta."""
    rng = np.random.default_rng(0)
    samples, scored, i = [], [], 0
    for cond, delta in deltas_per_cond.items():
        for _ in range(n):
            samples.append({"index": i, "condition": cond, "delta": delta})
            v = rng.normal(5.0 + shift * (0.5 if delta is None else delta), 0.2)
            scored.append({"index": i, "gpt2_nll": v, "steps": 12, "distinct_ratio": 0.7})
            i += 1
    return scored, {"samples": samples}


def test_compare_doses_has_one_row_per_dose_and_metric_against_d0():
    scored, manifest = _fake({"d0": 0.0, "d0.01": 0.01, "d0.1": 0.1, "lcg": None}, 30, 10.0)
    rows = compare_doses(scored, manifest, n_perm=500)
    assert {(r["condition"], r["metric"]) for r in rows} == {
        (c, m) for c in ("d0.01", "d0.1", "lcg") for m in ("gpt2_nll", "steps", "distinct_ratio")
    }
    big = next(r for r in rows if r["condition"] == "d0.1" and r["metric"] == "gpt2_nll")
    assert big["p"] < 0.01 and big["d"] > 1


def test_threshold_is_the_smallest_significant_dose():
    scored, manifest = _fake({"d0": 0.0, "d0.0001": 1e-4, "d0.01": 0.01, "d0.1": 0.1, "lcg": None}, 30, 10.0)
    rows = compare_doses(scored, manifest, n_perm=500)
    assert threshold(rows, "gpt2_nll", alpha=0.01) == 0.01


def test_threshold_is_none_when_nothing_moves():
    scored, manifest = _fake({"d0": 0.0, "d0.1": 0.1}, 30, 0.0)
    rows = compare_doses(scored, manifest, n_perm=500)
    assert threshold(rows, "gpt2_nll", alpha=0.01) is None
