import numpy as np
import pytest

from analyze import cohen_d, compare, groups, perm_test, verdict


def test_cohen_d_known_value():
    assert cohen_d(np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0])) == pytest.approx(-1.0)


def test_perm_test_detects_a_shift():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 50), rng.normal(1, 1, 50)
    assert perm_test(a, b, n_perm=2000) < 0.001


def test_perm_test_is_quiet_on_same_distribution():
    rng = np.random.default_rng(2)
    a, b = rng.normal(0, 1, 50), rng.normal(0, 1, 50)
    assert perm_test(a, b, n_perm=2000) > 0.05


def _manifest(n_per):
    conds = ("anomaly", "ordinary", "csprng", "biased")
    return {"samples": [{"index": i, "condition": conds[i % 4]} for i in range(4 * n_per)]}


def test_groups_splits_csprng_by_rank_parity():
    m = _manifest(4)
    scored = [{"index": i, "gpt2_nll": float(i)} for i in range(16)]
    g, excluded = groups(scored, m, "gpt2_nll")
    assert excluded == 0
    # csprng sits at global indices 2, 6, 10, 14: split by rank, not global parity
    assert g["csprng_even"].tolist() == [2.0, 10.0]
    assert g["csprng_odd"].tolist() == [6.0, 14.0]


def test_groups_excludes_null_and_counts():
    m = _manifest(2)
    scored = [{"index": i, "gpt2_nll": None if i == 0 else 1.0} for i in range(8)]
    g, excluded = groups(scored, m, "gpt2_nll")
    assert excluded == 1 and len(g["anomaly"]) == 1


def _results(pc, nc, pr):
    rows = {("positive_control", pc), ("null_calibration", nc), ("primary", pr), ("anomaly_vs_csprng", 0.5)}
    return [{"comparison": c, "metric": "gpt2_nll", "p": p} for c, p in rows]


def test_verdict_inconclusive_when_positive_control_fails():
    assert verdict(_results(pc=0.2, nc=0.5, pr=0.001)).startswith("INCONCLUSIVE")


def test_verdict_candidate_effect():
    v = verdict(_results(pc=0.0001, nc=0.5, pr=0.004))
    assert v.startswith("CANDIDATE EFFECT") and "suspicion" not in v


def test_verdict_no_difference_with_null_warning():
    v = verdict(_results(pc=0.0001, nc=0.01, pr=0.4))
    assert v.startswith("NO DETECTABLE DIFFERENCE") and "suspicion" in v


def test_compare_covers_every_comparison_and_metric():
    m = _manifest(10)
    rng = np.random.default_rng(3)
    scored = [{"index": i, "gpt2_nll": float(rng.normal()), "steps": 12, "distinct_ratio": 0.7} for i in range(40)]
    results = compare(scored, m, n_perm=200)
    assert len(results) == 4 * 3
    assert {r["comparison"] for r in results} == {"positive_control", "null_calibration", "primary", "anomaly_vs_csprng"}
