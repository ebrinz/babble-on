import numpy as np

from babble_harness.analysis import (
    cv_probe_accuracy, mann_whitney, pooled_activation, probe_with_permutation_null, sample_scalars, text_scalars,
)
from babble_harness.models.stub import StubDenoiser
from babble_harness.noise import budget_bytes, prng_tape
from babble_harness.sampler import SamplerConfig, sample_canvas


def test_mann_whitney_detects_shift_and_not_null():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 60), rng.normal(1.5, 1, 60)
    r = mann_whitney(a, b)
    assert r.p < 1e-6 and r.median1 < r.median2
    same = mann_whitney(rng.normal(0, 1, 60), rng.normal(0, 1, 60))
    assert same.p > 0.01
    assert mann_whitney([], [1.0]).n1 == 0
    assert mann_whitney([1, 1, 2], [1, 2, 2]).p > 0.1  # ties handled


def test_probe_separable_vs_null():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 10))
    y = np.array([1] * 40 + [0] * 40)
    Xs = X.copy()
    Xs[:40, 0] += 4.0
    assert cv_probe_accuracy(Xs, y) > 0.9
    assert cv_probe_accuracy(Xs, y, method="ridge") > 0.85
    # a 3-sigma shift in one of 16 dims with 15 per class: the mass-mean probe stays useful
    Xw = rng.normal(size=(30, 16)); Xw[:15, 0] += 3.0
    assert cv_probe_accuracy(Xw, np.array([1] * 15 + [0] * 15)) > 0.75
    r = probe_with_permutation_null(Xs, y, n_perm=30)
    assert r.accuracy > 0.9 and r.p_value < 0.05 and abs(r.null_mean - 0.5) < 0.15
    rn = probe_with_permutation_null(X, y, n_perm=30)
    assert rn.p_value > 0.05


def test_sample_scalars_and_pooling_on_stub():
    cfg = SamplerConfig(canvas_length=16, vocab_size=64, max_denoising_steps=8)
    r = sample_canvas(StubDenoiser(), prng_tape(1, budget_bytes(16, 8)), cfg)
    s = sample_scalars(r)
    assert s["n_steps"] == r.n_steps and s["accepted_mean"] > 0
    assert "router_entropy_mean" in s and "logit_lens_depth" in s
    h = pooled_activation(r, layer_index=-1, step_index=0)
    assert h.shape == (8,)
    t = text_scalars(r.final_canvas)
    assert 0 < t["distinct1"] <= 1
    assert text_scalars(np.array([])) == {"distinct1": 0.0, "distinct2": 0.0, "repeat_frac": 0.0}
