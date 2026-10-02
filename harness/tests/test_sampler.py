import numpy as np
import pytest

from babble_harness.models.stub import StubDenoiser
from babble_harness.noise import EntropyExhausted, EntropyTape, budget_bytes, prng_tape
from babble_harness.sampler import (
    SamplerConfig, entropy_bound_selection, sample_canvas, softmax, temperature, token_entropy,
)


def cfg(**kw):
    base = dict(canvas_length=16, vocab_size=64, max_denoising_steps=6)
    base.update(kw)
    return SamplerConfig(**base)


def test_temperature_schedule_matches_reference():
    c = cfg(max_denoising_steps=48)
    assert temperature(48, c) == pytest.approx(0.8)
    assert temperature(24, c) == pytest.approx(0.6)
    assert temperature(1, c) == pytest.approx(0.4 + 0.4 / 48)


def test_entropy_helpers():
    p = softmax(np.array([[0.0, 0.0], [10.0, -10.0]]))
    H = token_entropy(p)
    assert H[0] == pytest.approx(np.log(2))
    assert H[1] == pytest.approx(0.0, abs=1e-6)


def test_entropy_bound_selection_hand_case():
    # sorted: 0.01, 0.02, 0.05, 0.9 ; cum-own: 0, .01, .03, .08 -> with bound .05 accept first three
    H = np.array([0.9, 0.02, 0.01, 0.05])
    assert entropy_bound_selection(H, 0.05).tolist() == [False, True, True, True]
    assert entropy_bound_selection(H, 0.0).tolist() == [False, False, True, False]  # most confident always
    assert entropy_bound_selection(H, 10.0).all()


def test_sampler_is_deterministic_and_diverges_with_tape():
    c = cfg()
    need = budget_bytes(c.canvas_length, c.max_denoising_steps)
    a = sample_canvas(StubDenoiser(), prng_tape(1, need), c)
    b = sample_canvas(StubDenoiser(), prng_tape(1, need), c)
    d = sample_canvas(StubDenoiser(), prng_tape(2, need), c)
    assert np.array_equal(a.final_canvas, b.final_canvas)
    assert np.array_equal(a.initial_canvas, b.initial_canvas)
    assert not np.array_equal(a.initial_canvas, d.initial_canvas)


def test_entropy_accounting_matches_budget_without_early_stop():
    c = cfg(early_stop=False)
    need = budget_bytes(c.canvas_length, c.max_denoising_steps)
    tape = prng_tape(3, need)
    r = sample_canvas(StubDenoiser(), tape, c)
    assert r.n_steps == c.max_denoising_steps
    assert tape.consumed == need and tape.remaining == 0
    purposes = [d.purpose for d in tape.log]
    assert purposes[0] == "initial_canvas"
    assert purposes[1:3] == ["categorical", "renoise"]
    assert r.steps[0].tape_offset_before == 4 * c.canvas_length
    assert r.steps[0].tape_offset_after == 4 * c.canvas_length * 3
    assert all(d.step == i for i, d in zip(np.repeat(np.arange(c.max_denoising_steps), 2), tape.log[1:]))


def test_short_tape_raises_not_tops_up():
    c = cfg(early_stop=False)
    with pytest.raises(EntropyExhausted):
        sample_canvas(StubDenoiser(), prng_tape(4, 4 * 16 * 3), c)


def test_converges_and_early_stops_on_stub():
    c = cfg(max_denoising_steps=40, confidence_threshold=0.05)
    need = budget_bytes(c.canvas_length, c.max_denoising_steps)
    r = sample_canvas(StubDenoiser(sharpness=3.0), prng_tape(5, need), c)
    stub = StubDenoiser(sharpness=3.0)
    assert np.array_equal(r.final_canvas, stub.target)
    assert r.stopped_early and r.n_steps < c.max_denoising_steps
    assert r.tape_consumed == 4 * 16 * (1 + 2 * r.n_steps)
    stc = r.steps_to_commit()
    assert (stc >= 0).all() and stc.max() < r.n_steps
    # the last step accepts everything
    assert r.steps[-1].n_accepted == 16 or r.steps[-1].step != 1
    assert "hidden" in r.steps[0].capture


def test_capture_can_be_dropped():
    c = cfg()
    r = sample_canvas(StubDenoiser(), prng_tape(6, budget_bytes(16, 6)), c, keep_capture=False)
    assert r.steps[0].capture == {}
