import json
from pathlib import Path

import numpy as np
import pytest

from babble_harness.noise import (
    EntropyExhausted, EntropyTape, budget_bytes, bytes_to_uniforms,
    categorical_inverse_cdf, prng_tape, uniform_to_index,
)

CONTRACT = Path(__file__).resolve().parents[2] / "docs" / "contract" / "noise_vectors.json"


def test_contract_vectors():
    v = json.loads(CONTRACT.read_text())
    raw = bytes.fromhex(v["bytes_hex"])
    u = bytes_to_uniforms(raw, len(v["uniforms"]))
    np.testing.assert_allclose(u, v["uniforms"], rtol=0, atol=1e-15)
    assert uniform_to_index(u, v["index_n"]).tolist() == v["indices"]


def test_uniform_open_interval_little_endian():
    u = bytes_to_uniforms(b"\x00\x00\x00\x00\xff\xff\xff\xff\x01\x00\x00\x00", 3)
    assert 0 < u[0] < u[2] < u[1] < 1
    assert u[2] == pytest.approx(1.5 / 2**32, abs=1e-18)
    with pytest.raises(ValueError):
        bytes_to_uniforms(b"\x00\x00\x00", 1)


def test_index_clamps():
    assert uniform_to_index(np.array([0.0, 0.35, 0.9999999999, 1.0]), 10).tolist() == [0, 3, 9, 9]


def test_categorical_inverse_cdf_exact_and_unnormalised():
    probs = np.array([[0.1, 0.2, 0.7], [0.5, 0.5, 0.0]])
    u = np.array([0.05, 0.99])
    assert categorical_inverse_cdf(probs, u).tolist() == [0, 1]
    assert categorical_inverse_cdf(probs, np.array([0.1 + 1e-9, 0.5 - 1e-9])).tolist() == [1, 0]
    # a bf16-ish row summing to 0.98 is treated as normalised
    assert categorical_inverse_cdf(np.array([[0.49, 0.49]]), np.array([0.999]))[0] == 1
    # zero-mass tail never selected; clamp at V-1
    assert categorical_inverse_cdf(np.array([[1.0, 0.0]]), np.array([0.9999999]))[0] == 0


def test_categorical_matches_empirical_frequency():
    rng = np.random.default_rng(0)
    probs = np.tile(np.array([0.2, 0.3, 0.5]), (20000, 1))
    draws = categorical_inverse_cdf(probs, rng.random(20000))
    freq = np.bincount(draws, minlength=3) / 20000
    np.testing.assert_allclose(freq, [0.2, 0.3, 0.5], atol=0.02)


def test_budget_formula():
    assert budget_bytes(256, 48) == 4 * 256 * 97 == 99_328


def test_tape_logs_and_exhausts():
    t = EntropyTape(data=bytes(range(16)), label="x")
    t.step = 3
    a = t.uniforms(2, "canvas")
    b = t.randint(10, 2, "renoise")
    assert t.consumed == 16 and t.remaining == 0
    assert [(d.purpose, d.offset, d.nbytes, d.step) for d in t.log] == [("canvas", 0, 8, 3), ("renoise", 8, 8, 3)]
    assert a.shape == (2,) and b.shape == (2,)
    with pytest.raises(EntropyExhausted):
        t.uniforms(1)


def test_tape_from_seed_bundle(tmp_path):
    stem = tmp_path / "s"
    (tmp_path / "s.seed.bin").write_bytes(b"\x01" * 8)
    (tmp_path / "s.seed.json").write_text(json.dumps({"bytes": 8, "bank_fraction": 1.0, "tags": [], "label": "out_band"}))
    t = EntropyTape.from_seed(stem)
    assert t.label == "out_band" and t.remaining == 8
    assert EntropyTape.from_seed(str(stem) + ".seed.bin").label == "out_band"
    (tmp_path / "s.seed.json").write_text(json.dumps({"bytes": 9, "label": "x"}))
    with pytest.raises(ValueError):
        EntropyTape.from_seed(stem)


def test_prng_tape_is_deterministic():
    assert prng_tape(7, 64).data == prng_tape(7, 64).data
    assert prng_tape(7, 64).data != prng_tape(8, 64).data
