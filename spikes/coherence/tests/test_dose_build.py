import numpy as np
import pytest

from dose_build import DOSES, bias_bits, build


def _mean_bit(b: bytes) -> float:
    return float(np.unpackbits(np.frombuffer(b, dtype=np.uint8)).mean())


def test_zero_dose_leaves_bytes_untouched():
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, 4096, dtype=np.uint8).tobytes()
    assert bias_bits(data, 0.0, np.random.default_rng(1)) == data


def test_dose_shifts_the_mean_bit_by_delta():
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, 200_000, dtype=np.uint8).tobytes()
    assert _mean_bit(bias_bits(data, 0.1, np.random.default_rng(1))) == pytest.approx(0.6, abs=0.002)
    assert _mean_bit(bias_bits(data, 0.01, np.random.default_rng(1))) == pytest.approx(0.51, abs=0.002)


def test_build_round_robins_doses_then_lcg_and_records_delta():
    carrier = bytes(range(256)) * 100  # 25 600 bytes
    batch, samples = build(carrier, per_condition=2, sample_bytes=100, seed=0)
    conds = [s["condition"] for s in samples]
    expected_cycle = [f"d{d:g}" for d in DOSES] + ["lcg"]
    assert conds == expected_cycle * 2
    assert samples[0]["delta"] == 0.0 and samples[0]["condition"] == "d0"
    assert samples[len(DOSES)]["condition"] == "lcg" and samples[len(DOSES)]["delta"] is None
    assert len(batch) == len(samples) * 100
    # d0 chunks are verbatim carrier bytes, consumed front to back
    assert batch[:100] == carrier[:100]


def test_build_raises_when_carrier_runs_out():
    with pytest.raises(ValueError, match="carrier exhausted"):
        build(bytes(500), per_condition=1, sample_bytes=100, seed=0)


def test_build_honours_a_larger_sample_size():
    carrier = bytes(range(256)) * 400
    batch, samples = build(carrier, per_condition=1, sample_bytes=1000, seed=0)
    assert len(batch) == len(samples) * 1000
