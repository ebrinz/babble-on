import pytest

from batch_build import CONDITIONS, bad_rng, build


def test_bad_rng_matches_the_apps_lcg():
    assert bad_rng(8) == bytes.fromhex("31071d142c091307")


def test_bad_rng_only_uses_six_bits():
    assert all(b < 64 for b in bad_rng(10_000))


def test_build_lays_samples_out_round_robin():
    pools = {"anomaly": b"A" * 8, "ordinary": b"B" * 8, "csprng": b"C" * 8, "biased": b"D" * 8}
    batch, samples = build(pools, per_condition=2, sample_bytes=4)
    assert batch == b"AAAABBBBCCCCDDDD" * 2
    assert [s["condition"] for s in samples] == list(CONDITIONS) * 2
    assert [s["index"] for s in samples] == list(range(8))


def test_build_consumes_each_pool_sequentially():
    pools = {"anomaly": b"12345678", "ordinary": b"B" * 8, "csprng": b"C" * 8, "biased": b"D" * 8}
    batch, _ = build(pools, per_condition=2, sample_bytes=4)
    assert batch[0:4] == b"1234" and batch[16:20] == b"5678"


def test_build_raises_when_pool_exhausted():
    pools = {"anomaly": b"A" * 4, "ordinary": b"B" * 8, "csprng": b"C" * 8, "biased": b"D" * 8}
    with pytest.raises(ValueError, match="anomaly pool exhausted at sample 1"):
        build(pools, per_condition=2, sample_bytes=4)
