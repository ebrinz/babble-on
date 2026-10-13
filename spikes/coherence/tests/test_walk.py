import pytest

from walk import TRIAL_BYTES, Z95, classify, trial_z


def test_trial_z_all_zero_bytes():
    # 256 bytes = 2048 bits, zero ones: z = (0 - 1024) / sqrt(512)
    assert trial_z(bytes(256)) == pytest.approx(-1024 / 512 ** 0.5)


def test_trial_z_balanced_bytes_is_zero():
    assert trial_z(bytes([0b10101010]) * 256) == 0.0


def test_classify_all_zero_stream_is_all_anomalous():
    data = bytes(TRIAL_BYTES * 4)
    anomaly, ordinary, sigmas = classify(data)
    assert anomaly == data and ordinary == b""
    assert len(sigmas) == 4 and all(abs(s) >= Z95 for s in sigmas)


def test_classify_balanced_stream_is_all_ordinary():
    data = bytes([0b10101010]) * (TRIAL_BYTES * 3)
    anomaly, ordinary, sigmas = classify(data)
    assert anomaly == b"" and ordinary == data
    assert sigmas == [0.0, 0.0, 0.0]


def test_classify_drops_trailing_partial_trial():
    data = bytes([0b10101010]) * (TRIAL_BYTES + 100)
    anomaly, ordinary, sigmas = classify(data)
    assert len(anomaly) + len(ordinary) == TRIAL_BYTES
    assert len(sigmas) == 1


def test_classify_keeps_byte_order_within_each_pool():
    balanced = bytes([0b10101010]) * TRIAL_BYTES
    ones = bytes([0xFF]) * TRIAL_BYTES
    anomaly, ordinary, sigmas = classify(balanced + ones)
    assert ordinary == balanced and anomaly == ones
    assert sigmas[0] == 0.0 and sigmas[1] >= Z95


def test_classify_honours_custom_trial_size():
    _, _, sigmas = classify(bytes(1000), trial_bytes=100)
    assert len(sigmas) == 10
