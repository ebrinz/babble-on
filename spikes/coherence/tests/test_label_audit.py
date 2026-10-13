import numpy as np

from label_audit import FEATURES, adjacent_ordinary, perm_matrix, trial_features


def _trial_from_bits(bits: np.ndarray) -> bytes:
    return np.packbits(bits.astype(np.uint8)).tobytes()


def test_lag1_autocorrelation_sees_a_planted_correlation():
    rng = np.random.default_rng(0)
    n = 5120 * 8
    fair = rng.integers(0, 2, n)
    sticky = fair.copy()
    for i in range(1, n):  # copy the previous bit 90% of the time
        if rng.random() < 0.9:
            sticky[i] = sticky[i - 1]
    f_fair = trial_features(_trial_from_bits(fair))
    f_sticky = trial_features(_trial_from_bits(sticky))
    i = FEATURES.index("ac_bit_1")
    assert abs(f_fair[i]) < 0.05
    assert f_sticky[i] > 0.7


def test_byte_chi2_flags_a_skewed_histogram():
    rng = np.random.default_rng(0)
    fair = rng.integers(0, 256, 5120, dtype=np.uint8).tobytes()
    skewed = bytes([7]) * 5120
    i = FEATURES.index("byte_chi2")
    assert 150 < trial_features(fair)[i] < 400  # ~255 expected under uniform
    assert trial_features(skewed)[i] > 100_000


def test_adjacent_ordinary_picks_in_band_neighbours_of_excursions():
    anomalous = np.array([0, 0, 1, 1, 0, 0, 0, 1], dtype=bool)
    assert adjacent_ordinary(anomalous).tolist() == [1, 4, 6]


def test_perm_matrix_small_p_only_for_the_shifted_column():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(200, 2))
    b = rng.normal(size=(200, 2))
    b[:, 0] += 1.0
    p = perm_matrix(a, b, n_perm=2000)
    assert p[0] < 0.01
    assert p[1] > 0.05


def test_constant_column_gives_zero_effect_and_p_one():
    from label_audit import cohen_d_cols
    # identical floats whose mean lands 1 ulp off, as 5131/5120 did over 30k trials
    a = np.full((1343, 1), 5131 / 5120)
    b = np.full((28775, 1), 5131 / 5120)
    assert cohen_d_cols(a, b)[0] == 0.0
    assert perm_matrix(a, b, n_perm=200)[0] == 1.0


def test_zlib_is_not_a_feature():
    assert "zlib_ratio" not in FEATURES
