from classify import check_pools


def test_check_pools_names_short_pools():
    assert check_pools(10, 10, 100) == ["anomaly", "ordinary"]
    assert check_pools(100, 10, 100) == ["ordinary"]
    assert check_pools(100, 100, 100) == []
