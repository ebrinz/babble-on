import pytest

from score import distinct_ratio, score_line, steps_from_consumed


def test_distinct_ratio():
    assert distinct_ratio("a b a c") == pytest.approx(0.75)
    assert distinct_ratio("") == 0.0


def test_steps_from_consumed_matches_spike_measurement():
    # 4-byte seed + initial canvas + 13 re-noised canvases, 1024 B each
    assert steps_from_consumed(14340) == 13


def test_score_line_adds_metrics():
    line = {"index": 0, "text": "the cat sat", "consumed": 14340, "error": None}
    out = score_line(line, lambda text: 2.5)
    assert out["gpt2_nll"] == 2.5
    assert out["distinct_ratio"] == pytest.approx(1.0)
    assert out["steps"] == 13
    assert out["index"] == 0 and out["text"] == "the cat sat"


def test_score_line_errors_and_empty_text_get_null_nll():
    failed = {"index": 1, "text": "", "consumed": 51200, "error": "replay bytes exhausted"}
    out = score_line(failed, lambda text: 1.0)
    assert out["gpt2_nll"] is None and out["steps"] is None and out["distinct_ratio"] is None
    blank = {"index": 2, "text": "   \n", "consumed": 2052, "error": None}
    out = score_line(blank, lambda text: 1.0)
    assert out["gpt2_nll"] is None and out["steps"] is None and out["distinct_ratio"] is None


def test_score_line_keeps_a_recorded_step_count():
    # the harness driver records steps directly; its tape spends 2 KB per step, not 1 KB
    line = {"index": 3, "text": "the cat", "consumed": 25600, "steps": 12, "error": None}
    assert score_line(line, lambda text: 1.0)["steps"] == 12
