import json

import pytest

from generate import done_indices, run_sample, slice_for


def test_slice_for_returns_the_samples_reservoir():
    data = b"A" * 4 + b"B" * 4 + b"C" * 4
    assert slice_for(1, 4, data) == b"BBBB"


def test_slice_for_rejects_short_slice():
    with pytest.raises(ValueError, match="sample 3"):
        slice_for(3, 4, b"A" * 12)


def test_done_indices_reads_existing_lines(tmp_path):
    f = tmp_path / "out.jsonl"
    f.write_text(json.dumps({"index": 0}) + "\n" + json.dumps({"index": 2}) + "\n")
    assert done_indices(f) == {0, 2}


def test_done_indices_missing_file_is_empty(tmp_path):
    assert done_indices(tmp_path / "nope.jsonl") == set()


def test_run_sample_records_text_and_bytes_consumed():
    def fake_generate(pool):
        pool.take(10)
        return "hello"

    r = run_sample(b"\x00" * 64, fake_generate)
    assert r["text"] == "hello" and r["error"] is None
    assert r["consumed"] == 4 + 10  # 4-byte sampler seed + what fake_generate took
    assert r["elapsed_s"] >= 0


def test_run_sample_records_exhausted_reservoir_as_error():
    def fake_generate(pool):
        pool.take(100)
        return "never"

    r = run_sample(b"\x00" * 64, fake_generate)
    assert r["text"] == "" and "exhausted" in r["error"]
