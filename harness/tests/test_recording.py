import io
from pathlib import Path

import pytest

from babble_harness.recording import MAGIC, RecordingHeader, RecordingReader, RecordingWriter, read_recording

FIXTURE = Path(__file__).parent / "fixtures" / "tiny.bbrec"


def test_round_trip_in_memory():
    buf = io.BytesIO()
    w = RecordingWriter(buf, RecordingHeader(source="simulate", started_at_ms=5))
    w.frame(10, b"\x01\x02\x03")
    w.frame(20, b"")
    w.frame(30, b"\x04")
    assert (w.frames, w.bytes) == (2, 4)
    buf.seek(0)
    assert buf.getvalue()[:8] == MAGIC
    r = RecordingReader(buf)
    assert r.header == RecordingHeader(source="simulate", started_at_ms=5, trial_interval_ms=100, trial_min_bits=2048, notes="")
    assert [(f.t_ns, f.data) for f in r] == [(10, b"\x01\x02\x03"), (30, b"\x04")]


def test_truncated_tail_is_end():
    buf = io.BytesIO()
    w = RecordingWriter(buf, RecordingHeader(source="s", started_at_ms=0))
    w.frame(1, b"abc")
    w.frame(2, b"defg")
    data = buf.getvalue()[:-2]
    frames = RecordingReader(io.BytesIO(data)).frames()
    assert [f.data for f in frames] == [b"abc"]


def test_bad_magic():
    with pytest.raises(ValueError):
        RecordingReader(io.BytesIO(b"NOPE0001\x00\x00\x00\x00"))


def test_fixture_matches_rust_expectations():
    header, frames = read_recording(FIXTURE)
    assert header.source == "fixture" and header.trial_min_bits == 2048
    assert len(frames) == 3
    assert (frames[0].t_ns, frames[0].data) == (0, b"\xde\xad")
    assert frames[2].t_ns == 200_000_000 and len(frames[2].data) == 256
