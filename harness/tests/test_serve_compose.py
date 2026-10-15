import json
import subprocess
import sys
from pathlib import Path

from babble_harness.compose import BYTES_PER_BRIEF, compose

CODE = ("import sys; sys.path.insert(0, %r); from babble_harness.serve import serve; "
        "serve('stub', vocab_size=64, canvas_length=16)" % str(Path(__file__).resolve().parents[1]))
BUDGET = 4 * 16 * 11  # budget_bytes(16, 5)


def _serve(*reqs: dict) -> list[dict]:
    stdin = "".join(json.dumps(r) + "\n" for r in reqs)
    p = subprocess.run([sys.executable, "-c", CODE], input=stdin, capture_output=True, text=True, timeout=120)
    return [json.loads(l) for l in p.stdout.splitlines() if l.strip()]


def test_compose_emits_brief_before_steps_and_repeats_it_in_done():
    raw = bytes(range(256)) * ((BYTES_PER_BRIEF + BUDGET) // 256 + 1)
    raw = raw[: BYTES_PER_BRIEF + BUDGET]
    lines = _serve({"steps": 5, "seq_len": 16, "entropy_hex": raw.hex(), "compose": True, "prompt": "ignored"})
    types = [l["type"] for l in lines]
    assert types[0] == "ready" and types[1] == "brief" and "step" in types and types[-1] == "done"
    brief, done = lines[1], lines[-1]
    prompt, picks = compose(raw[:BYTES_PER_BRIEF])
    assert brief == {"type": "brief", "picks": picks, "prompt": prompt, "bytes": BYTES_PER_BRIEF}
    assert done["brief"] == {"picks": picks, "prompt": prompt}
    assert done["seed"] == "entropy"
    assert done["entropy_used"] > BYTES_PER_BRIEF


def test_compose_samples_from_the_bytes_after_the_brief():
    raw = bytes([9]) * BYTES_PER_BRIEF + bytes(range(256)) * (BUDGET // 256 + 1)
    raw = raw[: BYTES_PER_BRIEF + BUDGET]
    prompt, _ = compose(raw)
    composed = _serve({"steps": 5, "seq_len": 16, "entropy_hex": raw.hex(), "compose": True})[-1]
    plain = _serve({"steps": 5, "seq_len": 16, "entropy_hex": raw[BYTES_PER_BRIEF:].hex(), "prompt": prompt})[-1]
    assert composed["tokens"] == plain["tokens"]
    assert composed["entropy_used"] == plain["entropy_used"] + BYTES_PER_BRIEF


def test_compose_refuses_missing_or_short_entropy():
    short = bytes(BYTES_PER_BRIEF + BUDGET - 1)
    lines = _serve({"steps": 5, "seq_len": 16, "compose": True},
                   {"steps": 5, "seq_len": 16, "entropy_hex": short.hex(), "compose": True})
    errors = [l["message"] for l in lines if l["type"] == "error"]
    assert len(errors) == 2
    assert "entropy" in errors[0]
    assert str(BYTES_PER_BRIEF + BUDGET) in errors[1] and str(len(short)) in errors[1]
