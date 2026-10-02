import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from babble_harness.cli import main
from babble_harness.recording import RecordingHeader, RecordingWriter


def make_recording(path: Path, nbytes: int, biased_from: int | None = None, seed: int = 0):
    rng = np.random.default_rng(seed)
    w = RecordingWriter.create(path, RecordingHeader(source="test", started_at_ms=0))
    chunk = 512
    for i in range(nbytes // chunk):
        if biased_from is not None and i * chunk >= biased_from:
            data = bytes(np.where(rng.random(chunk) < 0.58, 0xFF, 0x00).astype(np.uint8))
        else:
            data = rng.bytes(chunk)
        w.frame((i + 1) * 100_000_000, data)
    w.close()


def test_label_command(tmp_path, capsys):
    rec = tmp_path / "r.bbrec"
    make_recording(rec, 512 * 100, biased_from=512 * 50)
    assert main(["label", str(rec), "--canvas", "16", "--steps", "6"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["trials"] == 100 and out["final_band"] == "99.9%"
    assert out["seeds_available"]["out_band"] >= 1 and out["seed_bytes"] == 4 * 16 * 13


def test_run_and_analyze_with_stub_end_to_end(tmp_path):
    rec = tmp_path / "r.bbrec"
    make_recording(rec, 512 * 120, biased_from=512 * 60, seed=3)
    out = tmp_path / "run1"
    rc = main(["run", str(out), "--model", "stub", "--recording", str(rec), "--prng", "4",
               "--canvas", "16", "--vocab", "64", "--steps", "6", "--max-per-group", "4",
               "--prompt", "hello", "--confidence", "0.05"])
    assert rc == 0
    man = json.loads((out / "manifest.json").read_text())
    assert man["samples"] >= 8 and man["skipped"] == 0
    rows = [json.loads(l) for l in (out / "samples.jsonl").read_text().splitlines()]
    conds = {r["condition"] for r in rows}
    assert {"in_band", "out_band", "prng"} <= conds
    r0 = rows[0]
    assert len(r0["ids"]) == 16 and r0["provenance"][0]["purpose"] == "initial_canvas"
    assert (out / "activations" / f"{r0['id']}.npz").exists()
    z = np.load(out / "activations" / f"{r0['id']}.npz")
    assert "pooled_first" in z and "step000_hidden" in z
    assert (out / "report.md").exists()
    rep = json.loads((out / "report.json").read_text())
    assert "n_steps" in rep["tests"]
    # re-analyze with a different pair
    assert main(["analyze", str(out), "--pair", "in_band", "prng", "--perm", "20"]) == 0


def test_serve_protocol_with_stub(tmp_path):
    req = json.dumps({"steps": 5, "seq_len": 16, "entropy_hex": (b"\x07" * (4 * 16 * 11)).hex(), "preview_every": 2})
    bad = json.dumps({"steps": 5, "seq_len": 16, "entropy_hex": "00"})
    code = "import sys; sys.path.insert(0, %r); from babble_harness.serve import serve; serve('stub', vocab_size=64, canvas_length=16)" % str(Path(__file__).resolve().parents[1])
    p = subprocess.run([sys.executable, "-c", code], input=req + "\n" + bad + "\n" + "{}\n", capture_output=True, text=True, timeout=120)
    lines = [json.loads(l) for l in p.stdout.splitlines() if l.strip()]
    assert lines[0] == {"type": "ready", "model": "stub"}
    types = [l["type"] for l in lines[1:]]
    assert types[-1] in ("done", "error")
    done = [l for l in lines if l["type"] == "done"]
    assert done and done[0]["seed"] == "entropy" and len(done[0]["tokens"]) == 16
    assert any(l["type"] == "step" for l in lines)
    assert any(l["type"] == "error" and "need" in l["message"] for l in lines)
    assert done[-1]["seed"] == "prng"  # the empty request falls back to PRNG and says so
