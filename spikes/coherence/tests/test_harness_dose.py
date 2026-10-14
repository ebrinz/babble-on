"""Run with the harness venv: ../../harness/.venv/bin/python -m pytest tests/test_harness_dose.py"""
import json

import numpy as np
import pytest

pytest.importorskip("babble_harness")

from babble_harness.models import load_denoiser  # noqa: E402
from babble_harness.noise import budget_bytes  # noqa: E402
from babble_harness.sampler import SamplerConfig  # noqa: E402

from harness_dose import run  # noqa: E402

L, V, S = 16, 64, 4


def _setup(n=3):
    sb = budget_bytes(L, S)
    data = np.random.default_rng(0).bytes(sb * n)
    manifest = {"sample_bytes": sb, "samples": [{"index": i, "condition": "d0", "delta": 0.0} for i in range(n)]}
    den = load_denoiser("stub", vocab_size=V, canvas_length=L)
    cfg = SamplerConfig(canvas_length=L, vocab_size=V, max_denoising_steps=S)
    return den, cfg, data, manifest


def _lines(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def test_writes_one_line_per_sample_with_steps_and_consumption(tmp_path):
    den, cfg, data, manifest = _setup()
    out = tmp_path / "b.jsonl"
    run(den, cfg, data, manifest, out, log=lambda *a: None)
    lines = _lines(out)
    assert [l["index"] for l in lines] == [0, 1, 2]
    assert all(l["error"] is None and 1 <= l["steps"] <= S and l["consumed"] > 0 for l in lines)


def test_is_a_pure_function_of_the_bytes(tmp_path):
    den, cfg, data, manifest = _setup(1)
    run(den, cfg, data, manifest, tmp_path / "a.jsonl", log=lambda *a: None)
    run(den, cfg, data, manifest, tmp_path / "b.jsonl", log=lambda *a: None)
    a, b = _lines(tmp_path / "a.jsonl")[0], _lines(tmp_path / "b.jsonl")[0]
    assert (a["text"], a["consumed"], a["steps"]) == (b["text"], b["consumed"], b["steps"])


def test_resumes_and_retries_only_failed_samples(tmp_path):
    den, cfg, data, manifest = _setup()
    out = tmp_path / "b.jsonl"
    out.write_text(json.dumps({"index": 0, "text": "kept", "consumed": 1, "steps": 1, "elapsed_s": 0, "error": None}) + "\n"
                   + json.dumps({"index": 1, "text": "", "consumed": 0, "steps": None, "elapsed_s": 0, "error": "OOM"}) + "\n")
    run(den, cfg, data, manifest, out, log=lambda *a: None)
    lines = _lines(out)
    assert sorted(l["index"] for l in lines) == [0, 1, 2]  # failed line replaced, not duplicated
    assert next(l for l in lines if l["index"] == 0)["text"] == "kept"
    assert all(l["error"] is None for l in lines)


def test_a_runtime_error_is_recorded_not_raised(tmp_path):
    den, cfg, data, manifest = _setup(1)

    class Boom:
        canvas_length, vocab_size = L, V

        def denoise(self, canvas, sc):
            raise RuntimeError("[METAL] out of memory")
    run(Boom(), cfg, data, manifest, tmp_path / "b.jsonl", log=lambda *a: None)
    line = _lines(tmp_path / "b.jsonl")[0]
    assert line["error"].startswith("[METAL]") and line["text"] == "" and line["steps"] is None


def test_refuses_reservoirs_smaller_than_the_worst_case_budget(tmp_path):
    den, cfg, data, manifest = _setup(1)
    manifest["sample_bytes"] = budget_bytes(L, S) - 4
    with pytest.raises(ValueError, match="budget"):
        run(den, cfg, data, manifest, tmp_path / "b.jsonl", log=lambda *a: None)


def test_text_drops_the_thought_channel_header():
    from harness_dose import text_of

    class Proc:
        def clean_output(self, text):  # mlx-vlm drops <|channel>…<channel|> spans
            import re
            return re.sub(r"<\|channel>.*?<channel\|>", "", text, flags=re.S).lstrip()

    class Den:
        processor = Proc()

        def decode(self, ids, skip_special=False):
            return "thought\nThe sun." if skip_special else "<|channel>thought\n<channel|>The sun."
    assert text_of(Den(), np.array([1, 2])) == "The sun."
