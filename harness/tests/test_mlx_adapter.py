"""MLX adapter held to the Transformers adapter on the tiny model.

Parity runs with MLX on the CPU (the GPU reorders fp32 arithmetic and differs
at ~1e-3). Skipped wherever mlx-vlm is not installed (Linux CI, Intel Macs).
"""
import os

import numpy as np
import pytest

pytest.importorskip("mlx_vlm")
torch = pytest.importorskip("torch")
import mlx.core as mx  # noqa: E402

from babble_harness.models.diffusion_gemma import DiffusionGemmaDenoiser  # noqa: E402
from babble_harness.models.mlx_gemma import MLXGemmaDenoiser, record_router  # noqa: E402
from babble_harness.models.tiny import CANVAS, EXPERTS, LAYERS, VOCAB, tiny_mlx_model, tiny_model  # noqa: E402
from babble_harness.noise import budget_bytes, prng_tape  # noqa: E402
from babble_harness.sampler import SamplerConfig, sample_canvas  # noqa: E402

PROMPT = [2, 10, 11, 12, 13]


@pytest.fixture
def cpu():
    # MLX JIT-compiles fused CPU kernels with the system toolchain, which may be
    # missing or broken; uncompiled CPU ops are the same arithmetic.
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.disable_compile()
    yield
    mx.enable_compile()
    mx.set_default_device(prev)


@pytest.fixture(scope="module")
def hf():
    d = DiffusionGemmaDenoiser(tiny_model(), capture_layers=(1, 2, 3))
    d.set_prompt_ids(PROMPT)
    return d


def _mlx(capture_layers=(1, 2, 3)):
    d = MLXGemmaDenoiser(tiny_mlx_model(), capture_layers=capture_layers)
    d.set_prompt_ids(PROMPT)
    return d


def test_tiny_config_translation_fills_none_and_refuses_unresolved():
    from babble_harness.models.tiny import translate_config
    cfg = tiny_model().config
    cfg.text_config.global_head_dim = None
    cfg.text_config.num_global_key_value_heads = None
    t = translate_config(cfg)["text_config"]
    assert t["global_head_dim"] == cfg.text_config.head_dim
    assert t["num_global_key_value_heads"] == cfg.text_config.num_key_value_heads
    d = cfg.to_dict()
    d["text_config"]["hidden_size"] = None  # Transformers itself refuses this; mimic a bad serialisation

    class Raw:
        def to_dict(self):
            return d
    with pytest.raises(ValueError, match="hidden_size"):
        translate_config(Raw())


def test_logits_match_transformers_on_cpu(hf, cpu):
    m = _mlx()
    canvas = np.random.default_rng(0).integers(0, VOCAB, CANVAS)
    a, _ = hf.denoise(canvas, None)
    b, _ = m.denoise(canvas, None)
    assert b.shape == (CANVAS, VOCAB) and b.dtype == np.float32
    np.testing.assert_allclose(b, a, atol=1e-5, rtol=1e-5)
    a2, _ = hf.denoise(canvas, a / 0.8)
    b2, _ = m.denoise(canvas, a / 0.8)
    np.testing.assert_allclose(b2, a2, atol=1e-5, rtol=1e-5)


def test_captures_match_transformers_on_cpu(hf, cpu):
    m = _mlx()
    canvas = np.random.default_rng(1).integers(0, VOCAB, CANVAS)
    _, ca = hf.denoise(canvas, None)
    _, cb = m.denoise(canvas, None)
    assert cb["hidden"].shape == (3, CANVAS, 64) and cb["hidden_layers"].tolist() == [1, 2, 3]
    np.testing.assert_allclose(cb["hidden"], ca["hidden"], atol=1e-5, rtol=1e-5)
    np.testing.assert_array_equal(cb["router_counts"], ca["router_counts"])
    assert cb["router_counts"].shape == (LAYERS, EXPERTS)
    np.testing.assert_allclose(cb["router_entropy"], ca["router_entropy"], atol=1e-6)
    np.testing.assert_array_equal(cb["logit_lens_agree"], ca["logit_lens_agree"])


def test_same_tape_same_trajectory_on_cpu(hf, cpu):
    m = _mlx()
    cfg = SamplerConfig(canvas_length=CANVAS, vocab_size=VOCAB, max_denoising_steps=4, early_stop=False)
    ra = sample_canvas(hf, prng_tape(3, budget_bytes(CANVAS, 4)), cfg)
    rb = sample_canvas(m, prng_tape(3, budget_bytes(CANVAS, 4)), cfg)
    for sa, sb in zip(ra.steps, rb.steps):
        np.testing.assert_array_equal(sb.canvas_out, sa.canvas_out)
    np.testing.assert_array_equal(rb.final_canvas, ra.final_canvas)


def test_gpu_repeat_is_bit_identical_and_cache_is_read_only():
    m = _mlx()
    canvas = np.random.default_rng(2).integers(0, VOCAB, CANVAS)
    before = m.cache_offset()
    a, _ = m.denoise(canvas, None)
    b, _ = m.denoise(canvas, None)
    np.testing.assert_array_equal(a, b)
    c, _ = m.denoise(canvas, a / 0.8)
    d, _ = m.denoise(canvas, a / 0.8)
    np.testing.assert_array_equal(c, d)
    assert not np.allclose(a, c)  # self-conditioning matters
    assert m.cache_offset() == before == len(PROMPT)


def test_gpu_close_to_cpu_within_loose_tolerance():
    m = _mlx()
    canvas = np.random.default_rng(3).integers(0, VOCAB, CANVAS)
    gpu, _ = m.denoise(canvas, None)
    mx.disable_compile()
    try:
        with mx.stream(mx.cpu):
            cpu_, _ = m.denoise(canvas, None)
    finally:
        mx.enable_compile()
    np.testing.assert_allclose(gpu, cpu_, rtol=1e-2, atol=1e-2)


def test_new_prompt_replaces_the_cache(cpu):
    m = _mlx()
    m.set_prompt_ids([2, 40, 41])
    fresh = MLXGemmaDenoiser(tiny_mlx_model(), capture_layers=())
    fresh.set_prompt_ids([2, 40, 41])
    canvas = np.random.default_rng(4).integers(0, VOCAB, CANVAS)
    assert m.cache_offset() == 3
    np.testing.assert_array_equal(m.denoise(canvas, None)[0], fresh.denoise(canvas, None)[0])


def test_router_patch_is_restored_even_on_error():
    from mlx_vlm.models.diffusion_gemma import language
    orig = language.Router.__call__
    with pytest.raises(RuntimeError):
        with record_router(True):
            assert language.Router.__call__ is not orig
            raise RuntimeError("boom")
    assert language.Router.__call__ is orig


def test_requires_prompt_and_describes_itself():
    d = MLXGemmaDenoiser(tiny_mlx_model(), capture_layers=())
    with pytest.raises(RuntimeError):
        d.denoise(np.zeros(CANVAS, dtype=np.int64), None)
    desc = d.describe()
    assert desc["backend"] == "mlx" and desc["mlx_vlm"] == "0.7.6" and desc["quantization"] is None


def test_loader_names_and_missing_mlx_hint(monkeypatch):
    import sys
    from babble_harness.models import load_denoiser
    d = load_denoiser("mlx_tiny")
    d.set_prompt("sea")
    assert d.prompt_ids[0] == 2 and len(d.prompt_ids) == 4
    monkeypatch.setitem(sys.modules, "mlx_vlm", None)
    with pytest.raises(ImportError, match=r"\.\[mlx\]"):
        load_denoiser("mlx")


@pytest.mark.skipif(os.environ.get("BABBLE_MLX_WEIGHTS") != "1", reason="real weights: set BABBLE_MLX_WEIGHTS=1")
def test_real_4bit_weights():
    from babble_harness.models.diffusion_gemma import load_processor
    d = MLXGemmaDenoiser.from_pretrained(capture_layers=(6, 30))
    d.set_prompt("Write a short passage of prose.")
    hf_ids = load_processor("google/diffusiongemma-26B-A4B-it").apply_chat_template(
        [{"role": "user", "content": "Write a short passage of prose."}], tokenize=True,
        add_generation_prompt=True, return_dict=True)["input_ids"]
    assert d.prompt_ids == np.asarray(hf_ids).reshape(-1).tolist()
    assert str(d._sc_dtype) == "mlx.core.bfloat16"
    canvas = np.random.default_rng(0).integers(0, d.vocab_size, d.canvas_length)
    a, cap = d.denoise(canvas, None)
    b, _ = d.denoise(canvas, None)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (256, 262_144) and cap["hidden"].shape == (2, 256, 2816)
    c, _ = d.denoise(canvas, a / 0.8)
    assert np.isfinite(c).all() and not np.allclose(a, c)
    q = d.describe()["quantization"]
    assert q["default"]["bits"] == 4 and q["overrides"] > 0


def test_cli_run_with_mlx_tiny_records_the_backend(tmp_path, monkeypatch):
    import json
    from babble_harness.cli import main
    monkeypatch.setenv("BABBLE_EXPERIMENT_LOG", "0")
    out = tmp_path / "mt"
    assert main(["run", str(out), "--model", "mlx_tiny", "--prng", "2", "--steps", "3", "--prompt", "sea",
                 "--no-early-stop"]) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["backend"]["backend"] == "mlx" and manifest["backend"]["mlx_vlm"] == "0.7.6"
    rows = [json.loads(l) for l in (out / "samples.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and rows[0]["n_steps"] == 3
    assert "backend: backend `mlx`" in (out / "report.md").read_text()


def test_cli_refuses_quant_with_mlx(tmp_path, capsys):
    from babble_harness.cli import main
    assert main(["run", str(tmp_path / "x"), "--model", "mlx", "--quant", "bnb4", "--prng", "1"]) == 2
    assert "--model-id" in capsys.readouterr().err


def test_log_entry_names_the_backend(tmp_path):
    from babble_harness.runner import log_entry
    entry = log_entry(tmp_path, {"model": "mlx", "n": {}, "backend": {"backend": "mlx", "model_id": "m",
                      "quantization": {"default": {"bits": 4}, "overrides": 3}}}, ("a", "b"), "now")
    assert "backend `mlx` · `m` · 4-bit (+3 overrides)" in entry
