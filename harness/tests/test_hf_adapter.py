import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from babble_harness.models.diffusion_gemma import DiffusionGemmaDenoiser
from babble_harness.noise import budget_bytes, prng_tape
from babble_harness.sampler import SamplerConfig, sample_canvas
from babble_harness.models.tiny import CANVAS, EXPERTS, LAYERS, VOCAB, tiny_model

PROMPT = [2, 10, 11, 12, 13]


@pytest.fixture(scope="module")
def den():
    d = DiffusionGemmaDenoiser(tiny_model(), capture_layers=(1, 2, 3))
    d.set_prompt_ids(PROMPT)
    return d


def test_shapes_and_capture(den):
    assert (den.canvas_length, den.vocab_size) == (CANVAS, VOCAB)
    canvas = np.arange(CANVAS) % VOCAB
    logits, cap = den.denoise(canvas, None)
    assert logits.shape == (CANVAS, VOCAB) and logits.dtype == np.float32
    assert np.abs(logits).max() <= den.softcap + 1e-5  # softcapped
    assert cap["hidden"].shape == (3, CANVAS, 64) and cap["hidden_layers"].tolist() == [1, 2, 3]
    assert cap["router_counts"].shape == (LAYERS, EXPERTS)
    assert cap["router_counts"].sum(axis=1).tolist() == [CANVAS * den.top_k] * LAYERS
    assert cap["router_entropy"].shape == (LAYERS,) and (cap["router_entropy"] >= 0).all()
    assert cap["logit_lens_agree"].shape == (3,)
    assert cap["logit_lens_agree"][-1] == pytest.approx(1.0)  # last layer == final prediction


def test_step_matches_top_level_forward_and_cache_is_read_only(den):
    canvas = np.random.default_rng(0).integers(0, VOCAB, CANVAS)
    before = den._cache.get_seq_length()
    logits, _ = den.denoise(canvas, None)
    with torch.no_grad():
        ref = den.model(decoder_input_ids=torch.as_tensor(canvas)[None], past_key_values=den._cache,
                        decoder_attention_mask=den._dec_mask, decoder_position_ids=den._dec_pos).logits[0].numpy()
    np.testing.assert_allclose(logits, ref, rtol=1e-5, atol=1e-5)
    assert den._cache.get_seq_length() == before == len(PROMPT)
    # self-conditioning changes the output
    logits2, _ = den.denoise(canvas, logits / 0.8)
    assert not np.allclose(logits, logits2)


def test_denoise_is_deterministic(den):
    canvas = np.random.default_rng(1).integers(0, VOCAB, CANVAS)
    a, _ = den.denoise(canvas, None)
    b, _ = den.denoise(canvas, None)
    np.testing.assert_array_equal(a, b)


def test_full_sample_on_tiny_model(den):
    cfg = SamplerConfig(canvas_length=CANVAS, vocab_size=VOCAB, max_denoising_steps=4, early_stop=False)
    r = sample_canvas(den, prng_tape(3, budget_bytes(CANVAS, 4)), cfg)
    assert r.n_steps == 4 and r.final_canvas.shape == (CANVAS,)
    assert all("router_entropy" in s.capture and "hidden" in s.capture for s in r.steps)
    assert den.decode(r.final_canvas)  # falls back to ids without a processor


def test_requires_prompt_before_denoise():
    d = DiffusionGemmaDenoiser(tiny_model(), capture_layers=())
    with pytest.raises(RuntimeError):
        d.denoise(np.zeros(CANVAS, dtype=np.int64), None)
    with pytest.raises(RuntimeError):
        d.set_prompt("hi")  # no processor
    assert d.quantized_fraction() == 0.0
