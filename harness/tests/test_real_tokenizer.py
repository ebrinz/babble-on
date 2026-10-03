"""The adapter's prompt path against the real DiffusionGemma tokenizer and
chat template (tokenizer files only — no weights). Skipped unless the files
are already in the Hugging Face cache (set HF_HUB_OFFLINE=1 to force the
cache; the CI job has no network to the Hub)."""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from babble_harness.models.diffusion_gemma import MODEL_ID, DiffusionGemmaDenoiser, load_processor
from babble_harness.models.tiny import CANVAS, tiny_model
from babble_harness.noise import budget_bytes, prng_tape
from babble_harness.sampler import SamplerConfig, sample_canvas


@pytest.fixture(scope="module")
def processor():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        return load_processor(MODEL_ID)
    except Exception as e:  # noqa: BLE001 — not cached / no network
        pytest.skip(f"DiffusionGemma tokenizer not available offline: {type(e).__name__}")


def test_chat_template_and_decode(processor):
    den = DiffusionGemmaDenoiser(tiny_model(vocab_size=262_144), processor, capture_layers=())
    den.set_prompt("Why is the sky blue?")
    ids = den.prompt_ids
    text = processor.decode(ids, skip_special_tokens=False)
    assert ids[0] == 2  # <bos>
    assert text.startswith("<bos><|turn>user\n") and text.endswith("<|turn>model\n")
    assert "Why is the sky blue?" in text
    assert den.decode_each(np.array(ids[:2])) == ["<bos>", "<|turn>"]
    # EOS trimming follows generation_config (the checkpoint's is [1, 106, 50]; 106 = <turn|>)
    den.model.generation_config.eos_token_id = [1, 106, 50]
    canvas = np.array([5, 6, 106, 7])
    assert den.trim_after_eos(canvas).tolist() == [5, 6]
    assert den.decode(den.trim_after_eos(canvas)) == processor.decode([5, 6])


def test_system_prompt_and_full_vocab_sample(processor):
    den = DiffusionGemmaDenoiser(tiny_model(vocab_size=262_144), processor, capture_layers=(1,), system="Be brief.")
    den.set_prompt("hi")
    assert "Be brief." in processor.decode(den.prompt_ids)
    cfg = SamplerConfig(canvas_length=CANVAS, vocab_size=den.vocab_size, max_denoising_steps=2, early_stop=False)
    r = sample_canvas(den, prng_tape(1, budget_bytes(CANVAS, 2)), cfg)
    assert r.final_canvas.shape == (CANVAS,) and int(r.final_canvas.max()) < 262_144
    assert isinstance(den.decode(r.final_canvas), str)
