"""Step-for-step parity between `babble_harness.sampler` and Transformers'
`DiffusionGemmaGenerationMixin.generate`, on the tiny random model.

The reference's two RNG calls (`torch.randint` for the initial canvas and
renoising, `torch.multinomial` for the categorical draw) are patched to pull
from an `EntropyTape` through the same bytes->uniform->index mapping the
harness uses. With identical draws, every intermediate canvas must match.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from babble_harness.models.diffusion_gemma import DiffusionGemmaDenoiser
from babble_harness.noise import EntropyTape, budget_bytes, categorical_inverse_cdf, prng_tape, uniform_to_index
from babble_harness.sampler import SamplerConfig, sample_canvas
from babble_harness.models.tiny import CANVAS, VOCAB, tiny_model

PROMPT = [2, 10, 11, 12, 13]


def run_reference(model, tape: EntropyTape, steps: int, early_stop: bool, confidence: float, stability: int):
    """Transformers generate() with its RNG routed through the tape; returns
    the per-step (current_canvas_out, argmax_canvas) pairs and the initial canvas."""
    import transformers.models.diffusion_gemma.generation_diffusion_gemma as gen

    log = {"init": None, "steps": []}
    real_randint, real_multinomial = torch.randint, torch.multinomial

    def fake_randint(low, high, size, device=None, **kw):
        n = int(np.prod(size))
        idx = tape.randint(int(high), n, "randint")
        t = torch.as_tensor(idx.reshape(size), dtype=torch.long, device=device)
        if log["init"] is None:
            log["init"] = t[0].clone().numpy()
        return t

    def fake_multinomial(probs, num_samples=1, **kw):
        assert num_samples == 1
        idx = categorical_inverse_cdf(probs.detach().cpu().numpy(), tape.uniforms(probs.shape[0], "categorical"))
        return torch.as_tensor(idx, dtype=torch.long, device=probs.device)[:, None]

    orig_step = model._denoising_step

    def spy_step(*a, **k):
        out = orig_step(*a, **k)
        log["steps"].append((out[0][0].clone().numpy(), out[1][0].clone().numpy()))
        return out

    gen.torch.randint, gen.torch.multinomial = fake_randint, fake_multinomial
    model._denoising_step = spy_step
    try:
        kw = dict(max_new_tokens=CANVAS, max_denoising_steps=steps, return_dict_in_generate=True)
        if early_stop:
            kw.update(confidence_threshold=confidence, stability_threshold=stability)
        else:
            # generation config validation forbids None here; make stopping unreachable instead
            kw.update(confidence_threshold=1e-12, stability_threshold=steps + 1)
        with torch.no_grad():
            out = model.generate(input_ids=torch.tensor([PROMPT]), **kw)
    finally:
        gen.torch.randint, gen.torch.multinomial = real_randint, real_multinomial
        model._denoising_step = orig_step
    return log, out.sequences[0, len(PROMPT):].numpy()


@pytest.mark.parametrize("early_stop,confidence,stability,expect_stop", [
    (False, 0.005, 1, False),   # plain run, all steps
    (True, 2.0, 1, False),      # stopping armed but a random model never goes stable
    (True, 100.0, 0, True),     # stability 0 = always stable, huge confidence: stops after step 1
])
def test_sampler_matches_transformers_generate(early_stop, confidence, stability, expect_stop):
    steps = 6
    model = tiny_model()
    need = budget_bytes(CANVAS, steps)
    ref_log, ref_seq = run_reference(model, prng_tape(11, need), steps, early_stop, confidence, stability)

    den = DiffusionGemmaDenoiser(model, capture_layers=(), capture_router=False, logit_lens=False)
    den.set_prompt_ids(PROMPT)
    cfg = SamplerConfig(canvas_length=CANVAS, vocab_size=VOCAB, max_denoising_steps=steps,
                        early_stop=early_stop, confidence_threshold=confidence, stability_threshold=stability)
    ours = sample_canvas(den, prng_tape(11, need), cfg)

    np.testing.assert_array_equal(ours.initial_canvas, ref_log["init"])
    assert len(ours.steps) == len(ref_log["steps"]), (len(ours.steps), len(ref_log["steps"]))
    for i, (st, (ref_canvas, ref_argmax)) in enumerate(zip(ours.steps, ref_log["steps"])):
        np.testing.assert_array_equal(st.argmax, ref_argmax, err_msg=f"argmax differs at step {i}")
        np.testing.assert_array_equal(st.canvas_out, ref_canvas, err_msg=f"canvas differs at step {i}")
    assert ours.stopped_early == expect_stop
    if expect_stop:
        assert len(ours.steps) == 1
    # generate() pads after an EOS; compare up to the first eos/pad only
    final = ours.final_canvas
    cut = next((i for i, t in enumerate(final) if int(t) in (0, 1)), len(final))
    np.testing.assert_array_equal(final[:cut], ref_seq[:cut])


def test_reference_consumes_the_budget_exactly():
    steps = 3
    model = tiny_model()
    need = budget_bytes(CANVAS, steps)
    tape = prng_tape(5, need)
    run_reference(model, tape, steps, False, 0.0, 0)
    assert tape.consumed == need and tape.remaining == 0
    assert [d.purpose for d in tape.log] == ["randint"] + ["categorical", "randint"] * steps
