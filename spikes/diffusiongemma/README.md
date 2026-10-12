# DiffusionGemma spike

Can `google/diffusiongemma-26B-A4B-it` replace Plaid while keeping the
"entropy authors the text" property? This spike tests whether it runs on our
Mac and whether randomness we supply steers its output.

## Setup

```bash
cd spikes/diffusiongemma
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -U mlx-vlm   # 0.7.6 at time of test
hf download mlx-community/diffusiongemma-26B-A4B-it-4bit   # ~15 GB
.venv/bin/python smoke.py [--prompt "..."] [--temperature 0.7]
```

## Results (2026-10-08, M1 Pro 32 GB, temperature 0, 48 steps)

- Load 7.8 s, roughly 15–21 s per 256-token canvas, peak memory 17.5 GB.
- Same canvas and same seed reproduce the output exactly. A new canvas gives
  0.36 word similarity; a new per-step seed alone gives 0.48.
- Both random sources matter. The initial canvas can be supplied through
  `decoder_input_ids`, but every step also re-noises unsettled positions with
  `mx.random`. For the output to depend only on TrueRNG, drive the seed (or
  patch the re-noise) from TrueRNG too.
- The prompt sets the subject: all four runs described a sunset over a valley.
  Entropy changes the wording, not the topic.

## Step 1: pure function of entropy (2026-10-12)

`entropy.py` routes all three random draws in mlx-vlm's loop (initial canvas,
per-step re-noise of unaccepted positions, sampler PRNG seed) through an
`EntropyPool` fed by the TrueRNG, recording every byte consumed. `pure.py`
checks it:

```bash
uv pip install --python .venv/bin/python pyserial
.venv/bin/python pure.py        # TrueRNG at /dev/cu.usbmodem112401
```

- Replaying the recorded bytes reproduces the text exactly (word similarity
  1.00) at temperature 0 and 0.7, even after mlx's global PRNG was advanced in
  between. Fresh bytes give 0.49 and 0.30 similarity respectively.
- A 256-token generation consumes 11–14 KB of TrueRNG bytes: 4 for the seed,
  then 1 KB (a full 256-token canvas, 4 bytes each) for the initial canvas and
  again for every denoising step, of which the confidence sampler ran 10–13.
  The device delivers about 50 KB/s, so reading live costs well under a second.

## Step 2: can entropy move the subject? (2026-10-12)

`sweep.py` runs prompt × temperature, three generations per cell with fresh
TrueRNG bytes each, and reports mean pairwise word similarity within the cell
(lower = entropy moved more). Every text is in `sweep_results.json`.

```bash
.venv/bin/python sweep.py
```

| prompt (chat template) | T=0 | T=0.7 | T=1.0 | what came out |
|---|---|---|---|---|
| "Write a short passage of prose." | 0.41 | 0.29 | 0.38 | always a sunset over a valley; wording varies |
| "Write." | 0.39 | 0.44 | 0.50 | assistant asks what you want written |
| "" (empty user turn) | 0.14 | 0.47 | 0.10 | assistant asks what you're referring to; the *guess* varies (riddle, image, war timeline, factoring a quadratic) |
| "" with no chat template | 0.11 | 0.07 | 0.02 | word salad with "(Wait, …)" self-corrections; one run came back blank |

- With a real prompt the subject never moves; entropy only re-phrases. The
  instruct tuning dominates.
- Dropping the prompt lets entropy pick the topic, but the register is still
  "assistant replying to a missing question", not prose.
- Dropping the chat template too gives the closest thing to Plaid babble, but it
  is degenerate (repetition, self-correction loops) and slow (40–75 s per
  generation vs ~20 s), since few positions ever reach the confidence threshold.
- Temperature barely matters for similarity once a prompt is present.

**Takeaway:** the entropy-is-the-author property holds mechanically (step 1),
but the instruct model's prior squeezes it into wording. Getting entropy to
choose the subject would need either a base (non-instruct) checkpoint, which
Google has not published for DiffusionGemma, or an authored prompt that
explicitly hands the choice over (e.g. "Pick any subject and write…"), which is
untested here.
