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

## Step 2b: hand-over prompts (2026-10-12)

Can a prompt delegate the subject choice to entropy? Two prompts via
`sweep.py --prompts handover,handover-wild --temperatures 0,1.0`; texts in
`handover_results.json`.

| prompt | T=0 | T=1.0 | subject chosen |
|---|---|---|---|
| "Pick any subject you like and write a short passage about it." | 0.38 | 0.34 | bioluminescence, 6 of 6 runs |
| "Choose a surprising subject at random and write a short passage of prose about it." | 0.40 | 0.30 | the wombat, 8 of 8 runs (one run: its square droppings) |

Asking the model to choose at random just moves the fixed point: "random
subject" is itself a mode of the instruct model. Entropy still only varies the
wording. The TrueRNG dropped off USB ("Device not configured") during the
first wild cell; it re-enumerated at the same path and the cell was rerun.

## Step 3: streaming preview (2026-10-12)

Can a sidecar get the canvas after every denoising step, live, for the app's
crystallising-text view? `preview.py` drives the generic
`stream_diffusion_generate` generator directly with
`diffusion_show_unmasking=True`. (The public `generate`/`stream_generate`
collect the per-step drafts and release them only after generation ends, so
they are no use for a live preview.)

- Yes: 13 draft events over a 36 s generation, first at 2.4 s, then one every
  ~2.5 s (max gap 5.7 s), each carrying the full canvas text with unrevealed
  positions rendered as `[Mask]`.
- The reveal is confidence-ordered and mostly left-to-right: accepted tokens
  are frozen and only the gaps change. This is a fill-in-the-blanks animation,
  not Plaid's every-token-boils-every-step effect. A sidecar could instead
  decode the raw canvas (`draft_canvas`) to show the noise tokens in the gaps.
- The canvas carries a `<|channel>thought\n<channel|>` prefix (Gemma's
  thinking channel) that must be stripped before display; the public
  `generate` does this via `skip_special_token_ids`.
- Only 13 of the nominal 48 steps ran: the confidence-threshold sampler stops
  once every position is accepted. That matches the 11–14 KB entropy budget
  measured in step 1.
