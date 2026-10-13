# Experiment log

One sequential log of every experiment in babble-on: spikes, audits, sweeps,
and "try it and see" UI changes. Newest entry at the bottom. Append an entry
in the same commit as the experiment's code or revert, so the log and the
history never drift apart. Each entry answers: what question, what was done,
what came out, and what was decided.

Entries before 2026-10-12 were backfilled from git history, the docs, and
session notes when the log was created.

---

## 2026-06-24 — Candle port of Plaid-1B validated against the torch sidecar

**Question:** Does the all-Rust candle engine reproduce the Python reference
forward pass closely enough to replace it?

**Done:** `cargo run` validate mode compares the candle forward output against
the torch sidecar on the same input.

**Result:** max |Δ| < 1e-2. The candle engine became the app's only generation
backend (commit 4335ce8); the Python sidecar stayed as the reference.

---

## 2026-07-09 to 2026-07-12 — Sampler parity audit

**Question:** Is the candle sampler drifting from the reference pipeline, and
could that explain the weak text quality?

**Done:** Side-by-side audit of `diffusion-rs/src/sampler.rs` against
`sidecar/sidecar.py` (timesteps, schedule, self-conditioning, score
temperature, ancestral update, final decode), plus a 768-step evidence run.
Full table in `docs/2026-07-09-sampler-parity-audit.md`.

**Result:** No divergence. Text quality is bounded by Plaid-1B itself. Shipped
the "ultra · 768 steps" preset (commit b95defc). RePlaid flagged as the upgrade
path once weights appear.

---

## 2026-07-13 — Prose substrate scan (brainstorm only, no code)

**Question:** How to make the generated text read more like prose without
muting the entropy-seeds-the-latent property.

**Done:** Code reading of where entropy enters and what mutes it (only the
initial latent is TrueRNG-seeded; prefix inpainting and low score temperature
collapse seeds together; DDIM makes output a pure function of the seed). Model
scan: RePlaid (no weights), Cola-DLM (continuous latent diffusion, Apache-2.0,
~4.6 GB, MPS untested), LLaDA/Dream (masked, 7–8B, no MPS sampler).

**Result:** Fine-tuning Plaid rejected. Decision between a Cola-DLM swap, a
Plaid decoding-sweep "jig", or masked diffusion with entropy-driven unmasking
order was left open. Superseded in practice by the DiffusionGemma spike below.

---

## 2026-07-14 — Rolling window for the coherence walk (rejected)

**Question:** Would a fixed-x-scale 2-minute rolling window read better than
the anchored envelope funnel on long sessions?

**Done:** Built it (commit 467df97), ran it live.

**Result:** Rejected. The envelopes and scene visibly translating read worse
than the whole session compressing into a static funnel. Reverted in 1fc8284.
Do not re-propose a scrolling window; the alternative offered and declined was
a σ-normalized view with fixed horizontal bands.

---

## 2026-10-08 — DiffusionGemma smoke test

**Question:** Can `diffusiongemma-26B-A4B-it` (MLX 4-bit) run on the M1 Pro,
and does randomness we supply steer its output?

**Done:** `spikes/diffusiongemma/smoke.py` via mlx-vlm 0.7.6. Four runs
separating the initial canvas from the per-step re-noise seed. Branch
`spike/diffusiongemma`, commit 75e12d8.

**Result:** Load ~8 s, ~20 s per 256-token canvas, peak 17.5 GB. Same canvas
and seed reproduce exactly; new canvas gives 0.36 word similarity, new seed
alone 0.48, so both sources must be entropy-driven. Prose quality far above
Plaid, but the prompt fixed the subject in every run.

---

## 2026-10-12 — DiffusionGemma: entropy purity check and prompt × temperature sweep

**Question:** (1) Can the output be made a pure function of TrueRNG bytes?
(2) Can entropy move the subject, not just the wording?

**Done:** `spikes/diffusiongemma/entropy.py` routes the initial canvas, every
per-step re-noise, and the sampler seed through a recording TrueRNG pool.
`pure.py` replays recorded bytes after perturbing MLX's global PRNG. `sweep.py`
runs four prompts (templated passage request, "Write.", empty turn, no chat
template) × temperature {0, 0.7, 1.0} × three fresh-entropy runs. Commit
7dc3ce8; all 36 texts in `spikes/diffusiongemma/sweep_results.json`.

**Result:** (1) Yes. Replay reproduces the text exactly at T=0 and T=0.7.
A generation consumes 11–14 KB of TrueRNG bytes; the device delivers ~50 KB/s.
(2) No, not with a prompt. Any real prompt pins the subject (always a sunset
over a valley) and entropy only rephrases (similarity 0.3–0.5). An empty user
turn lets entropy pick the topic (similarity ~0.1) but the register is an
assistant asking what you meant. No chat template gives degenerate word salad
at 2–3× the time. Temperature barely matters once a prompt is present.

**Decision:** Open. The mechanical property holds, but the instruct prior
squeezes entropy into wording. No base DiffusionGemma checkpoint is published.
Untested idea: a prompt that explicitly hands over the subject choice. The
Tauri swap (Python sidecar, entropy plumbing, streaming previews, bundling)
is not started.

---

## 2026-10-12 — DiffusionGemma: hand-over prompts and streaming preview

**Question:** (1) Can a prompt that delegates the subject choice let entropy
pick the topic while keeping prose? (2) Does mlx-vlm expose the canvas after
every denoising step, live, so the app's crystallising-text view survives a
swap?

**Done:** `sweep.py --prompts handover,handover-wild` ("Pick any subject you
like…" and "Choose a surprising subject at random…") × T {0, 1.0} × 3 runs;
texts in `spikes/diffusiongemma/handover_results.json`. `preview.py` drives
`stream_diffusion_generate` directly with unmasking drafts on and records the
event timeline. The TrueRNG dropped off USB mid-sweep; it came back at the
same path and the affected cells were rerun.

**Result:** (1) No. "Pick any subject" chose bioluminescence in 6 of 6 runs;
"choose at random" chose the wombat in 8 of 8. The random-subject request is
itself a fixed point of the instruct model; entropy still only varies wording
(similarity 0.3–0.4). (2) Yes. 13 live draft events over a 36 s generation,
~2.5 s apart, each with the full canvas and `[Mask]` for unrevealed positions.
The reveal is confidence-ordered and mostly left-to-right, so the preview is a
fill-in-the-blanks animation rather than Plaid's boil; a sidecar could decode
the raw canvas to show noise in the gaps. The public `generate` and
`stream_generate` batch drafts until the end, so a sidecar must use the
generator directly. The canvas carries a thinking-channel prefix to strip.

**Decision:** Open, with the technical questions now all answered. Entropy
authorship of the *subject* is not achievable with the instruct checkpoint by
prompting; it is a product call whether wording-only authorship (with far
better prose) is worth a 15 GB model, a Python sidecar, and a 32 GB floor.

---

## 2026-10-12 — Coherence spike: do anomaly-bank bytes measurably change DiffusionGemma's text?

**Question:** Is text generated from out-of-band TrueRNG bytes measurably
different from text generated from in-band bytes, and is the measurement
sensitive enough that a null result means something?

**Done:** Pre-registered A/B (spec
`docs/superpowers/specs/2026-10-12-coherence-spike-design.md`, plan
`docs/superpowers/plans/2026-10-12-coherence-spike.md`). 147 MiB
TrueRNG capture in two files (first cut at 51 MiB by an unplug) split offline
by a port of the app's coherence walk; 100 samples per condition (anomaly,
ordinary, CSPRNG, biased LCG), round-robin, DiffusionGemma at temperature 0
with the "short passage of prose" prompt, every random draw from the sample's
own reservoir. Primary metric GPT-2 log-perplexity; secondary denoising steps
and distinct-word ratio; permutation test, 20 000 shuffles. Code in
`spikes/coherence/`, artefacts in `spikes/coherence/runs/2026-10-12/`.
400 of 400 samples generated and scored, 0 excluded, 2.4 h.

Two deviations from the spec, both ledgered: the two-part capture, and the
walk is restarted every 1000 trials (~100 s of app time) because with the
app's never-resetting walk only 1 of 30118 trials was out of band.
That is a property of the statistic, not the device (per-trial z std 1.00):
σ = C_k/√k is correlated over a span ~k, so a single session gets only
~ln(n) independent chances to excurse. With resets, 4.5%
of trials were anomalous. The window of 1000 was chosen after seeing the
never-reset walk, before any generation, with the metric, test and decision
rule untouched; it is a convenience value, not a measured session length.
Known weakness: the ordinary arm is the first ~100 s of capture 1 (pools are
consumed front to back) while the anomaly arm spans both captures, so the
two TrueRNG arms are not time-matched. That can add a difference, not hide
one, so it does not touch this null; a replication should draw ordinary
bytes from trials adjacent to each excursion.

**Result:** Positive control (biased vs CSPRNG) d = +0.88, p < 0.0001: the
metric sees gross bias. Null calibration p = 0.47. Primary (anomaly vs
ordinary) d = −0.01, p = 0.92; anomaly vs CSPRNG d = +0.09, p = 0.51.
Secondary metrics all null (steps p = 0.38, distinct ratio p = 0.65).
Verdict line: "NO DETECTABLE DIFFERENCE: anomaly vs ordinary gpt2_nll p=0.920".

**Decision:** No detectable effect of anomaly-bank entropy on the generated
text, with a working positive control and a quiet null, at n = 100 per arm
(powered for d ≈ 0.5). The app's provenance stamp remains a statement about
where the bytes came from, not about the text. No replication capture is
warranted on this evidence. Side finding for the app: in one continuous
session the anomaly bank fills rarely, and then for a long stretch, because
of the walk's log-time correlation; resets are what give it fresh chances.
