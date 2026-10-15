# Experiment log

Append-only record of every harness run and every manual experiment on the
coherence → DiffusionGemma question. The harness appends an entry
automatically when `analyze_run` finishes (`babble-harness run` / `analyze`);
humans and Claude add entries for anything done by hand (recording sessions,
hardware changes, failed attempts, decisions). Newest at the bottom.

Format of an automatic entry: date · run name · model · conditions and counts
· the scalars that moved (Mann–Whitney p < 0.05) · probe results · a link to
the report. Reports committed under `docs/experiments/runs/<name>/` keep their
figures (`report-assets/*.svg`).

The experiment stream and the application stream are separate for now; this
log is where experiment outcomes are meant to accumulate until they are solid
enough to shape the app.

Entries dated before 2026-10-03 were backfilled from git history, the docs,
and session notes when the application's own experiments were folded into
this log on 2026-10-13.

---
## 2026-06-24 — Candle port of Plaid-1B validated against the torch sidecar

**Question:** Does the all-Rust candle engine reproduce the Python reference
forward pass closely enough to replace it?

**Done:** `cargo run` validate mode compares the candle forward output against
the torch sidecar on the same input.

**Result:** max |Δ| < 1e-2. The candle engine became the app's only generation
backend (commit 4335ce8); the Python sidecar stayed as the reference.

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

## 2026-07-14 — Rolling window for the coherence walk (rejected)

**Question:** Would a fixed-x-scale 2-minute rolling window read better than
the anchored envelope funnel on long sessions?

**Done:** Built it (commit 467df97), ran it live.

**Result:** Rejected. The envelopes and scene visibly translating read worse
than the whole session compressing into a static funnel. Reverted in 1fc8284.
Do not re-propose a scrolling window; the alternative offered and declined was
a σ-normalized view with fixed horizontal bands.

## 2026-10-03 21:31 · `dry-run-tiny` · model `tiny`

- conditions: in_band=48, out_band=48, prng=48, remote=48; compared `in_band` vs `out_band`
- scalars that moved (p < 0.05): `logit_lens_mean` 0.742 → 0.721 (p=0.00064)
- probe `scalars`: accuracy 0.55 vs null 0.50±0.06 (p=0.26, n=96)
- probe `pooled_first`: accuracy 0.58 vs null 0.49±0.07 (p=0.095, n=96)
- probe `pooled_last`: accuracy 0.58 vs null 0.49±0.07 (p=0.11, n=96)
- report: [docs/experiments/runs/2026-10-03-dry-run-tiny/report.md](runs/2026-10-03-dry-run-tiny/report.md)
- outcome: **pipeline dry run, no scientific content.** Model = `tiny` (random weights), streams = simulated
  (device: healthy for 100 s then P(bit=1)=0.5004, which drifts the walk to +3.5σ; remote: healthy PRNG). Every
  condition is therefore a true null, and the report behaves like one: the probes sit on their permutation nulls
  (p = 0.26 / 0.095 / 0.11) and exactly one of 15 scalars crosses p < 0.05 (`logit_lens_mean`, p = 6e-4) — with
  a random model this is a reminder to Bonferroni-correct and to compare against `prng`/`remote` before believing
  a hit, not a finding. What the run proves: labelling, seed cutting (in-band 9606 / out-band 1422 seeds
  available at the tiny budget), the real Transformers adapter path, figures, and the log entry all work end to
  end. Next: the same command on a real `.bbrec` with `--model diffusion_gemma --quant …` on a GPU box.

## 2026-10-03 · method change · recording headers now carry the walk state

- Manual entry (no run). A code review of the branch found that the offline coherence replay
  (`harness/babble_harness/coherence.py`) started every recording from a zero walk, while the app's
  walk carries history when **● record** is pressed mid-session — so in-band/out-band labels could
  differ from what the app's bank actually deposited. Fixed at the source: `.bbrec` headers now store
  `walk_cum`, `walk_k`, the partial trial and the time since the last trial, and the replay continues
  from them. Reset and source switches now end a recording, since both restart the walk.
- Consequence: any `.bbrec` written before this change replays from zero and is only exact if the
  recording began on a fresh session. None exist yet (the only recordings so far are simulated).
- Also fixed in the same review: a DiffusionGemma generation now refuses to run on a short seed
  instead of silently using a PRNG tape (the UI flags any non-entropy seed), the bank is spent only
  after the engine is ready, and a dead sidecar is respawned.

## 2026-10-06 · spike · 4-bit DiffusionGemma on Apple Silicon (MLX) as a local bench

- Manual entry (throwaway spike, no harness code changed). Question: can the 26B-A4B model run on the
  M5 / 32 GiB Mac with the entropy contract intact? bf16 (51.7 GB) doesn't fit; NVFP4 and bnb4 need
  CUDA; GGUF/llama.cpp owns its own RNG and sampler, which would take the draws away from the
  `EntropyTape` — ruled out. Tried `mlx-community/diffusiongemma-26B-A4B-it-4bit` (mlx-vlm 0.7.6;
  experts 4-bit, attention/dense MLP/router/embeddings 8-bit; 15 GB on disk) behind the `Denoiser`
  protocol, driving the unchanged `sample_canvas` loop.
- Outcome: works. Peak 17.6 GB, ~0.5 s per forward, ~1.1–1.4 s per sampler step, 11–15 steps to
  early-stop on a 256-token canvas (12–17 s per sample). Forward pass bit-identical on repeat (with
  and without self-conditioning); same tape ⇒ identical trajectory step by step; different tapes ⇒
  different canvases. Captures carried over: hidden [5, 256, 2816] at layers 6/12/18/24/30, router
  counts [30, 128] (top-8), logit-lens agreement. Text is coherent. No QRNG on hand: PCG64 tapes plus
  one `os.urandom` tape stood in for device bytes.
- Not yet checked: fidelity to bf16 (needs the CUDA box, or the 8-bit MLX build as an intermediate);
  logit-lens agreement reads 0 at layers 6–18 on step 1 (plausible on a random canvas, unverified);
  output opens with an empty `thought` channel marker, which text metrics should strip.
- Implication: a 4-bit MLX adapter is a viable local bench. Results from it describe the quantised
  model, so every condition must run on the same build, and any finding should be re-run on bf16.

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

## 2026-10-13 — Coherence label audit: do the walk's anomalous trials differ on anything held out?

**Question:** The coherence spike found no text-level difference. Before
spending more generation time: are the bytes the walk flags as anomalous
distinguishable from ordinary bytes on any property the walk does not select
on? If not, the label is a selection artefact and no downstream test can see
anything.

**Done:** `spikes/coherence/label_audit.py` on the existing 2026-10-12
capture (154 MiB, 30 118 trials, no device needed). Nine held-out per-trial
features (bit autocorrelation at four lags, byte autocorrelation, byte χ²,
NIST runs, longest run, NIST DFT), mean bit as the selected-on reference.
Five comparisons by permutation test, 20 000 shuffles, Bonferroni α = 0.0011:
anomalous vs ordinary, anomalous vs the 179 in-band trials adjacent to an
excursion (time-matched control), adjacent vs far, biased-LCG positive
control, even/odd null. Output `spikes/coherence/runs/2026-10-12/label_audit.json`;
table in the spike README.

**Result:** No held-out feature differs in any of the three real comparisons
(anomalous vs ordinary all |d| ≤ 0.03, p ≥ 0.24; vs adjacent |d| ≤ 0.13,
p ≥ 0.11). The positive control lights 8 of 9 features at d from 4.7 to 642;
the null split is quiet. The selection itself is tiny: anomalous trials
average 0.50007 ones per bit vs 0.49999, about 7 × 10⁻⁵, three orders of
magnitude under the LCG control's 0.125. A compressibility feature was
dropped mid-run as degenerate (every trial compresses to 5131 bytes).

**Decision:** On this capture the anomaly label marks nothing but the
expected tail of a fair device: its only measurable property is the mean-bit
excursion it was selected on, and that excursion is ~10⁻⁴ per bit. The
text-level null of 2026-10-12 is therefore the expected outcome, not a
sensitivity failure. Next, with the TrueRNG reattached: a dose-response run
injecting known bias levels from 10⁻⁴ up to the LCG's level to find the
detection threshold of text and mechanistic metrics, so "coherence of
magnitude X would be visible" can be stated instead of searched for.

## 2026-10-13 — Coherence dose-response: how much bit bias can the text metric see?

**Question:** The label audit put the walk's anomaly deviation at ~7 × 10⁻⁵
per bit. What magnitude of deviation would be visible in DiffusionGemma's
text at all, through the current entry point (bytes → canvas token ids)?

**Done:** `spikes/coherence/dose_build.py` injects P(1) = 0.5 + δ for
δ ∈ {0, 10⁻⁴, 10⁻³, 10⁻², 3·10⁻², 10⁻¹} into the 2026-10-12 ordinary TrueRNG
bytes, with the app's biased LCG as the ceiling; 50 samples per arm,
interleaved; generation, scoring (GPT-2 log-perplexity, steps, distinct-word
ratio) and permutation test unchanged from the main spike; `dose_analyze.py`
compares each arm with δ = 0. No device needed. 350/350 generated (34 Metal
out-of-memory failures regenerated on identical bytes), 1.9 h. Artefacts in
`spikes/coherence/runs/2026-10-13-dose/` (manifest, scored.jsonl, report.md);
table in the spike README.

**Result:** No dose moves any metric at α = 0.01. GPT-2 nll d vs δ = 0:
−0.11, −0.09, −0.38, −0.18, −0.29 for the five doses (p from 0.06 to 0.65),
not monotonic. The LCG separates at d = +0.66, p = 0.0013, replicating the
main run. Steps and distinct ratio flat everywhere. Powered for d ≈ 0.57.

**Decision:** Through the current entry point, mean-bit bias is invisible
to text perplexity up to at least δ = 0.1, three orders of magnitude above
anything the coherence walk marks. What the metric does see is structure:
the LCG's 6-bit bytes confine canvas token ids to the bottom quarter of the
vocabulary with short periods. So "coherence" as a mean-bit excursion
cannot reach the text through random canvas tokens, whatever its size; if
the app is to make inference respond to byte statistics, the bytes must
enter through something the sampler integrates rather than hashes (e.g. the
confidence threshold, temperature, or unmasking order), or the question must
move to structure-sensitive features of the stream. Open, product call.

## 2026-10-13 · method change · MLX backend for the harness (`--model mlx`)

- Manual entry. Implemented the 2026-10-06 MLX backend spec on the M1 Pro / 32 GiB:
  `harness/babble_harness/models/mlx_gemma.py` wraps mlx-vlm 0.7.6 behind the `Denoiser`
  protocol; `mlx_tiny` rebuilds the tiny random-weight model in MLX with the same weights.
- Parity (MLX on CPU vs Transformers, tiny model): logits within 1e-5 with and without
  self-conditioning, hidden states within 1e-5, router counts and logit-lens equal, identical
  canvases at every step from the same tape. GPU repeat bit-identical; GPU vs CPU within 1e-2.
  Real 4-bit weights (opt-in test): prompt tokens identical to the Google processor's,
  self-conditioning in bf16, repeat bit-identical, 4-bit default with 236 8-bit overrides.
- Deviations from the spec: (1) the CPU parity tests disable MLX compilation, because MLX
  JIT-compiles fused CPU kernels with the system toolchain and this Mac's Command Line Tools
  cannot link against the macOS 27 SDK; the arithmetic is the same, unfused. (2) The config
  translation refuses a `None` only where mlx-vlm types the field as non-optional:
  `use_bidirectional_attention = None` is legal and only affects image tokens. (3) `describe()`
  reads quantisation from the checkpoint's config.json rather than counting modules, since the
  4-bit build has more 8-bit modules than 4-bit ones.
- Implication: the coherence dose-response can now run through the harness sampler, where every
  categorical draw is a tape uniform, instead of mlx-vlm's seeded PRNG at T = 0.

## 2026-10-14 — Coherence dose-response through the harness sampler (MLX, T 0.8 → 0.4)

**Question:** The T = 0 dose run (mlx-vlm, entropy only in the canvas) saw no
bias up to δ = 0.1. Does routing every categorical draw through the tape, as
the harness sampler does, let mean-bit bias reach the text?

**Done:** `spikes/coherence/harness_dose.py` on the new `--model mlx` backend
(commit 09868c0), driver commit fa32172. Same six doses plus the LCG ceiling,
50 per arm, 99 328-byte reservoirs from the 2026-10-12 ordinary TrueRNG pool,
default harness sampler (T 0.8 → 0.4, entropy bound 0.1, early stop).
350/350, 0 errors, 2.8 h. Artefacts in
`spikes/coherence/runs/2026-10-13-dose-harness/`; table in the spike README.
A follow-up re-ran three δ = 0 and three LCG samples keeping the traces.

**Result:** No dose effect on GPT-2 nll, steps or distinct ratio; the lone
p = 0.004 at δ = 10⁻³ fails a correction for 18 comparisons and is not
monotonic. The LCG ceiling also fails to separate (d = +0.32, p = 0.12).
Mechanism, measured: of 7 673 committed draws at δ = 0 the tape picked a
non-argmax token 0 times, and 2 of 8 295 under the LCG, whose uniforms are
all below 0.25. Median entropy at commit is 0.0000 nats. The entropy-bound
selector commits only positions that are already certain.

**Decision:** With the reference DiffusionGemma sampler, the device's bytes
never choose a word: they shape the noise the model writes over, and the
categorical draw is a formality. Mean-bit coherence cannot reach the text
through this sampler at any dose tested, and the positive control shows even
gross structure barely does. Giving entropy a real vote needs a sampler
change that departs from the reference, e.g. committing positions in a
tape-chosen order or at a looser entropy bound so draws land where the model
is uncertain. That is a design decision about what the app claims, open.

## 2026-10-14 — Prompt composition: TrueRNG bytes write DiffusionGemma's brief

**Question:** Every dose run showed DiffusionGemma's content is fixed by the
prompt and the bytes only touch wording. Can the bytes author the content
by composing the prompt instead?

**Done:** `spikes/prompt_compose/compose.py`: 16 TrueRNG bytes choose a
subject, setting, mood and form from four 32-entry lists ordered from
ordinary to strange (harness bytes → uniform → index contract); the passage
is generated through the harness sampler on the MLX backend, tape from the
following bytes. 12 passages from the 2026-10-12 capture. Texts in
`spikes/prompt_compose/runs/2026-10-14.jsonl`; write-up in the spike README.

**Result:** 12/12 passages follow every drawn slot, with good prose. Mean
pairwise word overlap 0.08, against 0.3–0.5 for fixed-prompt runs. 67 s per
passage (22.5 steps on average vs ~14 for a plain prompt).

**Decision:** Works; this is the content lever for the DiffusionGemma
engine. The device chooses the subject honestly and the app can show the
drawn brief. The coherence coupling (ordered lists, so a high-skewed stream
drifts toward the strange) is designed, not a finding. Open: whether the app
adopts it, list design, and whether Cola-DLM (smoke test next) makes it
unnecessary.

## 2026-10-14 — Cola-DLM smoke test on Apple Silicon

**Question:** Does Cola-DLM (continuous latent diffusion, ~2B, Apache-2.0)
run on MPS at a usable speed, and does its latent noise change the content
of the text, which DiffusionGemma's noise never did?

**Done:** `spikes/cola_dlm/smoke.py` with the vendor code at 7d1daee
(gitignored) and the 9.3 GB weights. Every block's Gaussian noise drawn from
TrueRNG bytes (2026-10-12 capture) by inverse normal CDF; greedy decoding.
Determinism check, then 6 samples × 2 open prompts at 128 tokens. A
precision check compared the MPS workaround with CUDA-style autocast on the
same bytes. Samples in `spikes/cola_dlm/runs/smoke.jsonl`; write-up in the
spike README.

**Result:** Runs on MPS after two spike-local dtype fixes (the code assumes
CUDA autocast); the CUDA-style autocast run gives identical text, so the fix
is neutral. 35–44 s per 128 tokens, 8 KB of bytes. Same bytes → same text.
The noise picks the subject: six unrelated stories per prompt, word overlap
0.10 (DiffusionGemma fixed prompt 0.3–0.5). Prose is poor, base-model web
text, often ungrammatical, closer to Plaid than to DiffusionGemma.

**Decision:** Cola-DLM has the Plaid property (entropy authors content)
without better prose, so it does not replace prompt composition on
DiffusionGemma for the app. Not pursued further unless an instruction-tuned
or larger checkpoint appears. Possible later use: entropy-authored seed text
from Cola-DLM as raw material for DiffusionGemma, which would keep content
authorship in the noise rather than in designed word lists.

## 2026-10-14 — Device-composed prompts shipped in the DiffusionGemma engine

**Question:** Can the app let the TrueRNG write DiffusionGemma's brief, so
the device authors content rather than wording (prompt-composition spike,
2026-10-14)?

**Done:** Spec `docs/superpowers/specs/2026-10-14-prompt-composition-design.md`,
plan `docs/superpowers/plans/2026-10-14-prompt-composition.md`. The spike's
composer moved to `harness/babble_harness/compose.py`; `serve.py` takes
`compose: true`, spends the first 16 entropy bytes on the brief and emits a
`brief` message; the app draws 16 more seed bytes when the prompt is blank
with DiffusionGemma selected, sends the flag, and shows the brief above the
text with bank-drawn picks in gold. Commits 6567f91, a7c26e9, a976717,
f7a8f44 (branch `feat/prompt-composition`). Two things the plan missed, both
found by tests: the Rust sidecar reader whitelisted message types and would
have dropped `brief`; and the pick class `bank` collided with the
anomaly-bank meter's flex rule.

**Result:** Tests: harness 82, src-tauri cargo 35, bbrec 13, vitest 8,
headless UI, all green; the stub-sidecar round trip sees `brief` before the
first step. Judged by eye: _(fill in after running the app with the TrueRNG:
does the brief line read well, do bank-drawn picks stand out, does the
composed text follow the brief as in the spike)_.

**Decision:** Shipped. Open: list design (the spike's lists are a first
draft) and whether a typed prompt should be allowed to extend a composed
brief (spec non-goal for now).
