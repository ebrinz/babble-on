# babble-on — Coherence Spike: does anomalous entropy measurably change the text? — Design

**Date:** 2026-10-12
**Status:** Approved (design); implementation plan to follow
**Builds on:** `2026-07-09-anomaly-bank-design.md` (anomaly bank, shipped) and
`spikes/diffusiongemma/` (entropy pool: output is a pure function of TrueRNG
bytes; sweep results 2026-10-12)

## One line

A pre-registered A/B experiment: generate DiffusionGemma text from
anomaly-bank bytes, ordinary TrueRNG bytes, a CSPRNG, and a deliberately
biased RNG, then test whether the anomaly condition is measurably different
on a fixed metric, with a positive control and a null calibration so a result
means something.

## The question

The app's premise is that bytes captured while the coherence walk is outside
its 95% envelope are special, and that seeding a generation from them makes
the text carry that anomaly. Physics says the hardware RNG is i.i.d. uniform
whether or not the walk is excursing, so the honest prior is no difference.
This spike asks the question properly: is there any measurable difference in
the generated text between anomaly bytes and ordinary bytes, and is the
measurement sensitive enough that a null result is informative?

## Goals

1. A reproducible pipeline: capture → classify → build → generate → score →
   analyse, every stage a pure function of files on disk.
2. Pre-registered metrics, comparisons, and decision rule, written here before
   any generation runs.
3. A positive control (biased bytes) and a null calibration (CSPRNG split in
   half) so the metric's sensitivity and the false-positive floor are both
   measured, not assumed.
4. An entry in `docs/experiments/LOG.md` whatever the outcome.

## Non-goals

- Changing the app. Nothing under `src-tauri/`, `src/`, or `diffusion-rs/`
  is touched.
- Proving or disproving the premise in general. One capture, one device, one
  model. A positive result triggers replication, not a claim.
- Plaid. It is the more entropy-sensitive substrate (DDIM makes the text a
  pure function of the seed and every token is entropy-authored), but the
  swap decision is about DiffusionGemma, its output is prose so perplexity is
  a lower-variance metric, and its confidence sampler gives a native
  coherence measure. A Plaid arm is a possible follow-up, not part of this.

## Key decisions (and the reasoning trail)

- **Substrate: DiffusionGemma 26B-A4B (MLX 4-bit) via mlx-vlm, driven by the
  spike's entropy pool.** `spikes/diffusiongemma/entropy.py` routes the
  initial canvas, every per-step re-noise, and the sampler seed through a
  pool; replaying recorded bytes reproduces the text exactly (verified
  2026-10-12). Temperature 0, prompt "Write a short passage of prose." via
  the chat template, 256 tokens, model-default 48 max denoising steps. With
  this prompt entropy varies wording only (sweep result), which is exactly
  the kind of subtle difference the experiment has to be able to see or
  rule out.
- **One capture, split offline.** Record one long raw TrueRNG stream to a
  file, then classify its trials with a Python port of the app's
  cumulative-deviation walk. Anomaly and ordinary bytes then come from the
  same device, same session, same thermal state. (Rejected: dumping the live
  app's bank — would need engine changes and couples the experiment to UI
  timing.)
- **Trials by byte count, not wall clock.** The app finalises a trial every
  100 ms with at least 2048 bits (`src-tauri/src/stats.rs` `TRIAL_INTERVAL`,
  `TRIAL_MIN_BITS`). Offline there is no clock, so a trial is a fixed 5120
  bytes, which is what 100 ms delivers at the measured ~50 KB/s. The monobit
  z-score `(ones − n/2) / √(n/4)`, the walk `C_k = Σz`, `σ = C_k/√k`, and
  the threshold `Z95 = 1.959_963_98` with the app's `|σ| >= Z95` comparison
  (`Band::of`) are copied verbatim. A trial's bytes are "anomalous" when σ
  after that trial is at or beyond the 95% band, mirroring
  `deposit_if_anomalous` banking the tick's drained bytes when the snapshot
  band is not `Inside`.
- **Walk never resets** during the capture, same as the app between resets. If
  the device has a small constant bias, σ drifts out and stays out; the
  classifier reports the anomalous fraction, and a fraction above 50% is
  itself a finding (reported, not hidden) that the split is bias-dominated.
- **Per-sample reservoir of 50 KiB.** A generation consumes 4 bytes of seed
  plus 1024 bytes per canvas (initial + one per denoising step). Measured
  runs used 11–14 KiB; the worst case at 48 steps is 50 180 bytes, so each
  sample owns a fixed 51 200-byte slice of the batch file and the pool
  replays from it. The pool raises if a slice is exhausted; the generator
  records that sample as failed rather than falling back to software noise.
  Bytes actually consumed are recorded per sample.
- **Four conditions, interleaved.**
  - **A anomaly:** TrueRNG bytes from out-of-band trials.
  - **B ordinary:** TrueRNG bytes from in-band trials.
  - **C csprng:** `os.urandom`.
  - **D biased:** the app's bad source, reproduced exactly: LCG
    `x = (x*1103515245 + 12345) mod 2^32`, byte `= ((x >> 16) & 0xff) & 0x3f`,
    seed `0x12345678` (`src-tauri/src/source.rs` `run_software`). First
    eight bytes are `31 07 1d 14 2c 09 13 07`.
  Samples are laid out round-robin A, B, C, D, A, B, … in a single batch
  file so thermal and clock drift during the multi-hour run spread evenly
  across conditions. The generator reads only the batch file and the sample
  size; condition labels live in the manifest.
- **Sample size: 100 per condition.** 400 generations at 20–30 s each is
  roughly 3 hours unattended. 100 per arm detects a medium effect (d ≈ 0.5)
  with ~94% power at α = 0.05 and a small one (d ≈ 0.3) about half the time.
  TrueRNG requirement: 100 × 50 KiB ≈ 5 MiB per TrueRNG arm. At ~5%
  out-of-band trials, the anomaly pool needs a capture of about 100 MiB
  (~35 min at 50 KB/s); the capture target is 128 MiB.
- **Primary metric: GPT-2 log-perplexity** (mean per-token negative
  log-likelihood under `gpt2`, 124M, in nats) of the generated text. GPT-2 is
  independent of the Gemma family. Log-perplexity is closer to normal than
  perplexity, so it is what the test runs on.
- **Secondary metrics (exploratory, reported, not decisive):**
  - *denoising steps*: how many canvases the confidence sampler needed before
    every position was accepted, derived exactly from bytes consumed:
    `(consumed − 4) / 1024 − 1`. This is the native "coherence during
    generation" measure.
  - *distinct-word ratio*: unique whitespace-split words / words.
- **Test: permutation test on the difference of means**, 20 000 shuffles,
  two-sided, plus Cohen's d. numpy only. (Rejected: Welch's t-test alone —
  fine, but the permutation test makes no normality assumption and costs
  nothing.)
- **Pre-registered comparisons and decision rule:**
  1. **Positive control** D vs C on the primary metric must reach p < 0.001.
     If it does not, the metric cannot see even gross bias and the experiment
     is *inconclusive*, full stop.
  2. **Null calibration** C₁ vs C₂ (the CSPRNG samples split by parity of
     their rank within the condition) is expected at p > 0.05. If it fails,
     report it and treat the primary result with suspicion.
  3. **Primary claim** A vs B on log-perplexity at α = 0.01. A vs C is
     reported alongside.
  4. Secondary metrics are reported with their p-values but do not change the
     verdict.
  5. A primary result below α is labelled *candidate effect*, not *effect*,
     until a second independent capture reproduces it.

## Pipeline

```
capture.py      TrueRNG ──► capture.bin  (128 MiB raw, ~45 min, incremental)
classify.py     capture.bin ──► anomaly.bin, ordinary.bin, classify.json
batch_build.py  anomaly.bin ordinary.bin ──► batch.bin (400 × 51 200 B, round-robin) + manifest.json
generate.py     batch.bin manifest.json ──► batch.jsonl  (mlx-vlm + entropy pool; resumable)
score.py        batch.jsonl ──► scored.jsonl  (+ gpt2_nll, distinct_ratio, steps)
analyze.py      scored.jsonl manifest.json ──► report.md  (comparisons table, verdict)
```

Every stage reads and writes files under `spikes/coherence/runs/<run-id>/`,
so a run can be re-scored or re-analysed without regenerating. `*.bin` files
are git-ignored (repo rule); `classify.json`, `manifest.json`, `scored.jsonl`
and `report.md` are committed.

## Interfaces (the contracts tasks build to)

- `batch.bin`: concatenation of `N` samples × 51 200 bytes.
- `manifest.json`: `{"sample_bytes": 51200, "prompt": "Write a short passage
  of prose.", "temperature": 0.0, "max_tokens": 256, "samples": [{"index": 0,
  "condition": "anomaly"}, …]}`.
- `batch.jsonl`, one line per sample, in file order:
  `{"index": 0, "text": "...", "consumed": 14340, "elapsed_s": 24.1,
  "error": null}`.
- `scored.jsonl`: `batch.jsonl` lines plus `"gpt2_nll": 3.12,
  "distinct_ratio": 0.71, "steps": 13`. `gpt2_nll` is `null` for failed or
  empty samples and such samples are excluded from every comparison, with the
  exclusion count in the report.
- `report.md`: one table, rows = comparisons × metrics, columns = n₁, n₂,
  mean₁, mean₂, d, p; then the verdict line from the decision rule.

## Risks

- **TrueRNG USB dropout** (seen 2026-10-12 as `Device not configured`).
  `capture.py` writes incrementally and reports bytes captured; a short
  capture is usable if the anomaly pool clears 100 × 51 200 bytes, and
  `classify.py` fails loudly if it does not.
- **Too few anomalous trials.** Under the null about 5% of trials are out of
  band, clustered. 128 MiB ≈ 26 000 trials ≈ 1 300 anomalous trials ≈ 6.5
  MiB, enough for 100 samples with margin.
- **Reservoir exhaustion.** Bounded by the 48-step cap; 51 200 ≥ 50 180.
  A failed sample is recorded and excluded, never silently replaced.
- **Runtime.** About 3 hours of generation. `generate.py` appends per sample
  and skips indices already present in `batch.jsonl`, so a crash or a second
  dropout mid-run costs nothing but a restart.
- **Metal non-determinism.** Replay reproduced text exactly in the spike, so
  the model is deterministic enough on this machine; no further handling.

## Out of scope, noted for later

- A second capture for replication if the primary comparison crosses α.
- A Plaid arm via the candle engine (DDIM, no prompt) as the maximally
  entropy-sensitive substrate.
- Scoring with a larger model.
