# Coherence spike

Does seeding DiffusionGemma from anomaly-bank bytes (TrueRNG bytes captured
while the coherence walk is outside its 95% envelope) measurably change the
text, compared with ordinary TrueRNG bytes? Pre-registered design:
`docs/superpowers/specs/2026-10-12-coherence-spike-design.md`.

## Pipeline

    .venv/bin/python capture.py runs/<id>/capture.bin --mib 128
    .venv/bin/python classify.py runs/<id>/capture.bin runs/<id>
    .venv/bin/python batch_build.py runs/<id>/anomaly.bin runs/<id>/ordinary.bin runs/<id>
    .venv/bin/python generate.py runs/<id>/batch.bin runs/<id>/manifest.json runs/<id>/batch.jsonl
    .venv/bin/python score.py runs/<id>/batch.jsonl runs/<id>/scored.jsonl
    .venv/bin/python analyze.py runs/<id>/scored.jsonl runs/<id>/manifest.json runs/<id>/report.md

Setup: `uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt`.
Tests: `.venv/bin/python -m pytest tests -q`.

## Run 2026-10-12

- **Capture:** 147 MiB of TrueRNG bytes in two files
  (the first was cut at 51 MiB when the dongle had to be unplugged; the second
  96 MiB file was concatenated after it). Device looks fair: per-trial monobit
  z has mean +0.0007 and standard deviation 1.00; 5.1% of single
  trials exceed 1.96.
- **Classification:** 30118 trials of 5120 bytes. With the
  app's never-resetting walk only **1** trial was out of band: σ = C_k/√k is
  correlated over a span ~k, so one long walk gets only ~ln(n) independent
  chances to excurse (20 simulated fair walks of this length: median 0.5%
  out of band, 8 of 20 at 0%). The classifier therefore restarts the walk
  every 1000 trials (~100 s of app time, i.e. a fresh session),
  a deviation from the spec. That gives 1343 anomalous
  trials (4.5%), 6.6 MiB.
- **Generation:** 400 samples, 0 failed, 21.6 s each
  (2.4 h total); 9220–17412 bytes consumed
  per sample (8–16 denoising steps).
- **Scoring:** 400 of 400 scored, none excluded.

| comparison | metric | n1 | n2 | mean1 | mean2 | d | p |
|---|---|---|---|---|---|---|---|
| positive_control | gpt2_nll | 100 | 100 | 3.7325 | 3.5551 | +0.883 | 0.0000 |
| null_calibration | gpt2_nll | 50 | 50 | 3.5682 | 3.5420 | +0.145 | 0.4708 |
| primary | gpt2_nll | 100 | 100 | 3.5748 | 3.5781 | -0.014 | 0.9197 |
| anomaly_vs_csprng | gpt2_nll | 100 | 100 | 3.5748 | 3.5551 | +0.093 | 0.5096 |
| positive_control | steps | 100 | 100 | 11.6300 | 11.6300 | +0.000 | 1.0000 |
| null_calibration | steps | 50 | 50 | 11.6200 | 11.6400 | -0.017 | 1.0000 |
| primary | steps | 100 | 100 | 11.7400 | 11.5500 | +0.132 | 0.3754 |
| anomaly_vs_csprng | steps | 100 | 100 | 11.7400 | 11.6300 | +0.082 | 0.5920 |
| positive_control | distinct_ratio | 100 | 100 | 0.7582 | 0.7565 | +0.069 | 0.6308 |
| null_calibration | distinct_ratio | 50 | 50 | 0.7565 | 0.7565 | +0.003 | 0.9877 |
| primary | distinct_ratio | 100 | 100 | 0.7593 | 0.7577 | +0.064 | 0.6469 |
| anomaly_vs_csprng | distinct_ratio | 100 | 100 | 0.7593 | 0.7565 | +0.110 | 0.4356 |

**Verdict:** NO DETECTABLE DIFFERENCE: anomaly vs ordinary gpt2_nll p=0.920

The positive control works (biased bytes raise GPT-2 log-perplexity by
d = +0.88, p < 0.0001), the null calibration is quiet (p = 0.47), and the
anomaly arm is indistinguishable from ordinary TrueRNG bytes on every metric
(primary d = −0.01). Anomaly bytes are also indistinguishable from the OS
CSPRNG. Denoising steps and distinct-word ratio show nothing either. The step
means of the biased and CSPRNG arms tie at 11.63 by coincidence (the
distributions differ slightly), and because steps are small integers the
permutation p-values of 1.0 in those rows are a floor effect, not a bug.

**Caveats a replication should fix.** The ordinary arm is not time-matched to
the anomaly arm: `batch_build.py` consumes each pool front to back, so the
100 ordinary samples are the first ~100 s of capture 1 while the anomaly
samples are spread across both captures. A confound of that kind can add a
difference, not remove one, so it cannot explain this null, but it is the
first objection a positive result would have met; a replication should draw
ordinary bytes from trials adjacent to each excursion. The reset window of
1000 trials was chosen after seeing the never-reset walk (before any
generation, with the metric, test and decision rule untouched) and is a
convenience value, not a measured session length.

## Label audit 2026-10-13

`label_audit.py` asks whether the trials the walk flags differ from ordinary
trials on anything the walk does not select on. It re-labels the 2026-10-12
capture (`capture_all.bin`, 30 118 trials, reset window 1000) and computes
nine held-out per-trial features: bit autocorrelation at lags 1, 2, 8 and 64,
byte autocorrelation at lag 1, byte-histogram χ², the NIST runs statistic,
longest run of ones, and the NIST DFT statistic. Mean bit is reported only as
a reference, since the walk selects on it. Five comparisons, each a 20 000
shuffle permutation test per feature with Bonferroni α = 0.01/9:

    .venv/bin/python label_audit.py runs/2026-10-12/capture_all.bin runs/2026-10-12/label_audit.json

| comparison | n | held-out features below α |
|---|---|---|
| anomalous vs all ordinary | 1343 vs 28 775 | none (all \|d\| ≤ 0.03, p ≥ 0.24) |
| anomalous vs adjacent in-band trials | 1343 vs 179 | none (\|d\| ≤ 0.13, p ≥ 0.11) |
| adjacent vs far ordinary | 179 vs 28 596 | none |
| biased LCG vs ordinary (positive control) | 1343 vs 28 775 | 8 of 9 (d from 4.7 to 642) |
| ordinary even vs odd (null) | 14 388 vs 14 387 | none |

Mean bit shows only the selection: anomalous trials average 0.50007 against
0.49999 for ordinary (d = 0.03; excursions were 749 positive, 594 negative,
so the signs largely cancel), and the adjacent trials sit at 0.49941 because
the trial that ends an excursion pushes the other way (d = 0.25, p = 0.001).
The biased control averages 0.375. The deviation the label marks is about
7 × 10⁻⁵ per bit, three orders of magnitude below the control, and nothing
else about those bytes differs from their neighbours.

A compressibility feature was dropped during the run: every 5120-byte trial
compresses to exactly 5131 bytes, so it has no resolution on random blocks,
and the zero variance had inflated its effect size.

## Dose-response 2026-10-13

How large a bit bias does the text metric need before it sees anything?
`dose_build.py` injects a known bias into the 2026-10-12 ordinary TrueRNG
bytes, P(bit = 1) = 0.5 + δ for δ ∈ {0, 10⁻⁴, 10⁻³, 10⁻², 3·10⁻², 10⁻¹},
plus the app's biased LCG as the ceiling, 50 samples per arm, interleaved so
every arm spans the same stretch of carrier. Generation and scoring are the
main spike's unchanged; `dose_analyze.py` compares each arm with δ = 0.

    .venv/bin/python dose_build.py runs/2026-10-12/ordinary.bin runs/2026-10-13-dose --per-condition 50
    .venv/bin/python generate.py runs/2026-10-13-dose/batch.bin runs/2026-10-13-dose/manifest.json runs/2026-10-13-dose/batch.jsonl
    .venv/bin/python score.py runs/2026-10-13-dose/batch.jsonl runs/2026-10-13-dose/scored.jsonl
    .venv/bin/python dose_analyze.py runs/2026-10-13-dose/scored.jsonl runs/2026-10-13-dose/manifest.json runs/2026-10-13-dose/report.md

| δ | mean bit | GPT-2 nll d vs δ=0 | p | steps p | distinct p |
|---|---|---|---|---|---|
| 10⁻⁴ | 0.5001 | −0.11 | 0.58 | 0.22 | 0.65 |
| 10⁻³ | 0.501 | −0.09 | 0.65 | 0.47 | 0.65 |
| 10⁻² | 0.510 | −0.38 | 0.06 | 0.37 | 0.14 |
| 3·10⁻² | 0.530 | −0.18 | 0.39 | 0.61 | 0.53 |
| 10⁻¹ | 0.600 | −0.29 | 0.15 | 0.95 | 0.83 |
| LCG | 0.375 | +0.66 | 0.0013 | 0.81 | 0.93 |

No dose up to δ = 0.1 (60% ones) moves any metric at α = 0.01, and the
effect sizes are not monotonic in δ. The LCG separates at d = +0.66, as in
the main run. The difference is structure, not mean: the LCG masks every
byte to 6 bits, so all canvas token ids fall in the bottom quarter of the
vocabulary and repeat with the generator's short low-bit periods, whereas
a biased carrier still yields token ids spread over the whole vocabulary,
merely skewed. The confidence sampler overwrites either canvas in 8–16
steps, and only the LCG's structure survives to the text.

**What a replication must know.** 34 samples (indices 114–160, spread
evenly over the arms) failed on the first pass with a Metal out-of-memory
error while another process held GPU memory; since each sample is a pure
function of its reservoir, they were regenerated on the same bytes and all
350 are present. GPT-2 log-perplexity on 256-token prose at n = 50 per arm
is powered for d ≈ 0.57; a dose effect smaller than that is not excluded.

## Dose-response through the harness sampler 2026-10-13

The same doses rerun through the harness's `sample_canvas` on the new MLX
backend (`harness_dose.py`, run in `harness/.venv`), where every draw is a
tape uniform: the initial canvas, one categorical draw per position per step
at the Transformers schedule T 0.8 → 0.4, and the renoise. 50 samples per
arm, 99 328-byte reservoirs, 350/350 generated with no errors in 2.8 h.

    .venv/bin/python dose_build.py runs/2026-10-12/ordinary.bin runs/2026-10-13-dose-harness --per-condition 50 --sample-bytes 99328
    ../../harness/.venv/bin/python harness_dose.py runs/2026-10-13-dose-harness/batch.bin runs/2026-10-13-dose-harness/manifest.json runs/2026-10-13-dose-harness/batch.jsonl

| δ | GPT-2 nll mean | d vs δ=0 | p | steps p | distinct p |
|---|---|---|---|---|---|
| 0 | 3.543 | | | | |
| 10⁻⁴ | 3.451 | −0.37 | 0.07 | 1.00 | 0.36 |
| 10⁻³ | 3.413 | −0.59 | 0.004 | 0.19 | 0.08 |
| 10⁻² | 3.442 | −0.37 | 0.07 | 0.24 | 0.16 |
| 3·10⁻² | 3.528 | −0.06 | 0.77 | 0.015 | 0.09 |
| 10⁻¹ | 3.479 | −0.26 | 0.19 | 0.29 | 0.29 |
| LCG | 3.615 | +0.32 | 0.12 | 0.88 | 0.83 |

No dose effect. The one p below 0.01 (δ = 10⁻³) does not survive a
correction for 18 comparisons and sits between two doses that show nothing;
every dose reads below δ = 0 because they share that arm, whose mean happens
to be the second highest. Unlike the T = 0 run, the LCG ceiling does not
separate either.

**Why.** Re-running three δ = 0 and three LCG samples with the traces kept:
of 7 673 committed draws at δ = 0 the tape chose a token other than the
argmax 0 times, and of 8 295 under the LCG 2 times, although every LCG
uniform is below 0.25. The median entropy at the moment of commit is 0.0000
nats. The sampler's entropy-bound selection only commits positions whose
distribution is already a point mass, so the categorical draw is a formality;
the tape reaches the text only through the canvas noise at uncommitted
positions, which the model overwrites. Sampling at T > 0 does not by itself
give entropy a vote.
