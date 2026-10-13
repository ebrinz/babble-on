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
