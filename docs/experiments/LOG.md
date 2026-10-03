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

---
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
