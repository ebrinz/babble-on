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
