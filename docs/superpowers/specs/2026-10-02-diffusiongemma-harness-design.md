# babble-on — DiffusionGemma + Coherence Experiment Harness — Design

**Date:** 2026-10-02
**Status:** Approved (design); phase 1 implemented alongside this spec
**Builds on:** `2026-06-22-babble-on-design.md`, `2026-07-09-anomaly-bank-design.md`

## One line

Replace Plaid-1B with Google's DiffusionGemma as the generation model, and add
a separate, offline **experiment harness** that drives DiffusionGemma's masked
diffusion sampler entirely from recorded entropy — labelled *in-coherence* or
*out-of-coherence* by the same walk the observatory uses — and captures
mechanistic-interpretability traces so the two conditions can be compared.

## Goals

1. **Model swap:** DiffusionGemma (`google/diffusiongemma-26B-A4B-it`,
   Apache 2.0, 26B MoE / 3.8B active, canvas of 256 tokens, masked discrete
   diffusion) becomes the target model.
2. **Harness, separate from the UI:** a Python package (`harness/`) that runs
   experiments offline from recorded byte streams, with hooks on the model.
3. **One entropy contract** shared by app and harness: the same bytes → noise
   mapping on both sides, verified by committed test vectors.
4. **App exports what the harness consumes:** raw stream recordings with
   timestamps, and seed bundles drawn bank-first with provenance.

## Key decisions (and the reasoning trail)

- **DiffusionGemma is masked/discrete, not Gaussian.** The 2026-06-22 spec
  rejected masked diffusion because it has no continuous noise layer. We
  accept that trade for a modern open model. Entropy still *is* the noise: in
  DiffusionGemma's sampler every random draw is a uniform — the random initial
  canvas, the categorical token draw per position per step, and the renoise
  draw per rejected position per step. We feed every one of those draws from
  the entropy tape via inverse-CDF (one uniform per draw), which is exact
  categorical sampling and entropy-frugal. (Rejected: Gumbel-max over the
  262k vocabulary — ~268 MB of entropy per step, infeasible from any source.)
- **Harness lives outside the UI.** Hooks, batched runs and offline analysis
  belong in PyTorch, not a Tauri webview. The app stays the live demo and the
  *recorder*; the harness is the lab.
- **Real time is not required for the experiment.** Coherence labels are a
  property of the stream, so they can be computed post hoc over a recording.
  This decouples the experiment from device throughput and makes a remote
  QRNG usable as a *control arm* (bulk-fetched, never live).
- **Entropy budget per generation** (256-token canvas, `S` steps):
  `4·256·(1 + 2·S)` bytes = 97 KiB at the default 48 steps (≈25–33 KiB when
  early stopping ends a canvas after 12–16 steps). Seeds are cut at the
  maximum budget so a run can never run dry mid-canvas.
- **Conditions:** `in_band` (device bytes while the walk was inside the 95%
  envelope), `out_band` (bytes while it was outside — what the anomaly bank
  harvests), `prng` (seeded PCG64 control), `remote` (ANU QRNG bulk fetch,
  a locality control). The harness treats all four identically: a byte file
  plus a label.
- **Transformers is the model runtime for the harness.** Day-zero support,
  hooks are trivial. The app's own engine swap (candle port or sidecar) is
  phase 2; the harness ships a sidecar mode speaking the app's existing
  newline-JSON protocol so phase 2 is wiring, not modelling.

## Architecture

```
 app (Tauri)                      harness/ (Python, offline)
 ├─ engine tick ──► .bbrec ───────► coherence.py  (walk replay, band per byte)
 │   (recorder)                    ├─ segments.py (cut seeds per condition)
 ├─ bank-first   ──► .seed.* ─────► tape (EntropyTape over a seed file)
 │   (export_seed)                 ├─ sampler.py  (DiffusionGemma loop,
 └─ Generate (phase 2: sidecar)◄── serve.py        all draws from the tape)
                                   ├─ models/      (HF adapter + hooks, stub)
                                   ├─ run.py       (conditions × seeds × prompts)
                                   └─ analyze.py   (stats, probes, report)
 shared contract: bbrec/ (Rust) ⇄ harness/babble_harness/{noise,recording}.py
                  verified by docs/contract/noise_vectors.json
```

### Unit 1 — `bbrec` crate (Rust, dependency-light, testable headless)

- Stream recording format `.bbrec`: magic `BBREC001`, u32 LE header length,
  header JSON (`source`, `started_at_ms`, `trial_interval_ms`,
  `trial_min_bits`), then frames `[u64 LE t_ns][u32 LE len][bytes]`.
  `t_ns` is monotonic time since recording start. Frames are written once per
  engine tick with the bytes that tick drained, so they align with the walk's
  trial clock.
- Seed bundle: `<name>.seed.bin` (raw bytes) + `<name>.seed.json`
  (`bytes`, `bank_fraction`, `tags[]`, `created_at_ms`, `source`).
- Noise contract: `bytes_to_uniforms` (4 bytes LE → `(u32 + 0.5) / 2^32`),
  `uniform_to_index(u, n) = min(floor(u·n), n−1)`.
- Lives outside `src-tauri` so `cargo test` covers it without GTK/WebKit.

### Unit 2 — App wiring (`src-tauri`)

- `ControlMsg::StartRecording(path)` / `StopRecording`; the engine writes one
  frame per tick with the drained bytes (nothing while paused — paused bytes
  are discarded from stats too). Snapshot DTO gains `recording` (path or
  null) and `recording_bytes`.
- `export_seed(n_bytes)` command: `GetSeed` (bank-first, destructive, as a
  generation would) → bundle in `<app-data>/exports/`. Returns the path.
- UI: a **record** toggle in the header, an **export seed** button in the
  diffusion pane.

### Unit 3 — Harness entropy side (`harness/babble_harness/`)

- `noise.py`: the contract above plus `EntropyTape` — sequential reader over
  a byte buffer with a consumption log `(purpose, offset, n_bytes)` so each
  sampler step is attributable to byte ranges.
- `coherence.py`: replay of `stats.rs`'s trial walk over a recording: a trial
  finalises when ≥ `trial_interval` has elapsed since the last trial *and*
  ≥ `trial_min_bits` have accumulated; `z = (ones − n/2)/√(n/4)`,
  `C_k = Σz`, `σ = C_k/√k`, bands at 1.96 / 2.576 / 3.291. Each frame is
  labelled with the band in effect after it is pushed — exactly when the
  app's bank would deposit it.
- `segments.py`: contiguous byte runs per band, cut into fixed-size seeds;
  PRNG control seeds; ANU bulk fetch writes a `.bbrec` so it flows through
  the same path.

### Unit 4 — Sampler and model (`sampler.py`, `models/`)

Mirror of DiffusionGemma's reference loop (DeepMind `gemma/diffusion` and
Transformers `generation_diffusion_gemma.py`), one canvas per sample:

1. canvas ← `tape.randint(V, 256)` (reference: `torch.randint` / `jax.random.randint`).
2. for `step = S … 1`: logits ← denoiser(canvas, self-cond logits);
   `T = t_min + (t_max − t_min)·step/S`; probs ← softmax(logits/T);
   per-position entropy; **entropy-bound selection**: sort ascending,
   accept while `cumsum − own ≤ bound` (all on the last step);
   sampled ← `tape.categorical(probs)` (reference: `torch.multinomial`);
   renoise ← `tape.randint(V, 256)` on rejected positions (reference:
   `torch.randint`); early stop when mean entropy ≤ threshold and the argmax
   canvas is unchanged for `stability_threshold` steps.
3. Each step records: canvas, accepted mask, entropies, argmax, temperature,
   tape byte ranges consumed, and whatever the model adapter captured.

`Denoiser` is a small protocol (`denoise(canvas, self_cond) → (logits,
capture)`); `StubDenoiser` makes the loop testable headless; the Transformers
adapter captures residual-stream hidden states at chosen layers, router
logits (expert choice histograms), and a logit lens (per-layer top-1
agreement with the final prediction).

### Unit 5 — Runner and analysis

- `run.py`: `conditions × seeds × prompts` → `runs/<name>/manifest.json`,
  `samples.jsonl` (text, per-step scalars, tape provenance) and
  `activations/<sample>.npz`.
- `analyze.py`: per condition — tokens accepted per step, steps-to-commit per
  position, entropy trajectory, early-stop step, routing entropy, logit-lens
  depth, text diversity (distinct-n, repetition). Between conditions —
  Mann–Whitney on scalars, a cross-validated linear probe on pooled
  activations with a label-permutation null, and the mean-difference
  direction. Output: `report.md` + `report.json`.

### Unit 6 — Phase 2 (app engine swap, not in this change)

`serve.py` speaks the sidecar protocol already documented in
`sidecar/README.md` (`steps`, `seq_len`, `entropy_hex`, `prompt` →
`step`/`done` events) so the Tauri backend can spawn it behind the existing
`diffusion` event stream. A candle port of the 26B MoE remains the single
largest risk and is explicitly deferred.

## Error handling

- Seed shorter than the budget: the tape raises before the first step; the
  runner skips the sample and records why. Never silently tops up — a
  condition label must mean what it says.
- Recording while paused: frames are not written (consistent with stats).
- Model unavailable: the runner still works with `--model stub` so the whole
  pipeline is exercisable without weights.

## Testing

- Contract vectors: `docs/contract/noise_vectors.json` checked by both
  `bbrec` (Rust) and `test_noise.py`.
- `.bbrec` round-trip on both sides; a Python-written fixture is read by Rust
  and vice versa through the committed fixture.
- Coherence replay reproduces `stats.rs` band classification on a biased
  stream (escapes) and a uniform stream (stays in band).
- Sampler: determinism (same tape ⇒ same canvas), divergence (different tape
  ⇒ different canvas), exact entropy accounting against the budget formula,
  entropy-bound selection against a hand-computed case, inverse-CDF
  categorical against a float64 reference.
- Analysis: probe accuracy ≈ 0.5 under a permuted null, ≈ 1.0 on separable
  synthetic activations.

## Out of scope (YAGNI)

- Multi-canvas (block-autoregressive) generation beyond 256 tokens.
- Image/video inputs to DiffusionGemma.
- Persisting recordings across app restarts automatically.
- The candle port.
