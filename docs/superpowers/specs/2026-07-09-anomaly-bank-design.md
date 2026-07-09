# babble-on — Anomaly Bank + Plaid Quality Pass — Design

**Date:** 2026-07-09
**Status:** Approved (design); implementation plan to follow
**Builds on:** `2026-06-22-babble-on-design.md` (all phases of which are implemented)

## One line

Bytes captured while the coherence walk is outside its 95% significance envelope
are banked as a scarce resource; each generation's initial noise latent is seeded
from that bank (destructively), so every text carries the provenance of the
physical anomaly that birthed it — plus a sampling-quality pass on Plaid-1B.

## Goals

1. **Anomaly Bank:** harvest out-of-band entropy into a visible, spendable
   reservoir; seed the initial diffusion latent from it; stamp each generation
   with its anomaly provenance.
2. **Quality pass:** improve Plaid-1B text quality through sampling fidelity and
   tuning (RePlaid weights confirmed unavailable as of 2026-07-07 — no code link
   on arXiv:2605.18530, nothing on GitHub/HF).
3. Keep the bank/provenance design **modality-agnostic** so a future image pane
   (candle already ships Stable Diffusion/FLUX) reuses it unchanged.

## Key decisions (and the reasoning trail)

- **What gets banked: raw bytes, not Gaussians.** Conversion stays in one place
  (`bytes_to_gaussians`); the bank is a byte pool with provenance tags.
- **Bank seeds the initial latent only.** Per-step noise stays on the live
  stream. A small bank goes far; generation is never blocked. (Rejected: all
  noise from the bank — conceptually purer but blocks generation for long
  stretches on a healthy source. Rejected: a selectable noise source — weaker
  statement.)
- **Spending is destructive** (FIFO). Each anomaly's bytes seed exactly one
  text, then they're gone. When the bank can't cover a full latent, the seed
  tops up from the live stream and the UI reports the split. (Rejected:
  non-destructive reservoir — always available, but "spending an anomaly" loses
  its meaning. Rejected: block-when-empty — a fresh session couldn't generate.)
- **Trigger condition:** `Band::of(sigma) != Band::Inside`, i.e. the walk is
  beyond the 95% envelope — the same classification the existing anomaly log
  uses (`stats.rs`). On a healthy source this has roughly a 5% duty cycle, so
  the bank fills slowly; on the `bad` source it floods. Both are features.
- **Model: stay on Plaid-1B.** RePlaid remains the preferred model; the engine
  stays swap-ready for when weights publish. RDLM (harryjo97/RDLM) was
  considered and rejected: hypersphere/manifold diffusion, not the
  literal-Gaussian-noise family the concept requires, and Text8/LM1B training
  data is weaker than OpenWebText.

## Architecture

Two new concerns, both on the entropy side of the existing `NoiseFn` boundary.
No changes to the sampler's signature semantics: `generate()`'s first `noise()`
call is the initial latent — that is the seam the bank plugs into.

### Unit 1 — `AnomalyBank` (new, backend)

- **What it does:** a bounded FIFO byte pool (default cap 64 KB, drop-oldest)
  with provenance. While the stats engine classifies the walk out-of-band, the
  ingest thread copies incoming source bytes into the bank, tagged with the
  active `AnomalyEvent` (start time, peak σ, band).
- **Interface (modality-agnostic):**
  - `deposit(bytes, event_ref)` — called from the ingest path only when
    out-of-band.
  - `withdraw(n) -> (Vec<u8>, Vec<ProvenanceTag>)` — destructive FIFO read of
    up to `n` bytes plus the tags of every anomaly event represented in what
    was withdrawn. Returns fewer than `n` when the bank runs short.
  - `fill() -> (usize, usize)` — current/capacity, for the UI meter.
- **Depends on:** the existing `entropy::stats` band classification and the
  source byte stream. Session-scoped; not persisted across restarts.
- **Concurrency:** ingest thread deposits; generation withdraws. A mutex is
  sufficient at these rates.

### Unit 2 — Seed provenance (engine wiring, `src-tauri/src/diffusion.rs`)

- **What it does:** when a generation starts, the noise closure handed to
  `generate()` serves its **first** call (the initial latent) by withdrawing
  from the bank and topping up any shortfall from the live stream, converting
  via the existing `bytes_to_gaussians`. All subsequent calls (per-step noise)
  draw from the live stream exactly as today.
- **Emits:** a `SeedProvenance` value on the generation-started/finished event:
  `bank_fraction` (0.0–1.0) and the list of consumed anomaly tags
  (timestamp, peak σ, band). Empty bank ⇒ `bank_fraction = 0`, today's
  behavior, no error.
- **Determinism property preserved:** same bank contents + same live byte
  buffer ⇒ identical text.

### Unit 3 — Frontend additions

- **Bank meter:** a gold fill bar (existing `#f2c14e` accent) adjacent to the
  coherence panel, showing fill level; visibly filling while the walk is
  out-of-band so the harvest moment reads live.
- **Provenance stamp:** under each generated text, e.g.
  `seed: 72% anomaly bank — 3.2σ @ 14:32, 2.8σ @ 15:01` (live-stream-only
  seeds say `seed: live stream`).
- Demo arc: switch source to `bad` → walk escapes → bank floods → generate →
  text stamped with the anomaly that birthed it.

### Unit 4 — Plaid-1B quality pass (`diffusion-rs`)

Sampling-side improvements only; no training, no new weights:

- Verify self-conditioning fidelity against the reference Plaid implementation
  (the highest-suspicion divergence point for quality).
- Tune default step count / noise schedule endpoints (`gamma_0`/`gamma_1`)
  for quality-vs-latency; expose better defaults in the UI controls.
- Improve the rounding/clamping stage if inspection shows a gap vs. reference.

Honest framing: incremental gains with a GPT-2-small ceiling. The plan phase
inspects `diffusion-rs/src/sampler.rs` against the reference to pick the
highest-value fixes; anything not clearly divergent from reference is left
alone.

## Data flow (delta over existing)

1. Ingest thread: bytes → stats (as today); when band ≠ Inside, the same bytes
   also → `AnomalyBank.deposit` with the active event tag.
2. Generate command: noise closure first call → `AnomalyBank.withdraw` →
   top-up from live stream → `bytes_to_gaussians` → initial latent; later
   calls → live stream (unchanged).
3. `SeedProvenance` emitted with the generation events; frontend renders the
   stamp and refreshes the bank meter.

## Error handling

- Empty/short bank: silent top-up from live stream; provenance reports the
  actual fraction. Never blocks, never errors.
- Bank overflow: drop-oldest; no user-facing error (meter simply stays full).
- Source switch/reset mid-session: bank survives source switches (bytes are
  bytes); the existing "reset" control also clears the bank.

## Testing

- **Bank unit tests:** deposits occur only when out-of-band (biased source
  fills fast; healthy source fills at ≈5% duty cycle); FIFO destructive
  withdraw; drop-oldest at capacity; provenance tags match the events whose
  bytes were withdrawn.
- **Seed wiring:** with a stocked bank, the initial latent consumes bank bytes
  first and `bank_fraction` is correct for full, partial, and empty banks.
- **Determinism:** same bank + same live buffer ⇒ identical text (extends the
  existing determinism test).
- **Quality pass:** self-conditioning outputs compared against reference
  activations where feasible; otherwise before/after sample sheets at fixed
  byte buffers.
- **Frontend:** render/contract tests for the meter and provenance stamp
  (extend `render.test.ts`).

## Future work (named, not specced)

- **Image pane:** candle-transformers' Stable Diffusion/FLUX seeded through the
  same `AnomalyBank.withdraw` → `bytes_to_gaussians` path — anomaly-born
  images with the same provenance stamp. The bank interface above is designed
  so this requires no bank/provenance changes.
- RePlaid swap-in when weights publish.
- Bank persistence across restarts.

## Out of scope (YAGNI)

- Image/audio/video generation in this iteration.
- RDLM or any manifold/discrete diffusion family.
- Configurable bank capacity in the UI (constant is fine).
- Persisting the bank to disk.
