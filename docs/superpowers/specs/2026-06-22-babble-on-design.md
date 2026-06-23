# babble-on — Design

**Date:** 2026-06-22
**Status:** Approved (design); implementation plan to follow

## One line

A Tauri desktop app that turns a TrueRNG hardware entropy stream into both a live
*coherence observatory* and the literal *Gaussian noise layer* of a continuous
embedding-space text-diffusion model — so every generation is physically,
irreproducibly seeded by hardware randomness.

## Goals

1. `git init` the repository. *(done as part of writing this spec)*
2. Reuse the proven entropy + coherence core from the sibling `../ghostty-rng`
   project, lifted into a Tauri desktop GUI.
3. Drive a continuous text-diffusion model from the hardware entropy stream, so
   the TrueRNG bytes *are* the diffusion noise — a novel use of physical entropy
   to influence generation.

## Key decisions (and the reasoning trail)

- **Entropy mechanism: TRNG = the noise source.** Hardware bytes become the
  diffusion noise rather than a PRNG seed. Identical prompt → genuinely unique
  output every run.
- **Modality: text** ("babble-on").
- **Runtime: all-Rust via `candle`, in-process. No Python sidecar, no localhost
  server.** Chosen over a Python+`diffusers` sidecar. Cost: we port the model to
  candle ourselves and lose the HF ecosystem's velocity. Benefit: a single
  binary, and the entropy injection becomes direct — the TRNG bytes already live
  in Rust and feed candle's sampler with zero IPC.
- **Model family: continuous embedding diffusion** (not masked/discrete). This is
  the only text-diffusion family with a *literal Gaussian noise layer* (real
  `N(0,σ²)` noise added to token-embedding vectors), which is the truest
  expression of "entropy = noise." Masked diffusion (LLaDA/MDLM) has no
  continuous noise — its randomness is discrete masking + categorical sampling —
  so it was rejected for this concept.
- **Model checkpoint: primary `RePlaid` (contingent on weight availability),
  confirmed fallback `Plaid-1B`, pipeline-proof `Diffusion-LM`.**
  - `RePlaid` (Yang et al., NVIDIA & Cornell, arXiv:2605.18530, May 2026) is a
    revised, modernized Plaid: same VDM continuous-diffusion family with a literal
    Gaussian noise layer on embeddings (`q(z_t|x) = N(α_t·e, σ_t²·I)`, `d_e=16`),
    but SOTA among continuous DLMs (22.1 PPL on OpenWebText, 20× AR compute gap
    vs. Plaid's 64×) with better generation quality. It is architecturally
    Plaid-shaped, so it drops into the same candle engine. **Risk: public weights
    are unconfirmed as of this writing** — the Phase-0 spike's first task is to
    locate them.
  - `Plaid-1B` is the confirmed fallback — the only *other* open-domain-English
    continuous diffusion LM with public weights ([igul222/plaid](https://github.com/igul222/plaid),
    ≈GPT-2 124M likelihood). If RePlaid weights are unavailable, we ship on
    Plaid-1B and swap RePlaid in later via the same engine.
  - `Diffusion-LM` (80M, BERT-base) is narrow-domain but the easiest to port and
    serves as the Phase-0 proof that the full pipeline works end-to-end.
- **Compute: not the constraint.** All candidates are ≤1B params and run
  trivially on the target Apple Silicon / 32 GB / Metal machine. The real axis is
  port-effort vs. text-quality, not Mac performance.
- **TrueRNG: hardware present.** Auto-detect `/dev/cu.usbmodem*`, fall back to the
  software simulator if unplugged.

## Honest risks

1. **Model-port risk is the primary risk.** Continuous text-diffusion checkpoints
   are research artifacts. Porting one to candle — embedding round-trip,
   timestep-conditioned denoiser, Plaid's self-conditioning + VDM noise schedule,
   the rounding/"clamping" trick — is the hard unknown. **Mitigation: a Phase-0
   spike** gets *one* checkpoint emitting *any* text through candle with an
   ordinary PRNG **before** any UI polish or TRNG wiring. Fallback ladder:
   RePlaid → Plaid-1B → Diffusion-LM.
4. **RePlaid weight availability is unconfirmed.** It is the preferred model but a
   6-week-old paper with no confirmed public checkpoint. The Phase-0 spike's first
   task is to locate the weights; if absent, Plaid-1B (confirmed public) is used
   and RePlaid is swapped in later through the same engine.
2. **Weight format.** None of these ship as Hugging Face safetensors — they are
   torch `.pt`/`.bin` on GitHub / Google Drive. A **one-time `torch → safetensors`
   conversion script** (Python, run once) precedes candle loading. No Python at
   runtime.
3. **"Prompt" semantics differ from a chat LLM.** Continuous diffusion models are
   not instruction-followers. The prompt acts as a **conditioning prefix / seed
   the babbler continues or in-fills**, not "answer my question." On-theme for
   "babble-on," but the UI copy must set this expectation.

## Architecture

Single Rust binary (Tauri) + webview frontend. Three internal units, each
independently testable.

```
┌─ Tauri app (single Rust binary + webview) ───────────────┐
│  Rust backend                         Web frontend        │
│  ├─ entropy::source ◄─ TrueRNG        ├─ Coherence chart  │
│  ├─ entropy::stats + coherence        ├─ Entropy/NIST     │
│  ├─ TrngRng  (bytes → N(0,1))         ├─ Bitstream ribbon │
│  └─ diffusion (candle, Metal/MPS)     └─ Prompt + "text   │
│       embed→denoise→round→tokens          crystallizing"  │
│       all Gaussian draws from TrngRng     denoise view    │
└───────────────────────────────────────────────────────────┘
```

### Unit 1 — `entropy` core (ported from ghostty-rng)

- **What it does:** owns the TrueRNG device, ingests bytes on a background
  thread, computes Shannon/NIST statistics and the coherence random-walk.
- **How it's used:** Tauri commands start/stop/select the source; the backend
  emits metric snapshots to the frontend as Tauri events.
- **Depends on:** `serialport` (device), the ported `source.rs` / `stats.rs` /
  coherence modules. The ratatui UI is *stripped out* — copy and trim rather than
  depend on the sibling binary crate.
- Source kinds: `Serial` (auto-detected), `Simulate` (xoshiro256**), `BadRng`
  (biased LCG) — exposed as a UI source switch. Auto-reconnect on unplug.

### Unit 2 — `TrngRng` (the novel bridge)

- **What it does:** converts live TrueRNG bytes → uniform → `N(0,1)` Gaussian
  samples on demand (inverse-CDF / Box–Muller).
- **How it's used:** implements the RNG interface the diffusion sampler pulls
  from. It supplies **every** Gaussian draw the sampler needs — the initial
  latent *and* the fresh per-step noise injected at each reverse step (continuous
  diffusion is stochastic at every step). This maximally showcases the device and
  makes the hardware literally the canvas.
- **Depends on:** the `entropy::source` byte stream. Buffers bytes; when a
  generation needs more Gaussians than buffered, it pulls more from the stream.

### Unit 3 — `diffusion` engine (candle)

- **What it does:** runs the continuous text-diffusion reverse process to turn a
  noise tensor + prompt prefix into tokens.
- **Components:**
  - token embedding ↔ vector (load embeddings from converted weights),
  - cosine / model-specific noise schedule,
  - timestep-conditioned transformer denoiser (port the forward pass; load
    weights from safetensors),
  - DDPM/DDIM-style reverse sampler that draws all Gaussian noise from `TrngRng`,
  - nearest-embedding rounding ("clamping trick") → tokens.
- **How it's used:** a Tauri command kicks off generation with a prompt; the
  engine streams intermediate decode states (text crystallizing across diffusion
  steps) back as events.
- **Depends on:** `candle` (+ Metal backend), `tokenizers`, `TrngRng`, the
  converted model weights.

### Unit 4 — Web frontend

- **What it does:** the ghostty-rng "observatory" reborn as GUI, plus the
  generation panel.
- **Panels:** coherence walk vs. parabolic significance envelopes, entropy /
  NIST-style test readouts, the truecolor bitstream ribbon, a prompt box, and a
  **generation view where text visibly crystallizes out of noise across diffusion
  steps** (mirroring the denoise).
- **Style:** reuse the ghostty-rng palette — turquoise `#6cf0d0` + gold
  `#f2c14e` — for visual continuity.
- **Depends on:** Tauri commands/events from the backend.

## Data flow

1. Background thread streams TrueRNG bytes → `entropy::source`.
2. `entropy::stats` + coherence recompute on a fixed cadence; snapshots emitted
   to the frontend as events (live charts).
3. User submits a prompt in the webview → Tauri command → `diffusion` engine.
4. The sampler pulls all its Gaussian noise from `TrngRng` (fed by the same live
   byte stream), denoises, rounds to tokens, and streams intermediate states back
   as events.
5. Frontend renders both the observatory (always live) and the crystallizing
   generation (during a run).

## Error handling

- **Device:** auto-detect; fall back to `Simulate` if no device; auto-reconnect
  on unplug (ported logic). Source state surfaced in the UI.
- **Model load / OOM:** surfaced as a UI banner with a "model loading…" state, not
  a hang.
- **Weight/asset missing:** clear actionable error pointing at the conversion
  step.

## Testing

- Port ghostty-rng's existing tests: entropy math (uniform → 8.0, constant → 0,
  biased → fail, window eviction), coherence/anomaly engine.
- **`TrngRng` distribution test:** sampled output is ≈`N(0,1)` (mean/variance /
  goodness-of-fit within tolerance).
- **Determinism test:** feeding the *same* byte buffer twice yields *identical*
  generated text; different buffers diverge — proves the TRNG genuinely drives
  sampling.
- **Diffusion smoke test:** the sampler/rounding harness runs against a tiny stub
  denoiser, so the pipeline is testable without the real weights.
- **Frontend:** headless render/command-event contract checks where practical.

## Build phases (high level — detailed plan to follow)

- **Phase 0 — De-risk spike:** (a) locate RePlaid weights; (b) one
  continuous-diffusion checkpoint emits text through candle with an ordinary PRNG.
  Decide RePlaid vs. Plaid-1B vs. Diffusion-LM. Includes the one-time
  `torch → safetensors` conversion.
- **Phase 1 — Entropy core port:** `source` / `stats` / coherence into the Tauri
  Rust backend, with tests.
- **Phase 2 — Tauri shell + observatory frontend:** commands/events, live charts,
  palette.
- **Phase 3 — `TrngRng` + entropy injection:** wire hardware noise into the
  sampler; determinism + distribution tests.
- **Phase 4 — Generation UI:** prompt box + crystallizing-text denoise view.

## Out of scope (YAGNI)

- Image / multimodal generation (text-only for now).
- Model training or fine-tuning (pretrained weights only).
- Masked/discrete diffusion models.
- Any hosted-API or Python-at-runtime path.
- Coherence-modulated sampling knobs (temperature/guidance steering) — the chosen
  mechanism is TRNG-as-noise only; this could be a future extension.
