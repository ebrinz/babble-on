# babble-on — TrueRNG Entropy Observatory

A Tauri 2 desktop app that reads from a TrueRNG hardware random-number generator (or falls back to a built-in software simulator) and visualises the stream quality in real time.

This is the babble-on project's entropy observatory (Plan 1) plus its bank-seeded diffusion text generator (Plans 2–3): entropy harvested from the live stream — and, preferentially, from an anomaly bank of out-of-band bytes — seeds the initial latent for on-device text generation.

## Requirements

- **Rust** (stable, 2021 edition)  
- **Node.js** ≥ 18 with npm  
- **TrueRNG hardware** (optional — the app falls back to a software simulator when no device is detected)

## Running

```bash
npm install
npm run tauri dev
```

A native window opens within a few seconds. On first launch the source defaults to **auto-detect**, which uses the first `/dev/cu.usbmodem*` (macOS) device found, or falls back to the simulator if none is plugged in. On Linux, auto-detect also matches `/dev/ttyACM*` devices.

## Source Switch

The drop-down in the top bar lets you choose the byte source at runtime:

| Option | Behaviour |
|--------|-----------|
| `auto` | Auto-detect TrueRNG; falls back to simulator |
| `simulate` | Software CSPRNG — metrics stay green |
| `bad` | Biased RNG — metrics turn red, coherence walk escapes |

The **pause** button freezes accumulation without discarding state; **reset** clears the sliding window and starts fresh.

## What Each Panel Shows

| Panel | Description |
|-------|-------------|
| **Coherence walk** | Cumulative signed sum of bits (±1 per bit). The gold envelope is the expected 2σ band; a healthy source wanders inside it. A biased source bolts out. |
| **NIST metrics list** | Shannon entropy, min-entropy, NIST monobit, chi-square, and serial correlation — each with a PASS / FAIL verdict and colour. |
| **Byte histogram** | 256-bar frequency distribution. A flat histogram indicates uniform byte output. |
| **Bitstream ribbon** | Live 0/1 tile strip; green = 1, dark = 0. |

## Anomaly Bank

Bytes that arrive while the coherence walk is outside its 95% envelope are
harvested into a 64 KiB FIFO **anomaly bank** (gold meter in the anomaly-log
panel). Each generation's initial latent is seeded from the bank first —
destructively, so every anomaly's bytes seed exactly one text — topped up from
the live stream when the bank runs short. The stamp under the output records
the provenance, e.g. `seed: 72% anomaly bank — +3.2σ @ 14:32`.

Demo arc: switch the source to **bad rng** → the walk escapes the gold band →
the bank floods → hit **Generate** → the text is stamped with the anomaly that
birthed it. **reset** clears the bank along with the stats.

## Visual 3-state smoke (user-run)

Because this README is authored by a headless CI agent, the following end-to-end check must be performed by a human on a machine with a display:

1. Run `npm run tauri dev` with source set to **simulate** → metrics should be green within ~2 s.  
2. Switch to **bad rng** → verdicts turn red and the coherence walk escapes the gold band.  
3. Switch to **auto** with a TrueRNG plugged in → the status label shows the `/dev/cu.usbmodem*` device path; without hardware it shows `simulate`.

## Anomaly bank smoke (user-run)

Also user-run, for the same reason as above:

1. `npm run tauri dev`, source **simulate** → bank meter present, near-empty (healthy source banks at ~5% duty cycle).
2. Switch to **bad rng** → walk escapes; bank meter visibly fills gold.
3. **Generate** → stamp shows a bank percentage and σ/time tags; bank meter drops by ~16 KiB (one 256-token latent).
4. Generate again with an empty bank → stamp reads `seed: live stream`.
5. **reset** → meter returns to zero.

## Verification Status (headless CI)

| Check | Result |
|-------|--------|
| `npm run build` (vite) | ✓ pass |
| `cargo build --release` | ✓ pass (13.8 s warm, no dead-code warnings) |
| `cargo test` | ✓ 25/25 pass |
| `npm run test` (vitest) | ✓ 6/6 pass |
| Visual 3-state smoke | **user-run** (see above) |
| Anomaly bank smoke | **user-run** (see above) |

## Credits

The entropy/coherence mathematics and NIST metric implementations were ported from the [ghostty-rng](https://github.com/crashy/ghostty-rng) sibling project. Core modules: `math.rs`, `stats.rs`, `source.rs`.
