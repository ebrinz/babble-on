# babble-on — TrueRNG Entropy Observatory

A Tauri 2 desktop app that reads from a TrueRNG hardware random-number generator (or falls back to a built-in software simulator) and visualises the stream quality in real time.

This is **Plan 1** of the babble-on project: the entropy observatory foundation. Diffusion-driven text generation (Plans 2–3) is not yet implemented.

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

## Visual 3-state smoke (user-run)

Because this README is authored by a headless CI agent, the following end-to-end check must be performed by a human on a machine with a display:

1. Run `npm run tauri dev` with source set to **simulate** → metrics should be green within ~2 s.  
2. Switch to **bad rng** → verdicts turn red and the coherence walk escapes the gold band.  
3. Switch to **auto** with a TrueRNG plugged in → the status label shows the `/dev/cu.usbmodem*` device path; without hardware it shows `simulate`.

## Verification Status (headless CI)

| Check | Result |
|-------|--------|
| `npm run build` (vite) | ✓ pass |
| `cargo build --release` | ✓ pass (48 s, 7 dead-code warnings — ported math helpers reserved for Plans 2–3) |
| `cargo test` | ✓ 15/15 pass |
| `npm run test` (vitest) | ✓ 2/2 pass |
| Visual 3-state smoke | **user-run** (see above) |

## Credits

The entropy/coherence mathematics and NIST metric implementations were ported from the [ghostty-rng](https://github.com/crashy/ghostty-rng) sibling project. Core modules: `math.rs`, `stats.rs`, `source.rs`.
