# babble-on experiment harness

Offline lab for the question the app poses: **does text diffused from
*out-of-coherence* entropy differ, mechanistically, from text diffused from
*in-coherence* entropy?** It drives Google's
[DiffusionGemma](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
with every random draw taken from a recorded byte stream, labels those bytes
with the observatory's coherence walk, and captures what the model does
inside while it crystallises a canvas.

Design: [`docs/superpowers/specs/2026-10-02-diffusiongemma-harness-design.md`](../docs/superpowers/specs/2026-10-02-diffusiongemma-harness-design.md).

## How entropy becomes text

DiffusionGemma's sampler (DeepMind `gemma/diffusion`, Transformers
`generation_diffusion_gemma.py`) makes exactly three kinds of random draw: a
uniformly random initial canvas, one categorical token draw per position per
step, and one uniform "renoise" token per rejected position per step.
`sampler.py` reproduces that loop and takes each draw from an `EntropyTape`
by inverse-CDF — one uniform (4 bytes) per draw — so a 256-token canvas at
48 steps consumes at most `4·256·(1 + 2·48)` = 97 KiB and is a pure
function of the bytes. The bytes → uniform mapping is the same one the app
uses (`bbrec/src/noise.rs`), pinned by `docs/contract/noise_vectors.json`.

## Conditions

| condition | bytes | why |
|---|---|---|
| `in_band`  | device stream while the walk was inside the 95 % envelope | baseline physical entropy |
| `out_band` | device stream while it was outside — what the app's anomaly bank keeps | the treatment |
| `prng`     | PCG64 | no-physics control |
| `remote`   | ANU QRNG, bulk-fetched | locality control: a remote quantum source should not carry a local effect |

## Setup

```bash
cd harness
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"          # numpy, torch, pytest
uv pip install --python .venv/bin/python -e ".[model]"        # transformers ≥ 5.8 for the real model
.venv/bin/python -m pytest -q                                 # 34 tests, no weights needed
```

The model is ~52 GB in bf16 (18 GB at 4-bit through llama.cpp/Unsloth; this
harness uses Transformers, so plan on a GPU box with ≥ 60 GB or
`device_map="auto"` offload). `--model stub` runs the whole pipeline on a
toy denoiser for plumbing checks.

## Workflow

1. **Record a stream.** In the app press **● record** (writes
   `<app-data>/recordings/stream-<ts>.bbrec`), or headless:
   `python -m babble_harness.cli record-serial /dev/cu.usbmodem212201 out.bbrec --seconds 3600`.
   A healthy source is out-of-band ~5 % of the time, so each out-band seed
   costs ~2 MiB of stream; a TrueRNG V3 (50 KB/s) yields ~1 out-band seed
   per 40 s, a TrueRNGpro V2 nine times that.
2. **Label it and see what you have.**
   `python -m babble_harness.cli label out.bbrec` → duty cycle per band and
   how many full-budget seeds each condition can cut.
3. **(optional) Remote control arm.**
   `ANU_API_KEY=… python -m babble_harness.cli fetch-anu anu.bbrec --bytes 20000000`
   (paid tier; the free tier is 100 requests/month ≈ 1 MiB).
4. **(optional) App-exported seeds.** **export seed** in the app spends the
   anomaly bank into `<app-data>/exports/seed-<ts>.seed.{bin,json}`; pass the
   directory with `--seed-dir`.
5. **Run.**
   ```bash
   python -m babble_harness.cli run runs/first --model diffusion_gemma \
     --recording out.bbrec --remote anu.bbrec --prng 20 --max-per-group 20 \
     --prompt "Write a short paragraph about the sea." --prompt ""
   ```
   Writes `samples.jsonl`, `activations/*.npz`, `manifest.json`, and a
   `report.md` (`analyze` re-runs the statistics on an existing run).
6. **Read `report.md`.** Per-condition medians, Mann–Whitney tests between
   the chosen pair, and a cross-validated linear probe on pooled residual
   stream activations against a label-permutation null. Treat a hit as real
   only if it survives against both `prng` and `remote`.

## What is captured per step

- sampler: canvas in/out, accepted mask, per-position entropy, argmax,
  temperature, the tape byte range the step consumed;
- model (Transformers adapter): residual stream at chosen decoder layers,
  expert-routing histograms and their entropy per MoE layer, and a logit lens
  (per-layer top-1 agreement with the final prediction).

`steps_to_commit` per position is the direct analogue of the app's
crystallisation heat map.

## Sidecar mode (phase 2 of the app engine swap)

`python -m babble_harness.cli serve` speaks the newline-JSON protocol in
[`../sidecar/README.md`](../sidecar/README.md) (`steps`, `seq_len`,
`entropy_hex`, `prompt` → `step`/`done` events), so the Tauri backend can
spawn it in place of the Plaid engine with the entropy it already draws.

## Caveats

- `models/diffusion_gemma.py` is written against the Transformers 5.8 source
  and has not yet been run on hardware; the two likely adjustment points are
  marked `# ADAPT`. Everything else in the package is covered by tests.
- One canvas per sample (256 tokens). Longer, block-autoregressive generation
  is out of scope.
