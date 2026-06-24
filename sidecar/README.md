# babble-on diffusion sidecar

A persistent Python service that runs **Plaid-1B** (a continuous, embedding-space
diffusion language model) on Apple Silicon (MPS) and streams text as it
crystallizes out of noise over the denoising steps. The Tauri app spawns this
process and feeds it **TrueRNG entropy** to seed the initial latent — so the
generated passage is a direct function of physical randomness.

## Attribution

The model and the code under `lib/` are derived from **Plaid**
([igul222/plaid](https://github.com/igul222/plaid), *Likelihood-Based Diffusion
Language Models*, Gulrajani & Hashimoto). `compat.py` adds pure-torch
replacements for the CUDA-only kernels (Apex `FusedRMSNorm`, FlashAttention
`FusedMLP`/attention/rotary) so the unmodified model runs on MPS/CPU. The four
`lib/models.py` patches (device literals, fp64 constants, bf16 autocast) are
noted inline.

## Setup

```bash
cd sidecar
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Download the Plaid-1B weights (~4.7 GB) and place them at `../models/plaid1b/`
(four files: `model.pt`, `embedding_matrix.pt`, `noise_schedule.pt`,
`gamma_bounds.pt`):

```bash
# from the babble-on repo root
mkdir -p models/plaid1b && cd models/plaid1b
gh release download v1.0.0 --repo igul222/plaid --pattern 'plaid1b_weights*'
cat plaid1b_weights.tar.gz.* | tar xzf - && mv plaid1b_weights/* . && rmdir plaid1b_weights
```

## Protocol

Newline-delimited JSON. **stdout carries only protocol JSON; all logs go to
stderr.** On startup, after the model is resident, the service emits
`{"type":"ready"}`, then reads one request per line from stdin:

```json
{"steps":256,"seq_len":256,"n_samples":1,"seed":0,"preview_every":24,
 "entropy_hex":"<optional hex bytes to seed the initial latent>"}
```

and streams back:

```json
{"type":"step","i":24,"total":256,"text":["partial ..."]}
{"type":"done","i":256,"total":256,"text":["final ..."],"elapsed":12.3}
```

If `entropy_hex` is present, those bytes (4 per Gaussian, `uint32 → uniform →
ndtri`) become the literal initial-noise tensor; otherwise a seeded PRNG is used.

## Try it standalone

```bash
echo '{"steps":48,"seq_len":96,"n_samples":1,"preview_every":12}' | \
  .venv/bin/python sidecar.py
```

## Performance

Compute-bound on MPS: ~10 ms/step/sample at seq-64 up to ~600 ms/step at
seq-256. A high-quality 384-step / 256-token generation takes a few minutes; a
48-step preview is a few seconds. Quality scales with `steps` (Plaid's reference
uses thousands; coherent babble emerges by ~256–384).
