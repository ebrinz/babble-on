# Spike: Cola-DLM on Apple Silicon (2026-10-14)

Throwaway spike. Question: does ByteDance's Cola-DLM (continuous latent
diffusion LM, ~2B, Apache-2.0, [code](https://github.com/ByteDance-Seed/Cola-DLM),
weights `ByteDance-Seed/Cola-DLM`) run on MPS at a usable speed, and does its
latent noise change *content*, as Plaid's does and DiffusionGemma's doesn't?

Setup (vendor code and weights are gitignored):

    git clone --depth 1 https://github.com/ByteDance-Seed/Cola-DLM vendor   # tested at 7d1daee
    uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -e ./vendor huggingface_hub
    ./fetch.sh            # resumes each weight file until complete (needs expected_sizes.txt; 9.3 GB)
    .venv/bin/python smoke.py ../coherence/runs/2026-10-12/ordinary.bin --device mps

Entropy enters as each 16-token block's Gaussian noise (16 × 16 latent
values); `smoke.py` routes every `torch.randn` the sampler makes through a
TrueRNG byte tape (4 bytes → uniform → inverse normal CDF, as Plaid's
`trng.rs`), and decodes greedily so the noise is the only source of variation.

## Result

- **Runs on MPS** after two spike-local fixes: the inference code casts its
  inputs to bfloat16 and relies on CUDA autocast, so on MPS the weights are
  loaded in bfloat16 and the latent is cast before the VAE decoder. Running
  instead under `torch.autocast("mps", bfloat16)` with fp32 weights (the CUDA
  setup) gives identical text from the same bytes, so the workaround does not
  change the output.
- **Speed:** load 5–9 s; 35–44 s per 128 tokens on the M1 Pro (8 blocks ×
  16 flow steps, CFG 7.0). 8 KB of TrueRNG bytes per 128 tokens.
- **Pure function of the bytes:** same bytes → identical text.
- **Noise changes content:** six byte slices on "Once upon a time" gave six
  unrelated stories (a gym routine in Vienna, a DJ, a push off a cliff, an old
  man named Theodore Chambers, a quarrel among friends, Mexican citizens
  against foreign occupation); on "The", a dated news fragment, a probability
  claim, a puppet show, Babylon, Fed lending rates, an essay. Mean pairwise
  word overlap 0.10 for both prompts (DiffusionGemma with a fixed prompt:
  0.3–0.5).
- **Prose is poor:** a base model continuing web text, often ungrammatical
  ("we were walking on toy on mountains when Rose dope a carelessness"),
  closer to Plaid than to DiffusionGemma.

Samples in `runs/smoke.jsonl`.
