# DiffusionGemma spike

Can `google/diffusiongemma-26B-A4B-it` replace Plaid while keeping the
"entropy authors the text" property? This spike tests whether it runs on our
Mac and whether randomness we supply steers its output.

## Setup

```bash
cd spikes/diffusiongemma
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -U mlx-vlm   # 0.7.6 at time of test
hf download mlx-community/diffusiongemma-26B-A4B-it-4bit   # ~15 GB
.venv/bin/python smoke.py [--prompt "..."] [--temperature 0.7]
```

## Results (2026-10-08, M1 Pro 32 GB, temperature 0, 48 steps)

- Load 7.8 s, roughly 15–21 s per 256-token canvas, peak memory 17.5 GB.
- Same canvas and same seed reproduce the output exactly. A new canvas gives
  0.36 word similarity; a new per-step seed alone gives 0.48.
- Both random sources matter. The initial canvas can be supplied through
  `decoder_input_ids`, but every step also re-noises unsettled positions with
  `mx.random`. For the output to depend only on TrueRNG, drive the seed (or
  patch the re-noise) from TrueRNG too.
- The prompt sets the subject: all four runs described a sunset over a valley.
  Entropy changes the wording, not the topic.
