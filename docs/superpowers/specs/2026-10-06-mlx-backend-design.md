# babble-on — MLX backend for DiffusionGemma (local 4-bit bench) — Design

**Date:** 2026-10-06
**Status:** Approved (design); not yet planned or implemented
**Builds on:** `2026-10-02-diffusiongemma-harness-design.md`

## One line

Add an Apple-Silicon backend to the harness so the 4-bit MLX build of
DiffusionGemma (`mlx-community/diffusiongemma-26B-A4B-it-4bit`) can serve as
the local experiment bench, behind the same `Denoiser` protocol, with the same
captures, driven by the same entropy tape, and pinned to the Transformers
adapter by a parity test.

## Why

The bf16 checkpoint is 51.7 GB and does not fit a 32 GiB Mac; the NVFP4 and
bitsandbytes paths need CUDA. GGUF via llama.cpp was ruled out: it owns its own
sampler and RNG, which would take the random draws away from the `EntropyTape`
and lose the per-step hidden-state and router captures.

A spike on 2026-10-06 (see `docs/experiments/LOG.md`) ran the 4-bit MLX build
behind the `Denoiser` protocol through the unchanged `sample_canvas` loop:
17.6 GB peak, ~0.5 s per forward, bit-identical forwards on repeat, identical
trajectories from the same tape, different canvases from different tapes, all
captures present, coherent text. A second probe rebuilt the harness's tiny
random-weight model in mlx-vlm and matched Transformers' logits to 1.2e-6
with MLX on CPU; the GPU differs at ~3e-3 from fp32 arithmetic order, not from
the model code.

## Goals

1. `--model mlx` for `run` and `serve`; the app reaches it through
   `BABBLE_SIDECAR_MODEL=mlx` with no Rust changes.
2. Same `Denoiser` interface and the same capture dict (shapes, conventions)
   as `models/diffusion_gemma.py`, so `analyze` and the reports are unchanged.
3. The sampler, the bytes → uniform → index contract, and the `.bbrec` / seed
   formats are untouched. Every random draw still comes from the tape.
4. A parity test holds the MLX adapter to the Transformers adapter on the tiny
   model: same logits, same captures, same trajectory from the same tape.
5. Every run records which build produced it, so 4-bit and bf16 results cannot
   be mixed unnoticed.

## Non-goals

- Fidelity of the 4-bit build to bf16. That needs the CUDA box (or the 8-bit
  MLX build as an intermediate) and is a separate experiment, not a code
  change. Results from the bench describe the quantised model.
- An app engine drop-down entry for MLX. The env var is enough for now.
- Paid CI. The MLX tests run locally only.

## Key decisions

- **MLX via mlx-vlm, pinned.** mlx-vlm already ships the DiffusionGemma model,
  the loader for the mixed 4/8-bit checkpoint layout (experts 4-bit;
  attention, dense MLP, router, embeddings 8-bit) and the tokenizer. We wrap it
  the way `diffusion_gemma.py` wraps Transformers. Rejected: a pure-MLX port
  (~1,000 lines of MoE / sliding-window / self-conditioning model code to own
  and keep correct). Fallback if the pin ever hurts: copy mlx-vlm's
  `diffusion_gemma/language.py` into the repo.
- **The adapter touches mlx-vlm's private internals in one place.**
  `_embed_canvas`, `_make_decoder_masks` and `_cache_offset` are called from a
  single `_forward` method. If mlx-vlm drifts, that is the one place to fix.
- **Parity is measured with MLX on CPU.** The GPU reorders fp32 arithmetic and
  disagrees with Transformers at ~3e-3 on the tiny model, enough to flip an
  argmax with a 2.7e-4 margin. On CPU the two agree at fp32 tolerance. The
  parity test therefore runs on CPU; a separate loose-tolerance test documents
  the GPU gap. The bench itself runs on the GPU; its own determinism (same
  input, bit-identical output) is tested on the GPU.
- **Self-conditioning logits are cast to the embedding dtype**, as the
  reference does (`generation_diffusion_gemma.py`). The spike cast them to
  bf16 unconditionally, which left a 3.5e-4 gap on the tiny fp32 model; the
  adapter follows the reference.
- **The captured last layer is post-norm**, matching Transformers'
  `hidden_states[-1]`, so layer-30 captures compare across backends.
- **Config translation fills in what Transformers serialises as `None`.**
  `global_head_dim` and `num_global_key_value_heads` come out as `None` from
  `config.to_dict()`; mlx-vlm then falls back to its 26B defaults (512 / 2),
  which silently builds the wrong model. The translation fills them from
  `head_dim` / `num_key_value_heads` and fails loudly on any other `None` it
  must resolve.
- **Quantisation is chosen by `--model-id`, not `--quant`.** The MLX
  checkpoints are pre-quantised builds (4-bit, 5, 6, 8, mxfp4 …). `--quant`
  with `--model mlx` is refused, not ignored.

## Components

### New: `harness/babble_harness/models/mlx_gemma.py` — `MLXGemmaDenoiser`

Same surface as `DiffusionGemmaDenoiser`: `from_pretrained(model_id,
capture_layers, capture_router, logit_lens, system)`, `set_prompt`,
`set_prompt_ids`, `denoise(canvas, self_cond) → (logits float32 [L, V],
capture)`, `decode`, `decode_each`, `trim_after_eos`; attributes
`canvas_length`, `vocab_size`, `softcap`, `top_k`, `prompt_ids`.

- **Prefill:** `model.diffusion_prefill_cache(ids)` once per prompt. The
  decoder reads the cache and never writes it; one prefill serves every step
  and every sample for the same prompt.
- **Per step (`_forward`):** `_embed_canvas(ids, self_cond)` →
  `_make_decoder_masks` → each decoder layer in turn, keeping the residual
  stream at `capture_layers` → `norm` → `softcap(embed_tokens.as_linear(h))`
  as float32. One `mx.eval` over logits, captured hidden states and router
  indices.
- **Router capture:** a context manager swaps `Router.__call__` for a
  recording wrapper around our forward only and restores it after. Nothing is
  patched at import time; mlx-vlm's own `generate` is left alone.
- **Captures:** `hidden [n_captured, L, hidden]` (last layer post-norm),
  `hidden_layers`, `logit_lens_agree [n_captured]` (argmax of
  `softcap(head(norm(h_l)))` vs final argmax), `router_counts [n_moe_layers,
  n_experts]`, `router_entropy [n_moe_layers]`. Same dtypes as the
  Transformers adapter (float32 / int32, numpy).
- **`describe()`** → `{"backend": "mlx", "model_id", "mlx": <version>,
  "mlx_vlm": <version>, "quantization": {"default": {bits, group_size, mode},
  "overrides": <count of 8-bit modules>}}`.
- **Version check:** warn (not fail) at import if `mlx_vlm.__version__` differs
  from the pin; the parity test is the real gate after an upgrade.

### New: `mlx_tiny` in `models/__init__.py`

The tiny random-weight model (`models/tiny.py`) rebuilt in mlx-vlm with the
same weights: translate the Transformers config (filling the `None` fields),
construct `mlx_vlm.models.diffusion_gemma.Model`, `sanitize` and
`load_weights` from `tiny_model(seed).state_dict()`. Prompt handling as for
`tiny` (characters → ids, no tokenizer). Used by the tests and by dry runs.

### Changed

- `models/__init__.py`: `load_denoiser("mlx", …)` and `"mlx_tiny"`. Missing
  `mlx_vlm` → `ImportError` with the install hint (`pip install -e .[mlx]`,
  Apple Silicon only).
- `cli.py`: `--model mlx` accepts `--model-id` (default the 4-bit build) and
  `--layers`; `--quant` is refused for `mlx`. Help strings list the new names.
- `runner.py`: if the denoiser has `describe()`, its result goes into
  `manifest.json` under `"backend"`; `analyze_run` prints it in the report
  header and the log entry.
- `pyproject.toml`: optional extra `mlx = ["mlx-vlm==0.7.6; sys_platform ==
  'darwin' and platform_machine == 'arm64'"]`. Torch stays a base dependency
  so the Transformers side of the parity test runs in the same venv.
- `src-tauri/src/sidecar.rs`: no code change; the doc comment lists `mlx`
  among the model names.

## Data flow

Unchanged from the 2026-10-02 spec. `sample_canvas` draws the initial canvas,
the categorical sample and the renoise from the tape; the denoiser only maps
`(canvas, self_cond) → logits`. The backend is invisible to the tape.

## Error handling

- Non-Apple machine or missing extra: `ImportError` from `load_denoiser` with
  the install hint; the CLI already reports loader exceptions and exits non-zero.
- `--model mlx --quant …`: refused with a message naming `--model-id`.
- Config translation: fails loudly on an unresolved `None` rather than
  letting mlx-vlm's defaults in.
- mlx-vlm version drift: warning at import; parity test is the gate.
- Sidecar: unchanged. `serve` already turns a load failure into an `error`
  message and the app respawns a dead sidecar.

## Testing

New `harness/tests/test_mlx_adapter.py`, skipped wherever `mlx_vlm` cannot be
imported (Linux CI, Intel Macs). Mirrors `test_hf_adapter.py`.

**Parity, tiny model, MLX on CPU vs Transformers:**
- logits within fp32 tolerance (`atol` ≈ 1e-5), with and without
  self-conditioning;
- `hidden` within ≈ 1e-5 per layer (last layer post-norm);
- `router_counts` and `logit_lens_agree` exactly equal;
- end to end: the same PRNG tape through `sample_canvas` on both backends
  gives identical canvases at every step.

**MLX adapter alone, on the GPU (the device the bench uses):**
- repeated forward bit-identical, with and without self-conditioning;
- cache offset unchanged after `denoise`;
- self-conditioning changes the output;
- GPU vs CPU within a loose relative tolerance (1e-2) — documents the known
  arithmetic gap without demanding a match.

**Real weights, opt-in only** (`BABBLE_MLX_WEIGHTS=1` and the 4-bit build in
the cache): load, shapes, bit-identical repeat, non-empty decoded text,
`describe()` reports 4-bit default with 8-bit overrides. Never in CI.

**Existing nets, unchanged:** `test_reference_parity.py` pins our sampler to
Transformers' `generate`; the new end-to-end test extends the chain to the
MLX backend.

**CI:** unchanged (Ubuntu only). The README notes that the MLX tests run
locally only.

## Documentation

- README "Setup": one line for the `mlx` extra; a row in the "Model size and
  4-bit" table (MLX 4-bit, `--model mlx`, Apple Silicon ≥ 24 GB, ~18 GB peak,
  bench caveat).
- README "Sidecar mode": `BABBLE_SIDECAR_MODEL=mlx` beside the other names.
- README "Caveats": results from an MLX build describe the quantised model;
  every condition in a run must use the same build; a finding is re-run on
  bf16. The manifest's `backend` block is what makes a mixed run detectable.
- CLAUDE.md "Contracts that must not drift": `models/mlx_gemma.py` ⇄
  `models/diffusion_gemma.py`, pinned by `tests/test_mlx_adapter.py`.
- `docs/experiments/LOG.md`: an entry when the backend lands; the automatic
  entries carry the `backend` block from then on.

## Open items (not blocking)

- 4-bit vs bf16 fidelity: an experiment for the CUDA box, or the 8-bit MLX
  build (~27 GB) on the Mac as an intermediate.
- Logit-lens agreement reads 0 at layers 6–18 on step 1 of the real model.
  Plausible on a random canvas; the parity test on the tiny model will show
  whether the capture is right, and a real-weights check on a converged
  canvas will show whether the reading is.
- Every real-model output opens with an empty `thought` channel marker; the
  text metrics should strip it. Belongs to the analysis, not the backend.
