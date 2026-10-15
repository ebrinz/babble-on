# babble-on — Device-composed prompts for the DiffusionGemma engine — Design

**Date:** 2026-10-14
**Status:** Approved (design); not yet planned or implemented
**Builds on:** `spikes/prompt_compose/` (spike, logged 2026-10-14), the
harness sidecar (`2026-10-02-diffusiongemma-harness-design.md`) and the MLX
backend (`2026-10-06-mlx-backend-design.md`)

## One line

With DiffusionGemma selected and the prompt field blank, the device writes
the brief: the first 16 bytes of the generation's seed pick a subject,
setting, mood and form from fixed ordered lists, the sidecar turns them into
the prompt, and the app shows the drawn brief above the text.

## Why

Every dose experiment showed DiffusionGemma's content is fixed by the prompt
and the seed bytes only touch wording. The prompt-composition spike showed
that letting bytes compose the prompt hands content to the device (12/12
passages followed every pick; word overlap between passages 0.08 against
0.3–0.5 with a fixed prompt) with the model's usual prose. Today a blank
prompt sends an empty user turn, which makes the model ask what the question
was; composition replaces that.

## Decisions (agreed)

- **Trigger:** DiffusionGemma + blank prompt field ⇒ compose. A typed prompt
  is used exactly as today. Plaid is unchanged.
- **Where:** the sidecar composes. The lists and `compose()` live in the
  harness beside the bytes → uniform → index contract they rely on; Rust and
  the frontend never hold the lists. Rejected: composing in Rust (duplicated
  lists and mapping) or in the frontend (entropy handling in the UI layer).
- **Which bytes:** the first 16 bytes of the seed the app already draws
  (anomaly bank first, then live stream). The sampler tape is the bytes
  after them. So bank bytes, when present, choose the brief.
- **Display:** one brief line above the output, e.g. "a recipe · a fisherman
  · inside a single raindrop · feverish", with each pick marked as drawn from
  the anomaly bank or the live stream.
- **Lists:** fixed, the spike's four 32-entry lists ordered roughly from
  ordinary to strange. Not editable in the UI.

## Components

### New: `harness/babble_harness/compose.py`

- `SLOTS: dict[str, list[str]]` — `subject`, `setting`, `mood`, `form`, 32
  unique entries each, in the spike's order (`spikes/prompt_compose/compose.py`).
- `BYTES_PER_BRIEF = 16` (4 bytes per slot, in `SLOTS` order).
- `compose(raw: bytes) -> tuple[str, dict[str, str]]` — `bytes_to_uniforms`
  then `uniform_to_index` per slot (the harness contract); returns the prompt
  `"Write {form} about {subject}, {setting}. The mood is {mood}. Keep it under
  200 words."` and the picks. Raises `ValueError` on fewer than 16 bytes.

### Changed: `harness/babble_harness/serve.py`

Request field `compose: bool` (default false). When true:
- `entropy_hex` is required (no PRNG fallback) and must hold
  `BYTES_PER_BRIEF + budget_bytes(seq_len, steps)` bytes, else an `error`
  reply naming both numbers.
- The first 16 bytes go to `compose()`; the tape is `EntropyTape` over the
  rest. A `prompt` in the same request is ignored.
- Emits `{"type": "brief", "picks": {...}, "prompt": "...", "bytes": 16}`
  before the first `step`.
- The `done` message carries `"brief": {"picks": ..., "prompt": ...}`.
- `entropy_used` in `done` counts the brief bytes too.

Module docstring and `sidecar/README.md` protocol section document the field
and the message.

### Changed: `src-tauri/src/lib.rs`

- `generate`: when `use_gemma` and the trimmed prompt is empty,
  `compose = true` and `need = BRIEF_BYTES + budget_bytes(CANVAS_LENGTH, steps)`
  (`BRIEF_BYTES = 16` in `sidecar.rs`, with a test pinning it to the Python
  constant's value).
- `run_gemma_generation` takes `compose: bool`; the request carries
  `"compose": true` and no `prompt` when set.
- The `seeded` event gains `banked_bytes` (bank bytes at the start of the
  seed: `round(bank_fraction × need)`), so the UI can tell which picks came
  from the bank.

### Changed: frontend

- `src/render.ts`: `briefLine(picks, bankedBytes)` returns the line in order
  form · subject · setting · mood, each pick marked bank or live: slot `i`
  (in `SLOTS` order) is bank-drawn when `bankedBytes ≥ 4·(i+1)`. Marking is a
  class on a span (rendered by `main.ts`), with a plain-text form for tests.
- `src/main.ts`: on `brief`, render the line into a new `#gen-brief` element
  above `#gen-output`; clear it when a generation starts; on `done` with a
  `brief` and an empty `#gen-brief`, render from `done`. Remember
  `banked_bytes` from `seeded`.
- `index.html`: `#gen-brief` element; the prompt placeholder says that with
  DiffusionGemma a blank prompt lets the device write the brief.
- `src/style.css`: brief line style; bank-drawn picks in the anomaly colour
  used elsewhere for the bank.

## Data flow

1. User presses generate with DiffusionGemma and a blank prompt.
2. Rust waits for the sidecar, then draws `16 + 99 328` bytes bank-first.
   Emits `seeded` with `bank_fraction`, `tags`, `banked_bytes`.
3. Request `{"steps", "seq_len", "preview_every", "entropy_hex", "compose": true}`.
4. Sidecar composes from bytes 0–15, emits `brief`, sets the prompt, samples
   from bytes 16…; emits `step`s and `done` (with `brief`).
5. UI shows the brief line, the crystallising text, and the seed stamp.

## Error handling

- Seed shorter than `16 + budget`: Rust refuses before the request, with the
  existing "only N of M seed bytes" message; the sidecar would refuse too.
- `compose` without `entropy_hex`: sidecar `error` ("compose needs entropy").
- Typed prompt: unchanged path, no `compose` field.
- Sidecar/model failure: unchanged (bank not spent before the engine is ready).

## Testing

- `harness/tests/test_compose.py`: lists unique and 32 long; same bytes →
  same brief; all-zero bytes → first entry of every list, all-0xFF → last;
  prompt contains every pick; short input refused.
- `harness/tests/test_serve_compose.py` (stub model): a compose request emits
  `brief` before any `step`; `done.brief` matches; the tape starts at byte 16
  (the result equals a non-compose request on `bytes[16:]` with the composed
  prompt); too-short entropy and missing entropy are refused.
- `src-tauri`: `BRIEF_BYTES` pinned; the stub-sidecar round trip sends a
  compose request and sees `brief` then `done` with `brief`.
- `src/render.test.ts`: `briefLine` order and bank/live marking at 0, 8 and
  16 banked bytes.
- `tests/ui/headless.mjs`: a simulated `brief` event renders `#gen-brief`;
  a new generation clears it.

## Documentation

- `sidecar/README.md` and `harness/README.md` (sidecar mode): the `compose`
  field and `brief` message.
- `docs/experiments/LOG.md`: an entry when it ships (UI change judged by eye).

## Non-goals

- Editable or user-supplied lists; more or different slots.
- Combining a typed prompt with a composed brief.
- Composition for Plaid, or in the harness `run` experiments and seed-bundle
  export (both unchanged).
