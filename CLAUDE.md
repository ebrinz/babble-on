# babble-on — working notes for Claude

Two streams live in this repository. Keep them separate unless a note here
says otherwise.

1. **The application** (`src-tauri/`, `src/`, `diffusion-rs/`, `bbrec/`): the
   Tauri observatory. Plaid-1B runs in-process; DiffusionGemma runs through the
   harness sidecar (engine drop-down). It records streams (`.bbrec`) and
   exports seed bundles for the harness.
2. **The experiment** (`harness/`, `docs/experiments/`): does text diffused
   from out-of-coherence entropy differ, mechanistically, from in-coherence
   entropy? Offline, Python, driven by recordings. Findings are expected to
   guide the app's design eventually; until then the app only *feeds* the
   experiment and does not depend on its results.

## The experiment log is mandatory

`docs/experiments/LOG.md` is the append-only record of everything tried.

- The harness appends an entry automatically at the end of `analyze_run`
  (every `run` and `analyze` command). Do not disable this.
- When you run an experiment by hand, change a method, record a session, hit
  a dead end, or decide something about the experiment, append an entry
  yourself: date, what was done, outcome, and what it implies. Short is fine;
  missing is not.
- When a run is worth keeping, copy its `report.md` and `report-assets/` into
  `docs/experiments/runs/<name>/` and link it from the entry.

## Reports

Reports are Markdown with SVG figures (`harness/babble_harness/svg.py`):
transparent backgrounds, mid-tone inks only, so they read on light and dark
pages. The four condition colours are validated for both surfaces; keep the
mapping `in_band` green, `out_band` gold, `prng` blue, `remote` orange. Charts
animate once (CSS keyframes, disabled under `prefers-reduced-motion`) and only
where motion carries meaning.

## Contracts that must not drift

- bytes → uniform → index: `bbrec/src/noise.rs` ⇄ `harness/babble_harness/noise.py`,
  pinned by `docs/contract/noise_vectors.json`. Change both or neither.
- `.bbrec` and seed-bundle formats: `bbrec/src/{recording,seed}.rs` ⇄
  `harness/babble_harness/{recording,noise}.py`.
- The sampler mirrors Transformers' `generation_diffusion_gemma.py`;
  `harness/tests/test_reference_parity.py` enforces it. If Transformers
  changes, the parity test tells you first.

## Running the checks

```
cd src-tauri && cargo test          # app (Linux needs libgtk-3-dev libwebkit2gtk-4.1-dev libudev-dev)
cd bbrec && cargo test              # shared contracts
cd harness && .venv/bin/python -m pytest -q
npm test && npm run build           # frontend
```

`--model stub` and `--model tiny` run the harness without weights; `tiny`
is the real Transformers adapter on a random-weight DiffusionGemma.

## Where the designs live

`docs/superpowers/specs/` (design), `docs/superpowers/plans/` (plans),
`docs/*.md` (audits). The 2026-10-02 spec covers the harness and the
DiffusionGemma move.
