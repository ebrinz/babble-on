# Device-Composed Prompts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** With DiffusionGemma selected and the prompt field blank, the first 16 seed bytes pick a subject, setting, mood and form; the sidecar builds the prompt from them and the app shows the drawn brief above the text.

**Architecture:** The harness gains `compose.py` (lists + `compose(raw)`), and `serve.py` honours a `compose` request flag: it splits the first 16 bytes off the entropy, emits a `brief` message, and samples from the rest. Rust adds 16 bytes to the seed it draws and sets the flag when the prompt is blank; the frontend renders the `brief` event as one line above the output, marking bank-drawn picks.

**Tech Stack:** Python 3.12 (harness venv, pytest), Rust/Tauri 2 (cargo test), TypeScript (vitest, Playwright headless test).

**Spec:** `docs/superpowers/specs/2026-10-14-prompt-composition-design.md`

## Global Constraints

- Trigger: DiffusionGemma engine + blank (trimmed) prompt ⇒ compose. A typed prompt is used exactly as today. Plaid unchanged.
- `BYTES_PER_BRIEF = 16` (4 bytes per slot, order subject, setting, mood, form); Rust `BRIEF_BYTES` must equal it.
- Lists: the spike's four 32-entry lists from `spikes/prompt_compose/compose.py`, verbatim, in the spike's order. Not editable.
- Prompt text: `"Write {form} about {subject}, {setting}. The mood is {mood}. Keep it under 200 words."`
- Composition uses `bytes_to_uniforms` and `uniform_to_index` from `babble_harness.noise`; no other mapping.
- `compose: true` requires `entropy_hex` (no PRNG fallback) of at least `16 + budget_bytes(seq_len, steps)` bytes.
- Message `brief`: `{"type": "brief", "picks": {...}, "prompt": "...", "bytes": 16}`, before the first `step`; `done` carries `"brief": {"picks", "prompt"}` and `entropy_used` includes the 16 bytes.
- Brief line order: form · subject · setting · mood; slot `i` in `SLOTS` order is bank-drawn when `banked_bytes ≥ 4·(i+1)`.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Run the harness tests with `BABBLE_EXPERIMENT_LOG=0` (CI does) so no test appends to the experiment log.

## Review Focus

1. A whitespace-only prompt with DiffusionGemma must compose, not send `"   "` as the prompt (Task 3 test: `generate` trims before deciding).
2. A `compose` request that also carries a `prompt` must use the composed prompt and ignore the typed one (Task 2 test).
3. A `brief` event arriving for a Plaid generation, or a stale one, must not leave a brief line over the next generation: the line is cleared when generate is clicked (Task 4 headless test).
4. `banked_bytes` must never exceed the brief's 16 bytes in its effect on marking: with a 100 % bank seed all four picks are marked bank, with 8 bytes exactly two (Task 4 vitest at 0, 8, 16, 99 344).
5. A `done` without a preceding `brief` (missed event) must still show the brief if `done.brief` is present, and must not duplicate it if `brief` was shown (Task 4 headless test).

---

### Task 1: `compose.py` — lists and the composer

**Files:**
- Create: `harness/babble_harness/compose.py`
- Test: `harness/tests/test_compose.py`

**Interfaces:**
- Consumes: `babble_harness.noise.bytes_to_uniforms(raw, n) -> np.ndarray`, `babble_harness.noise.uniform_to_index(u, n) -> int|np.ndarray`.
- Produces: `SLOTS: dict[str, list[str]]`, `BYTES_PER_BRIEF: int = 16`, `compose(raw: bytes) -> tuple[str, dict[str, str]]`.

- [ ] **Step 1: Write the failing tests**

```python
# harness/tests/test_compose.py
import numpy as np
import pytest

from babble_harness.compose import BYTES_PER_BRIEF, SLOTS, compose


def test_slots_are_four_lists_of_32_unique_entries_in_order():
    assert list(SLOTS) == ["subject", "setting", "mood", "form"]
    for name, items in SLOTS.items():
        assert len(items) == 32, name
        assert len(set(items)) == 32, name
    assert BYTES_PER_BRIEF == 16


def test_same_bytes_same_brief_and_different_bytes_differ():
    rng = np.random.default_rng(0)
    a, b = rng.bytes(16), rng.bytes(16)
    assert compose(a) == compose(a)
    assert compose(a)[0] != compose(b)[0]


def test_extreme_bytes_pick_the_ends_of_every_list():
    _, lo = compose(bytes(16))
    _, hi = compose(bytes([255]) * 16)
    for name, items in SLOTS.items():
        assert lo[name] == items[0]
        assert hi[name] == items[-1]


def test_prompt_has_the_fixed_shape_and_every_pick():
    prompt, picks = compose(bytes(16))
    assert prompt == (f"Write {picks['form']} about {picks['subject']}, {picks['setting']}. "
                      f"The mood is {picks['mood']}. Keep it under 200 words.")


def test_only_the_first_16_bytes_matter_and_short_input_is_refused():
    raw = np.random.default_rng(1).bytes(40)
    assert compose(raw) == compose(raw[:16])
    with pytest.raises(ValueError, match="16"):
        compose(raw[:15])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd harness && BABBLE_EXPERIMENT_LOG=0 .venv/bin/python -m pytest tests/test_compose.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'babble_harness.compose'`

- [ ] **Step 3: Write the module**

Copy the four lists verbatim from `spikes/prompt_compose/compose.py` (`SLOTS`), then:

```python
# harness/babble_harness/compose.py
"""The device writes the brief: 16 entropy bytes pick a subject, setting,
mood and form from fixed lists ordered roughly from ordinary to strange, and
the prompt is built from them. Uses the bytes → uniform → index contract in
`noise.py`, so a byte stream skewed high picks from the strange end of every
list. Promoted from spikes/prompt_compose (2026-10-14)."""
from __future__ import annotations

from .noise import bytes_to_uniforms, uniform_to_index

SLOTS: dict[str, list[str]] = {
    "subject": [ ... 32 entries, verbatim from the spike ... ],
    "setting": [ ... ],
    "mood": [ ... ],
    "form": [ ... ],
}
BYTES_PER_BRIEF = 4 * len(SLOTS)


def compose(raw: bytes) -> tuple[str, dict[str, str]]:
    """(prompt, picks) from the first BYTES_PER_BRIEF bytes of `raw`."""
    if len(raw) < BYTES_PER_BRIEF:
        raise ValueError(f"brief needs {BYTES_PER_BRIEF} bytes, got {len(raw)}")
    us = bytes_to_uniforms(raw[:BYTES_PER_BRIEF], len(SLOTS))
    picks = {name: items[int(uniform_to_index(float(u), len(items)))]
             for (name, items), u in zip(SLOTS.items(), us)}
    prompt = (f"Write {picks['form']} about {picks['subject']}, {picks['setting']}. "
              f"The mood is {picks['mood']}. Keep it under 200 words.")
    return prompt, picks
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd harness && BABBLE_EXPERIMENT_LOG=0 .venv/bin/python -m pytest tests/test_compose.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add harness/babble_harness/compose.py harness/tests/test_compose.py
git commit -m "feat(harness): compose a brief from 16 entropy bytes

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `serve.py` — the `compose` request flag and the `brief` message

**Files:**
- Modify: `harness/babble_harness/serve.py` (docstring lines 1–17; request handling lines 52–79)
- Modify: `sidecar/README.md` (protocol section, after the request/response examples around lines 56–72)
- Modify: `harness/README.md:117-121` (sidecar mode paragraph)
- Test: `harness/tests/test_serve_compose.py`

**Interfaces:**
- Consumes: `compose`, `BYTES_PER_BRIEF` from Task 1.
- Produces: request field `compose: bool`; messages `brief` and `done.brief` as in Global Constraints.

- [ ] **Step 1: Write the failing test**

```python
# harness/tests/test_serve_compose.py
import json
import subprocess
import sys
from pathlib import Path

from babble_harness.compose import BYTES_PER_BRIEF, compose

CODE = ("import sys; sys.path.insert(0, %r); from babble_harness.serve import serve; "
        "serve('stub', vocab_size=64, canvas_length=16)" % str(Path(__file__).resolve().parents[1]))
BUDGET = 4 * 16 * 11  # budget_bytes(16, 5)


def _serve(*reqs: dict) -> list[dict]:
    stdin = "".join(json.dumps(r) + "\n" for r in reqs)
    p = subprocess.run([sys.executable, "-c", CODE], input=stdin, capture_output=True, text=True, timeout=120)
    return [json.loads(l) for l in p.stdout.splitlines() if l.strip()]


def test_compose_emits_brief_before_steps_and_repeats_it_in_done():
    raw = bytes(range(256)) * ((BYTES_PER_BRIEF + BUDGET) // 256 + 1)
    raw = raw[: BYTES_PER_BRIEF + BUDGET]
    lines = _serve({"steps": 5, "seq_len": 16, "entropy_hex": raw.hex(), "compose": True, "prompt": "ignored"})
    types = [l["type"] for l in lines]
    assert types[0] == "ready" and types[1] == "brief" and "step" in types and types[-1] == "done"
    brief, done = lines[1], lines[-1]
    prompt, picks = compose(raw[:BYTES_PER_BRIEF])
    assert brief == {"type": "brief", "picks": picks, "prompt": prompt, "bytes": BYTES_PER_BRIEF}
    assert done["brief"] == {"picks": picks, "prompt": prompt}
    assert done["seed"] == "entropy"
    assert done["entropy_used"] > BYTES_PER_BRIEF


def test_compose_samples_from_the_bytes_after_the_brief():
    raw = bytes([9]) * BYTES_PER_BRIEF + bytes(range(256)) * (BUDGET // 256 + 1)
    raw = raw[: BYTES_PER_BRIEF + BUDGET]
    prompt, _ = compose(raw)
    composed = _serve({"steps": 5, "seq_len": 16, "entropy_hex": raw.hex(), "compose": True})[-1]
    plain = _serve({"steps": 5, "seq_len": 16, "entropy_hex": raw[BYTES_PER_BRIEF:].hex(), "prompt": prompt})[-1]
    assert composed["tokens"] == plain["tokens"]
    assert composed["entropy_used"] == plain["entropy_used"] + BYTES_PER_BRIEF


def test_compose_refuses_missing_or_short_entropy():
    short = bytes(BYTES_PER_BRIEF + BUDGET - 1)
    lines = _serve({"steps": 5, "seq_len": 16, "compose": True},
                   {"steps": 5, "seq_len": 16, "entropy_hex": short.hex(), "compose": True})
    errors = [l["message"] for l in lines if l["type"] == "error"]
    assert len(errors) == 2
    assert "entropy" in errors[0]
    assert str(BYTES_PER_BRIEF + BUDGET) in errors[1] and str(len(short)) in errors[1]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd harness && BABBLE_EXPERIMENT_LOG=0 .venv/bin/python -m pytest tests/test_serve_compose.py -q`
Expected: 3 failed (`types[1] == "brief"` is false; token mismatch; only one error, from the PRNG fallback path there are zero errors for the first request)

- [ ] **Step 3: Implement in `serve.py`**

Replace the block from `need = budget_bytes(seq_len, steps)` through `den.set_prompt(...)` with:

```python
            compose_brief = bool(req.get("compose", False))
            need = budget_bytes(seq_len, steps)
            brief = None
            if compose_brief:
                if not req.get("entropy_hex"):
                    emit({"type": "error", "message": "compose needs entropy_hex: the brief must come from the stream"}); continue
                raw = bytes.fromhex(req["entropy_hex"])
                total = BYTES_PER_BRIEF + need
                if len(raw) < total:
                    emit({"type": "error", "message": f"entropy_hex holds {len(raw)} bytes, need {total} "
                                                      f"({BYTES_PER_BRIEF} for the brief + {need} for the sampler)"}); continue
                prompt_text, picks = compose(raw[:BYTES_PER_BRIEF])
                brief = {"picks": picks, "prompt": prompt_text}
                tape, seed = EntropyTape(raw[BYTES_PER_BRIEF:], label="app"), "entropy"
                emit({"type": "brief", **brief, "bytes": BYTES_PER_BRIEF})
            elif req.get("entropy_hex"):
                raw = bytes.fromhex(req["entropy_hex"])
                if len(raw) < need:
                    emit({"type": "error", "message": f"entropy_hex holds {len(raw)} bytes, need {need}"}); continue
                tape, seed = EntropyTape(raw, label="app"), "entropy"
            else:
                tape, seed = prng_tape(int(req.get("seed", 0)), need), "prng"
            if hasattr(den, "set_prompt"):
                den.set_prompt(brief["prompt"] if brief else (req.get("prompt") or None))
```

In the `done` emit, add `"brief": brief,` and change `"entropy_used": res.tape_consumed` to `"entropy_used": res.tape_consumed + (BYTES_PER_BRIEF if brief else 0)`. Add `from .compose import BYTES_PER_BRIEF, compose` to the imports. Extend the module docstring:

```
  stdin : ... optional "compose": true — the first 16 entropy bytes pick a
          subject/setting/mood/form (compose.py) and become the prompt; any
          "prompt" in the request is ignored; entropy_hex is then required
          and must hold 16 + budget_bytes(seq_len, steps) bytes.
  stdout: ... with "compose", {"type":"brief","picks":{...},"prompt":"...","bytes":16}
          before the first step, and "brief" repeated in "done".
```

- [ ] **Step 4: Run the tests to verify they pass, then the whole harness suite**

Run: `cd harness && BABBLE_EXPERIMENT_LOG=0 .venv/bin/python -m pytest tests/test_serve_compose.py -q`
Expected: `3 passed`
Run: `cd harness && BABBLE_EXPERIMENT_LOG=0 .venv/bin/python -m pytest -q`
Expected: all pass (82 passed, 1 skipped — the prior 74 + 5 + 3)

- [ ] **Step 5: Document the protocol**

Append to the protocol section of `sidecar/README.md` (after the `done` example):

```markdown
With `"compose": true` (the app's DiffusionGemma engine with a blank prompt)
the first 16 bytes of `entropy_hex` choose a subject, setting, mood and form
(`babble_harness/compose.py`), the prompt is built from them (any `prompt`
field is ignored), and the sidecar emits
`{"type":"brief","picks":{"subject":…,"setting":…,"mood":…,"form":…},"prompt":"…","bytes":16}`
before the first `step`; `done` repeats it under `"brief"`. `entropy_hex`
must then hold `16 + budget_bytes(seq_len, steps)` bytes; there is no PRNG
fallback for a composed brief.
```

In `harness/README.md` sidecar-mode paragraph change `(`steps`, `seq_len`, `entropy_hex`, `prompt` → `step`/`done` events)` to `(`steps`, `seq_len`, `entropy_hex`, `prompt` or `compose` → `brief`/`step`/`done` events)`.

- [ ] **Step 6: Commit**

```bash
git add harness/babble_harness/serve.py harness/tests/test_serve_compose.py sidecar/README.md harness/README.md
git commit -m "feat(sidecar): compose flag — the first 16 entropy bytes write the brief

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Rust — draw 16 more bytes, send `compose`, report `banked_bytes`

**Files:**
- Modify: `src-tauri/src/sidecar.rs:26-31` (add `BRIEF_BYTES`), tests at lines 209–213 and 259–302
- Modify: `src-tauri/src/lib.rs:60-121` (`generate`), `133-199` (`run_gemma_generation`)

**Interfaces:**
- Consumes: sidecar `compose` flag and `brief` message (Task 2).
- Produces: `pub const BRIEF_BYTES: usize = 16;` `seeded` event field `banked_bytes: usize`; `run_gemma_generation(app, gemma, steps, prompt, compose: bool, need, draw_seed)`.

- [ ] **Step 1: Write the failing tests**

In `src-tauri/src/sidecar.rs` tests, extend `budget_matches_python`:

```rust
    #[test]
    fn budget_matches_python() {
        assert_eq!(budget_bytes(256, 48), 99_328);
        assert_eq!(budget_bytes(16, 5), 4 * 16 * 11);
        assert_eq!(BRIEF_BYTES, 16); // babble_harness.compose.BYTES_PER_BRIEF
    }
```

In `stub_sidecar_round_trip`, before the "too little entropy" block, add:

```rust
        // compose: 16 extra bytes, a brief before the steps, repeated in done
        let entropy = vec![7u8; BRIEF_BYTES + budget_bytes(seq_len, steps)];
        sc.request(&serde_json::json!({
            "steps": steps, "seq_len": seq_len, "entropy_hex": hex(&entropy), "preview_every": 2, "compose": true
        }))
        .unwrap();
        let mut brief_seen = false;
        let done = loop {
            match sc.next_event(Duration::from_secs(60)).unwrap() {
                Event::Message(v) => match v["type"].as_str() {
                    Some("brief") => {
                        assert!(!brief_seen, "brief emitted once");
                        assert_eq!(v["bytes"], BRIEF_BYTES);
                        assert!(v["picks"]["subject"].is_string() && v["prompt"].as_str().unwrap().starts_with("Write "));
                        brief_seen = true;
                    }
                    Some("step") => assert!(brief_seen, "brief precedes the first step"),
                    Some("done") => break v,
                    Some(t) => panic!("unexpected {t}: {v}"),
                    None => unreachable!(),
                },
                Event::Ready { .. } => continue,
                Event::Exit => panic!("sidecar exited"),
            }
        };
        assert!(brief_seen);
        assert_eq!(done["brief"]["picks"]["mood"].is_string(), true);
        assert_eq!(done["seed"], "entropy");
```

In `src-tauri/src/lib.rs` add a unit test module (at the end of the file) for the trigger decision, after extracting it into a function:

```rust
#[cfg(test)]
mod compose_tests {
    use super::should_compose;

    #[test]
    fn blank_or_whitespace_prompt_composes_only_for_gemma() {
        assert!(should_compose(true, None));
        assert!(should_compose(true, Some("   ")));
        assert!(!should_compose(true, Some(" sea ")));
        assert!(!should_compose(false, None));
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --manifest-path src-tauri/Cargo.toml -q 2>&1 | tail -5`
Expected: compile error (`BRIEF_BYTES` and `should_compose` not found)

- [ ] **Step 3: Implement**

`src-tauri/src/sidecar.rs`, after `budget_bytes`:

```rust
/// Bytes the sidecar spends on a composed brief (4 per slot: subject,
/// setting, mood, form) before the sampler tape starts. Mirrors
/// `babble_harness.compose.BYTES_PER_BRIEF`.
pub const BRIEF_BYTES: usize = 16;
```

`src-tauri/src/lib.rs`:

```rust
/// A blank prompt with the DiffusionGemma engine means the device writes the brief.
fn should_compose(use_gemma: bool, prompt: Option<&str>) -> bool {
    use_gemma && prompt.map(str::trim).map_or(true, str::is_empty)
}
```

In `generate`, replace the `need` computation:

```rust
    let compose = should_compose(use_gemma, prompt.as_deref());
    let need = if use_gemma {
        let budget = sidecar::budget_bytes(sidecar::CANVAS_LENGTH, steps);
        if compose { sidecar::BRIEF_BYTES + budget } else { budget }
    } else {
        n_samples * seq_len * EMBED_DIM * 4
    };
```

and pass `compose` through: `run_gemma_generation(&app2, &gemma2, steps, prompt, compose, need, draw_seed)`.

In `run_gemma_generation`, add the parameter `compose: bool` after `prompt`, emit `banked_bytes` with `seeded`, and build the request:

```rust
    let banked_bytes = (seed.bank_fraction * need as f64).round() as usize;
    let _ = app.emit(
        "diffusion",
        json!({"type": "seeded", "bank_fraction": seed.bank_fraction, "tags": seed.tags, "banked_bytes": banked_bytes}),
    );
    let mut req = json!({"steps": steps, "seq_len": sidecar::CANVAS_LENGTH, "preview_every": 1,
                         "entropy_hex": sidecar::hex(&seed.bytes[..need])});
    if compose {
        req["compose"] = json!(true);
    } else if let Some(p) = prompt.as_deref().map(str::trim).filter(|p| !p.is_empty()) {
        req["prompt"] = json!(p);
    }
```

The event loop already forwards every message (including `brief`) to the frontend; no change there.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --manifest-path src-tauri/Cargo.toml -q 2>&1 | grep "test result"`
Expected: first line `test result: ok. 36 passed` (34 + the new lib test + no change in count for the extended tests… the stub round trip and budget test are existing tests, so 35 passed if only `blank_or_whitespace_prompt_composes_only_for_gemma` is new; accept 35 or 36, zero failed)

- [ ] **Step 5: Commit**

```bash
git add src-tauri/src/sidecar.rs src-tauri/src/lib.rs
git commit -m "feat(app): blank prompt lets the device write the DiffusionGemma brief

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Frontend — the brief line

**Files:**
- Modify: `src/render.ts` (add `briefLine` after `seedStamp`, line 76)
- Modify: `src/main.ts:176-208` (event switch), `220-230` (generate click)
- Modify: `index.html:42-43` (placeholder), `56` (add `#gen-brief` above `#gen-seed`)
- Modify: `src/style.css` (after line 33)
- Test: `src/render.test.ts`, `tests/ui/headless.mjs` (after the `seeded` assertion, line 95)

**Interfaces:**
- Consumes: `brief` event `{picks, prompt, bytes}`, `done.brief`, `seeded.banked_bytes` (Tasks 2–3).
- Produces: `briefLine(picks: Record<string, string>, bankedBytes: number): { text: string; parts: { slot: string; value: string; bank: boolean }[] }`.

- [ ] **Step 1: Write the failing vitest**

Add to `src/render.test.ts` (import `briefLine` from `./render`):

```ts
  it("briefLine orders form · subject · setting · mood and marks bank-drawn slots", () => {
    const picks = { subject: "a dog", setting: "on a farm", mood: "calm", form: "a letter" };
    const none = briefLine(picks, 0);
    expect(none.text).toBe("a letter · a dog · on a farm · calm");
    expect(none.parts.map((p) => p.bank)).toEqual([false, false, false, false]);
    // 8 banked bytes cover subject and setting (slots 0 and 1 in list order)
    const half = briefLine(picks, 8);
    expect(half.parts.map((p) => [p.slot, p.bank])).toEqual([
      ["form", false], ["subject", true], ["setting", true], ["mood", false],
    ]);
    expect(briefLine(picks, 16).parts.every((p) => p.bank)).toBe(true);
    expect(briefLine(picks, 99_344).parts.every((p) => p.bank)).toBe(true);
    // 7 bytes: subject's 4 are covered, setting's are not
    expect(briefLine(picks, 7).parts.map((p) => p.bank)).toEqual([false, true, false, false]);
  });
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npm test 2>&1 | tail -5`
Expected: FAIL, `briefLine` is not exported

- [ ] **Step 3: Implement `briefLine`**

```ts
// src/render.ts, after seedStamp
const SLOT_ORDER = ["subject", "setting", "mood", "form"]; // byte order in babble_harness.compose
const DISPLAY_ORDER = ["form", "subject", "setting", "mood"];

/** One line naming the device's brief; a slot is bank-drawn when the bank
 *  covered all four of its bytes (slot i spans bytes 4i..4i+3). */
export function briefLine(
  picks: Record<string, string>,
  bankedBytes: number,
): { text: string; parts: { slot: string; value: string; bank: boolean }[] } {
  const parts = DISPLAY_ORDER.map((slot) => ({
    slot,
    value: picks[slot] ?? "",
    bank: bankedBytes >= 4 * (SLOT_ORDER.indexOf(slot) + 1),
  }));
  return { text: parts.map((p) => p.value).join(" · "), parts };
}
```

- [ ] **Step 4: Run the vitest to verify it passes**

Run: `npm test 2>&1 | tail -3`
Expected: `Tests  8 passed (8)`

- [ ] **Step 5: Write the failing headless assertions**

In `tests/ui/headless.mjs`, after the `seed stamp rendered` assertion, insert:

```js
await page.evaluate(() => window.__emit("diffusion", { type: "brief", bytes: 16, prompt: "Write a letter about a dog, on a farm. The mood is calm. Keep it under 200 words.", picks: { subject: "a dog", setting: "on a farm", mood: "calm", form: "a letter" } }));
assert((await text("#gen-brief")) === "a letter · a dog · on a farm · calm", "brief line rendered from the brief event");
assert((await page.$$eval("#gen-brief .pick.bank", (t) => t.length)) === 4, "72% of 99344 bytes covers all four picks as bank-drawn");
```

Change the earlier `seeded` emit to include `banked_bytes: 71528` (72 % of 99 344). After the `done ... stopped_early` assertion add:

```js
assert((await text("#gen-brief")) === "a letter · a dog · on a farm · calm", "done with no brief leaves the line alone");
await page.evaluate(() => window.__emit("diffusion", { type: "done", i: 5, total: 5, tokens: ["x"], elapsed: 1, seed: "entropy", brief: { picks: { subject: "a cup of tea", setting: "in a small town", mood: "warm", form: "a postcard" }, prompt: "…" } }));
assert((await text("#gen-brief")) === "a letter · a dog · on a farm · calm", "a brief already shown is not replaced by done");
```

And at the top of the generate section, before `await page.fill("#prompt", ...)`:

```js
await page.evaluate(() => window.__emit("diffusion", { type: "brief", bytes: 16, prompt: "x", picks: { subject: "stale", setting: "s", mood: "m", form: "f" } }));
```

then after `await page.click("#gen-btn");` add:

```js
assert((await text("#gen-brief")) === "", "clicking generate clears a stale brief line");
```

Also add a case where `done` arrives with a brief and nothing was shown: before the final `export seed` section:

```js
await page.click("#gen-btn"); // new generation, line cleared
await page.evaluate(() => window.__emit("diffusion", { type: "done", i: 5, total: 5, tokens: ["x"], elapsed: 1, seed: "entropy", brief: { picks: { subject: "a cup of tea", setting: "in a small town", mood: "warm", form: "a postcard" }, prompt: "…" } }));
assert((await text("#gen-brief")) === "a postcard · a cup of tea · in a small town · warm", "done renders the brief when the brief event was missed");
```

- [ ] **Step 6: Run the headless test to verify it fails**

Run: `npm run test:ui 2>&1 | tail -3`
Expected: fails at "brief line rendered from the brief event" (no `#gen-brief` element)

- [ ] **Step 7: Implement the UI**

`index.html`: change the prompt placeholder to `prompt — leave blank: Plaid babbles from pure entropy; DiffusionGemma lets the device write the brief` and insert `<div id="gen-brief" class="gen-brief"></div>` before the `#gen-seed` div.

`src/main.ts`: import `briefLine`; add state and a renderer near `renderCrystal`:

```ts
let bankedBytes = 0;
const genBrief = document.getElementById("gen-brief")!;

function renderBrief(picks: Record<string, string>) {
  const { parts } = briefLine(picks, bankedBytes);
  genBrief.innerHTML = parts
    .map((p) => `<span class="pick${p.bank ? " bank" : ""}" title="${p.slot}${p.bank ? " · anomaly bank" : " · live stream"}">${escape(p.value)}</span>`)
    .join(" · ");
}
```

In the `diffusion` listener add:

```ts
    case "brief":
      renderBrief(m.picks || {});
      break;
```

in `case "seeded":` add `bankedBytes = m.banked_bytes || 0;`, and in `case "done":` add, before `renderCrystal`:

```ts
      if (m.brief && m.brief.picks && !genBrief.textContent) renderBrief(m.brief.picks);
```

In the `genBtn` click handler, next to clearing `#gen-seed`, add `genBrief.textContent = ""; bankedBytes = 0;`.

`src/style.css`, after the `.gen-output` rule:

```css
.gen-brief { font-size:13px; color:#9aaaa5; margin:0 0 4px; min-height:1.2em; }
.gen-brief .pick.bank { color:var(--gold); }
```

- [ ] **Step 8: Run the headless test and the frontend build to verify they pass**

Run: `npm run test:ui 2>&1 | tail -3`
Expected: `UI TEST PASSED`
Run: `npm run build 2>&1 | tail -2`
Expected: `✓ built`

- [ ] **Step 9: Commit**

```bash
git add src/render.ts src/render.test.ts src/main.ts index.html src/style.css tests/ui/headless.mjs
git commit -m "feat(ui): show the device's brief above the DiffusionGemma text

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Full verification and the experiment-log entry

**Files:**
- Modify: `docs/experiments/LOG.md` (append)

- [ ] **Step 1: Run every suite on the final tree**

Run: `cargo test --manifest-path src-tauri/Cargo.toml -q 2>&1 | grep "test result" | head -1`
Expected: `ok`, 0 failed
Run: `cargo test --manifest-path bbrec/Cargo.toml -q 2>&1 | grep "test result" | head -1`
Expected: `ok. 13 passed`
Run: `cd harness && BABBLE_EXPERIMENT_LOG=0 .venv/bin/python -m pytest -q 2>&1 | tail -1`
Expected: `82 passed, 1 skipped`
Run: `npm test 2>&1 | tail -2 && npm run build 2>&1 | tail -1 && npm run test:ui 2>&1 | tail -1`
Expected: `8 passed`, `✓ built`, `UI TEST PASSED`

- [ ] **Step 2: Append the log entry**

```markdown

## 2026-10-14 — Device-composed prompts shipped in the DiffusionGemma engine

**Question:** Can the app let the TrueRNG write DiffusionGemma's brief, so
the device authors content rather than wording (prompt-composition spike,
2026-10-14)?

**Done:** Spec `docs/superpowers/specs/2026-10-14-prompt-composition-design.md`,
plan `docs/superpowers/plans/2026-10-14-prompt-composition.md`. The spike's
composer moved to `harness/babble_harness/compose.py`; `serve.py` takes
`compose: true`, spends the first 16 entropy bytes on the brief and emits a
`brief` message; the app draws 16 more seed bytes when the prompt is blank
with DiffusionGemma selected, sends the flag, and shows the brief above the
text with bank-drawn picks in gold. Commits <hashes>.

**Result:** Tests: harness 82, src-tauri cargo, vitest 8, headless UI, all
green; the stub-sidecar round trip sees `brief` before the first step.
Judged by eye: _(fill in after running the app with the TrueRNG: does the
brief line read well, do bank-drawn picks stand out, does the composed text
follow the brief as in the spike)_.

**Decision:** Shipped. Open: list design (the spike's lists are a first
draft) and whether a typed prompt should be allowed to extend a composed
brief (spec non-goal for now).
```

Fill `<hashes>` with `git log --oneline -4`.

- [ ] **Step 3: Commit**

```bash
git add docs/experiments/LOG.md
git commit -m "docs(experiments): log the device-composed prompt feature

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
