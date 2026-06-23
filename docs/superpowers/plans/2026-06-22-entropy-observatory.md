# Entropy Observatory Implementation Plan (Plan 1 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Tauri desktop app that opens a TrueRNG (or simulator), streams its bytes, and renders a live entropy/coherence observatory — the ghostty-rng TUI reborn as a GUI.

**Architecture:** A single Rust binary (Tauri 2) owns the entropy source and statistics engine on a background thread, emitting a serialized `Snapshot` to the webview at ~20 Hz. The webview renders the coherence random-walk, entropy/NIST panels, byte histogram, and bitstream ribbon on HTML canvases. The proven `source` / `stats` / `math` modules are **ported verbatim** from `../ghostty-rng` (they ship with their own test suites); only the Tauri shell, a serialization DTO, and the frontend are new.

**Tech Stack:** Rust, Tauri 2, `serialport`, `serde`; TypeScript + Vite (vanilla-ts template), HTML5 canvas; `vitest` for pure frontend helpers.

## Global Constraints

- **Source of truth for ported modules:** `/Users/crashy/Development/ghostty-rng/src/{math.rs,stats.rs,source.rs}`. Copy verbatim; do not refactor logic. The only edits permitted are removing the `audit` module reference (Plan 1 does not ship the audit) and adjusting `use crate::...` paths if needed.
- **Rust edition 2021**, Tauri **2.x**.
- **Palette (reuse from ghostty-rng):** turquoise `#6cf0d0`, gold `#f2c14e`, on a near-black background. Significance bands: 95% gold, 99% orange, 99.9% red.
- **No audit, no diffusion** in this plan (those are Plans 2–3). Text-generation UI is out of scope here.
- **JSON cannot represent `NaN`.** Warmup metrics carry `f64::NAN`; every DTO boundary must map `NaN → null`. This is a recurring requirement for any task touching the DTO.
- **Coherence envelope constants** (verbatim): `Z95 = 1.95996398`, `Z99 = 2.57582930`, `Z999 = 3.29052673`.
- Frequent commits: one per task minimum.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src-tauri/src/math.rs` | Ported: erfc / normal / chi-square p-values. |
| `src-tauri/src/stats.rs` | Ported: sliding-window stats + coherence walk. |
| `src-tauri/src/source.rs` | Ported: threaded TrueRNG/simulate/bad reader, autodetect. |
| `src-tauri/src/dto.rs` | New: serde-serializable `SnapshotDto` (NaN→null). |
| `src-tauri/src/engine.rs` | New: owns Source+Stats, per-tick advance → `SnapshotDto`. |
| `src-tauri/src/lib.rs` | New: Tauri commands, control channel, background emit loop. |
| `src/index.html` | Observatory layout (canvases + controls). |
| `src/main.ts` | Listen to `snapshot` events, dispatch to renderers, wire controls. |
| `src/render.ts` | Pure draw helpers (envelopes, color mapping) + canvas painters. |
| `src/render.test.ts` | vitest unit tests for the pure helpers. |
| `src/style.css` | Palette + grid layout. |

---

## Task 1: Scaffold the Tauri app

**Files:**
- Create: project scaffold at repo root (`src-tauri/`, `src/`, `package.json`, etc.)

**Interfaces:**
- Produces: a runnable Tauri 2 app with `invoke('ping')` returning `"pong"`.

- [ ] **Step 1: Scaffold with the vanilla-ts template (non-interactive)**

Run from repo root:
```bash
npm create tauri-app@latest . -- --template vanilla-ts --manager npm --yes
npm install
```
If the directory-not-empty prompt blocks `--yes`, scaffold in a temp dir and move `src/`, `src-tauri/`, `package.json`, `index.html`, `vite.config.ts`, `tsconfig.json` into the repo root, preserving the existing `docs/` and `.gitignore`.

- [ ] **Step 2: Add a trivial command to prove the bridge**

In `src-tauri/src/lib.rs`, ensure a command exists:
```rust
#[tauri::command]
fn ping() -> String {
    "pong".to_string()
}
```
Register it in the builder: `.invoke_handler(tauri::generate_handler![ping])`.

- [ ] **Step 3: Build the Rust side to verify the toolchain**

Run: `cd src-tauri && cargo build`
Expected: compiles successfully (first build downloads Tauri crates).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: scaffold Tauri 2 vanilla-ts app"
```

---

## Task 2: Port `math.rs`

**Files:**
- Create: `src-tauri/src/math.rs` (verbatim copy)
- Modify: `src-tauri/src/lib.rs` (add `mod math;`)

**Interfaces:**
- Produces: `erfc`, `normal_sf`, `normal_two_sided`, `chi_square_sf`, `ln_gamma`, `gamma_q`, `chi_square_p` — all `pub fn(f64,..) -> f64`.

- [ ] **Step 1: Write failing tests**

Create `src-tauri/src/math.rs` initially containing only this test module:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn known_values() {
        // Tolerance 1e-6: the A&S 7.1.26 erfc approximation has documented max
        // error ~1.5e-7, so a tighter bound would fail on a correct port.
        assert!((erfc(0.0) - 1.0).abs() < 1e-6);
        assert!((normal_sf(0.0) - 0.5).abs() < 1e-6);
        assert!((normal_two_sided(0.0) - 1.0).abs() < 1e-6);
        // chi-square at its mean (x=k) gives p≈0.5 for large k.
        assert!((chi_square_sf(255.0, 255.0) - 0.5).abs() < 0.05);
    }
}
```
Add `mod math;` to `lib.rs`.

- [ ] **Step 2: Run to verify failure**

Run: `cd src-tauri && cargo test math::`
Expected: FAIL — `erfc` etc. not found.

- [ ] **Step 3: Copy the implementation verbatim**

Copy the function bodies (everything above the `#[cfg(test)]` block) from `/Users/crashy/Development/ghostty-rng/src/math.rs` into `src-tauri/src/math.rs`, above the test module. Do not modify the math.

- [ ] **Step 4: Run to verify pass**

Run: `cd src-tauri && cargo test math::`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src-tauri/src/math.rs src-tauri/src/lib.rs
git commit -m "feat: port numerical helpers (math.rs) from ghostty-rng"
```

---

## Task 3: Port `stats.rs` (with its full test suite)

**Files:**
- Create: `src-tauri/src/stats.rs` (verbatim copy)
- Modify: `src-tauri/src/lib.rs` (add `mod stats;`)

**Interfaces:**
- Consumes: `crate::math::{chi_square_sf, normal_sf, normal_two_sided}`.
- Produces:
  - `pub const Z95/Z99/Z999: f64`
  - `pub struct Stats` with `new(capacity: usize)`, `push(&mut self, &[u8])`, `tick_trials(&mut self)`, `tick_curve(&mut self, f64)`, `take_alert(&mut self) -> bool`, `set_capacity(&mut self, usize)`, `reset(&mut self)`, `snapshot(&self) -> Snapshot`.
  - `pub struct Snapshot { window_len, total_bytes, throughput_bps, shannon, min_entropy, monobit, chi_square, serial_corr: Metric; monobit_p, chi_square_p, hist_expected: f64; histogram: [u32;256]; recent: Vec<u8>; coherence_sigma: f64; coherence_band: Band; trial_count: u64; walk: Vec<(f64,f64)>; anomalies: Vec<AnomalyEvent> }`
  - `pub struct Metric { value: f64, quality: f64, verdict: Verdict }`
  - `pub enum Verdict { Pass, Warn, Fail, Warmup }`
  - `pub enum Band { Inside, P95, P99, P999 }` with `of(f64)`, `label()`
  - `pub struct AnomalyEvent { at_secs, peak_sigma, duration_secs: f64, band: Band, ongoing: bool }`

- [ ] **Step 1: Copy the file verbatim**

Copy `/Users/crashy/Development/ghostty-rng/src/stats.rs` to `src-tauri/src/stats.rs` unchanged (it already contains its own `#[cfg(test)]` module with 9 tests). Add `mod stats;` to `lib.rs`.

- [ ] **Step 2: Run the ported tests to verify they pass**

Run: `cd src-tauri && cargo test stats::`
Expected: PASS — all of `uniform_stream_is_ideal`, `constant_stream_is_dead`, `min_entropy_passes_for_healthy_window`, `biased_source_fails`, `band_classification`, `biased_stream_triggers_coherence_anomaly`, `uniform_stream_stays_in_band`, `window_eviction_keeps_counts_consistent`.

(If a `use crate::math::...` path fails to resolve, fix only the `use` path — never the logic.)

- [ ] **Step 3: Commit**

```bash
git add src-tauri/src/stats.rs src-tauri/src/lib.rs
git commit -m "feat: port sliding-window stats + coherence engine (stats.rs)"
```

---

## Task 4: Port `source.rs`

**Files:**
- Create: `src-tauri/src/source.rs` (verbatim copy)
- Modify: `src-tauri/src/lib.rs` (add `mod source;`), `src-tauri/Cargo.toml` (add `serialport`)

**Interfaces:**
- Consumes: `crate::stats::Stats` (via `drain_into`).
- Produces:
  - `pub enum SourceKind { Serial { path: String, baud: u32 }, Simulate, BadRng }`
  - `pub enum SourceStatus { Streaming, Reconnecting(String), Error(String) }`
  - `pub struct Source` with `spawn(SourceKind) -> Source`, fields `data_rx: Receiver<Vec<u8>>`, `status_rx: Receiver<SourceStatus>`, `label: String`.
  - `pub fn autodetect() -> Option<String>`
  - `pub fn drain_into(&Receiver<Vec<u8>>, &mut Stats) -> usize`

- [ ] **Step 1: Add the serialport dependency**

In `src-tauri/Cargo.toml` under `[dependencies]`:
```toml
serialport = "4.7"
```

- [ ] **Step 2: Copy the file verbatim**

Copy `/Users/crashy/Development/ghostty-rng/src/source.rs` to `src-tauri/src/source.rs` unchanged. Add `mod source;` to `lib.rs`.

- [ ] **Step 3: Write a behavior test for the simulator**

Append to `src-tauri/src/source.rs` inside a `#[cfg(test)]` module:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::Stats;
    use std::time::Duration;

    #[test]
    fn simulate_source_produces_bytes() {
        let src = Source::spawn(SourceKind::Simulate);
        std::thread::sleep(Duration::from_millis(120));
        let mut s = Stats::new(1 << 16);
        let n = drain_into(&src.data_rx, &mut s);
        assert!(n > 0, "simulator produced no bytes");
    }

    #[test]
    fn autodetect_does_not_panic() {
        let _ = autodetect(); // None is fine on CI with no device
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cd src-tauri && cargo test source::`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add src-tauri/src/source.rs src-tauri/src/lib.rs src-tauri/Cargo.toml
git commit -m "feat: port threaded entropy source + autodetect (source.rs)"
```

---

## Task 5: Serialization DTO (`NaN → null`)

**Files:**
- Create: `src-tauri/src/dto.rs`
- Modify: `src-tauri/src/lib.rs` (add `mod dto;`), `src-tauri/Cargo.toml` (ensure `serde` with `derive`)

**Interfaces:**
- Consumes: `crate::stats::{Snapshot, Metric, Verdict, Band, AnomalyEvent}`.
- Produces: `pub struct SnapshotDto` (all fields `Serialize`) and `impl From<&Snapshot> for SnapshotDto`. `Metric.value`/p-values become `Option<f64>` (None when NaN).

- [ ] **Step 1: Write the failing test**

Create `src-tauri/src/dto.rs`:
```rust
use serde::Serialize;
use crate::stats::{Snapshot, Metric, Verdict, Band, AnomalyEvent};

#[derive(Serialize)]
pub struct MetricDto { pub value: Option<f64>, pub quality: f64, pub verdict: &'static str }
#[derive(Serialize)]
pub struct AnomalyDto { pub at_secs: f64, pub peak_sigma: f64, pub band: &'static str, pub duration_secs: f64, pub ongoing: bool }

#[derive(Serialize)]
pub struct SnapshotDto {
    pub window_len: usize,
    pub total_bytes: u64,
    pub throughput_bps: f64,
    pub shannon: MetricDto,
    pub min_entropy: MetricDto,
    pub monobit: MetricDto,
    pub chi_square: MetricDto,
    pub serial_corr: MetricDto,
    pub monobit_p: Option<f64>,
    pub chi_square_p: Option<f64>,
    pub histogram: Vec<u32>,
    pub hist_expected: f64,
    pub recent: Vec<u8>,
    pub coherence_sigma: f64,
    pub coherence_band: &'static str,
    pub trial_count: u64,
    pub walk: Vec<(f64, f64)>,
    pub anomalies: Vec<AnomalyDto>,
    pub label: String,
    pub status: String,
}

fn nz(x: f64) -> Option<f64> { if x.is_finite() { Some(x) } else { None } }

fn verdict_str(v: Verdict) -> &'static str {
    match v { Verdict::Pass => "pass", Verdict::Warn => "warn", Verdict::Fail => "fail", Verdict::Warmup => "warmup" }
}
fn metric_dto(m: Metric) -> MetricDto {
    MetricDto { value: nz(m.value), quality: m.quality, verdict: verdict_str(m.verdict) }
}

impl SnapshotDto {
    pub fn from_snapshot(s: &Snapshot, label: String, status: String) -> Self {
        SnapshotDto {
            window_len: s.window_len,
            total_bytes: s.total_bytes,
            throughput_bps: s.throughput_bps,
            shannon: metric_dto(s.shannon),
            min_entropy: metric_dto(s.min_entropy),
            monobit: metric_dto(s.monobit),
            chi_square: metric_dto(s.chi_square),
            serial_corr: metric_dto(s.serial_corr),
            monobit_p: nz(s.monobit_p),
            chi_square_p: nz(s.chi_square_p),
            histogram: s.histogram.to_vec(),
            hist_expected: s.hist_expected,
            recent: s.recent.clone(),
            coherence_sigma: s.coherence_sigma,
            coherence_band: s.coherence_band.label(),
            trial_count: s.trial_count,
            walk: s.walk.clone(),
            anomalies: s.anomalies.iter().map(|a| AnomalyDto {
                at_secs: a.at_secs, peak_sigma: a.peak_sigma, band: a.band.label(),
                duration_secs: a.duration_secs, ongoing: a.ongoing,
            }).collect(),
            label, status,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::Stats;

    #[test]
    fn warmup_metric_serializes_nan_as_null() {
        // Few bytes → warming → shannon.value may be finite but chi_square is NaN at warmup.
        let mut s = Stats::new(1 << 16);
        s.push(&[1u8; 10]);
        let snap = s.snapshot();
        let dto = SnapshotDto::from_snapshot(&snap, "test".into(), "streaming".into());
        let json = serde_json::to_string(&dto).unwrap();
        assert!(!json.contains("NaN"), "JSON must not contain NaN: {json}");
        assert!(json.contains("\"warmup\""), "expected a warmup verdict");
    }
}
```
Add `mod dto;` to `lib.rs`. Ensure `serde = { version = "1", features = ["derive"] }` and `serde_json = "1"` are in `Cargo.toml` (`serde_json` may be dev-only — put it under `[dev-dependencies]` if unused elsewhere; it is used by Tauri at runtime so `[dependencies]` is fine).

- [ ] **Step 2: Run to verify fail-then-pass**

Run: `cd src-tauri && cargo test dto::`
Expected: PASS (write code and test together here since the DTO is pure mapping; if it fails, the most likely cause is a field-name mismatch with `Snapshot` — fix the DTO to match `stats.rs`).

- [ ] **Step 3: Commit**

```bash
git add src-tauri/src/dto.rs src-tauri/src/lib.rs src-tauri/Cargo.toml
git commit -m "feat: add SnapshotDto with NaN-safe serialization"
```

---

## Task 6: Engine + Tauri commands + emit loop

**Files:**
- Create: `src-tauri/src/engine.rs`
- Modify: `src-tauri/src/lib.rs`

**Interfaces:**
- Consumes: `Source`, `SourceKind`, `autodetect`, `drain_into`, `Stats`, `SnapshotDto`.
- Produces:
  - `pub enum ControlMsg { SetSource(SourceKind), Reset, SetWindow(usize), SetPaused(bool) }`
  - `pub struct Engine` with `new(initial: SourceKind) -> Engine`, `apply(&mut self, ControlMsg)`, `tick(&mut self) -> SnapshotDto`.
  - Tauri commands `start_source(req)`, `reset()`, `set_window(n)`, `set_paused(p)`, all sending a `ControlMsg` over a managed channel.

- [ ] **Step 1: Write the failing engine test**

Create `src-tauri/src/engine.rs`:
```rust
use crate::source::{Source, SourceKind, SourceStatus, autodetect, drain_into};
use crate::stats::Stats;
use crate::dto::SnapshotDto;

pub enum ControlMsg {
    SetSource(SourceKind),
    Reset,
    SetWindow(usize),
    SetPaused(bool),
}

pub struct Engine {
    source: Source,
    stats: Stats,
    paused: bool,
    status: String,
}

impl Engine {
    pub fn new(initial: SourceKind) -> Self {
        let source = Source::spawn(initial);
        Engine { source, stats: Stats::new(1 << 16), paused: false, status: "connecting".into() }
    }

    pub fn apply(&mut self, msg: ControlMsg) {
        match msg {
            ControlMsg::SetSource(kind) => { self.source = Source::spawn(kind); self.status = "connecting".into(); }
            ControlMsg::Reset => self.stats.reset(),
            ControlMsg::SetWindow(n) => self.stats.set_capacity(n),
            ControlMsg::SetPaused(p) => self.paused = p,
        }
    }

    pub fn tick(&mut self) -> SnapshotDto {
        while let Ok(st) = self.source.status_rx.try_recv() {
            self.status = match st {
                SourceStatus::Streaming => "streaming".into(),
                SourceStatus::Reconnecting(m) => format!("reconnecting: {m}"),
                SourceStatus::Error(m) => format!("error: {m}"),
            };
        }
        if !self.paused {
            drain_into(&self.source.data_rx, &mut self.stats);
            self.stats.tick_trials();
            let shannon = self.stats.snapshot().shannon.value;
            self.stats.tick_curve(shannon);
        }
        let snap = self.stats.snapshot();
        SnapshotDto::from_snapshot(&snap, self.source.label.clone(), self.status.clone())
    }
}

/// Resolve a UI request string into a concrete source kind.
pub fn resolve_kind(kind: &str, path: Option<String>, baud: u32) -> SourceKind {
    match kind {
        "bad" => SourceKind::BadRng,
        "serial" => match path.or_else(autodetect) {
            Some(p) => SourceKind::Serial { path: p, baud },
            None => SourceKind::Simulate,
        },
        "auto" => match autodetect() {
            Some(p) => SourceKind::Serial { path: p, baud },
            None => SourceKind::Simulate,
        },
        _ => SourceKind::Simulate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn engine_accumulates_bytes_over_ticks() {
        let mut e = Engine::new(SourceKind::Simulate);
        std::thread::sleep(Duration::from_millis(120));
        let a = e.tick();
        std::thread::sleep(Duration::from_millis(120));
        let b = e.tick();
        assert!(b.total_bytes >= a.total_bytes);
        assert!(b.total_bytes > 0);
    }

    #[test]
    fn pause_stops_accumulation() {
        let mut e = Engine::new(SourceKind::Simulate);
        std::thread::sleep(Duration::from_millis(120));
        let _ = e.tick();
        e.apply(ControlMsg::SetPaused(true));
        let a = e.tick();
        std::thread::sleep(Duration::from_millis(120));
        let b = e.tick();
        assert_eq!(a.total_bytes, b.total_bytes);
    }

    #[test]
    fn resolve_kind_falls_back_to_simulate() {
        assert!(matches!(resolve_kind("bad", None, 9600), SourceKind::BadRng));
        // "auto" with no device present resolves to Simulate.
        let _ = resolve_kind("auto", None, 9600);
    }
}
```
Add `mod engine;` to `lib.rs`.

- [ ] **Step 2: Run engine tests**

Run: `cd src-tauri && cargo test engine::`
Expected: PASS (all three).

- [ ] **Step 3: Wire commands + background emit loop in `lib.rs`**

```rust
use std::sync::mpsc::{channel, Sender};
use std::sync::Mutex;
use std::time::Duration;
use tauri::{Emitter, Manager};
use engine::{Engine, ControlMsg, resolve_kind};

mod math; mod stats; mod source; mod dto; mod engine;

struct Control(Mutex<Sender<ControlMsg>>);

#[tauri::command]
fn start_source(kind: String, path: Option<String>, baud: Option<u32>, ctrl: tauri::State<Control>) {
    let k = resolve_kind(&kind, path, baud.unwrap_or(9600));
    let _ = ctrl.0.lock().unwrap().send(ControlMsg::SetSource(k));
}
#[tauri::command]
fn reset(ctrl: tauri::State<Control>) { let _ = ctrl.0.lock().unwrap().send(ControlMsg::Reset); }
#[tauri::command]
fn set_window(n: usize, ctrl: tauri::State<Control>) { let _ = ctrl.0.lock().unwrap().send(ControlMsg::SetWindow(n)); }
#[tauri::command]
fn set_paused(paused: bool, ctrl: tauri::State<Control>) { let _ = ctrl.0.lock().unwrap().send(ControlMsg::SetPaused(paused)); }

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let (tx, rx) = channel::<ControlMsg>();
    tauri::Builder::default()
        .manage(Control(Mutex::new(tx)))
        .invoke_handler(tauri::generate_handler![start_source, reset, set_window, set_paused])
        .setup(move |app| {
            let handle = app.handle().clone();
            std::thread::spawn(move || {
                // Start on auto-detected device, else simulate.
                let mut engine = Engine::new(resolve_kind("auto", None, 9600));
                loop {
                    while let Ok(msg) = rx.try_recv() { engine.apply(msg); }
                    let dto = engine.tick();
                    let _ = handle.emit("snapshot", &dto);
                    std::thread::sleep(Duration::from_millis(50)); // ~20 Hz
                }
            });
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```
Delete the placeholder `ping` command. Ensure `src-tauri/src/main.rs` calls `app_lib::run()` (the template wires this; keep it).

- [ ] **Step 4: Build**

Run: `cd src-tauri && cargo build`
Expected: compiles. (Run-time emit is verified in Task 8.)

- [ ] **Step 5: Commit**

```bash
git add src-tauri/src/engine.rs src-tauri/src/lib.rs
git commit -m "feat: engine tick loop + source-control commands + snapshot emit"
```

---

## Task 7: Frontend rendering

**Files:**
- Create/replace: `src/index.html`, `src/main.ts`, `src/render.ts`, `src/render.test.ts`, `src/style.css`
- Modify: `package.json` (add `vitest`), `package.json` test script

**Interfaces:**
- Consumes: `snapshot` events with the `SnapshotDto` shape (see Task 5) via `@tauri-apps/api/event`; commands via `@tauri-apps/api/core` `invoke`.
- Produces: pure helpers in `render.ts` — `envelopePoints(maxK: number, z: number): [number,number][]`, `bandColor(band: string): string`, `verdictColor(v: string): string`.

- [ ] **Step 1: Write failing vitest for the pure helpers**

Create `src/render.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import { envelopePoints, bandColor, verdictColor } from "./render";

describe("render helpers", () => {
  it("envelope is the parabola z*sqrt(k)", () => {
    const pts = envelopePoints(100, 1.95996398);
    const [k, c] = pts[pts.length - 1];
    expect(k).toBe(100);
    expect(c).toBeCloseTo(1.95996398 * Math.sqrt(100), 6);
  });
  it("maps bands and verdicts to palette colors", () => {
    expect(bandColor("99.9%")).toBe("#ff4d4d");
    expect(bandColor("in-band")).toBe("#6cf0d0");
    expect(verdictColor("pass")).toBe("#6cf0d0");
    expect(verdictColor("fail")).toBe("#ff4d4d");
  });
});
```
Add `vitest` to devDependencies and a `"test": "vitest run"` script in `package.json`.

- [ ] **Step 2: Run to verify failure**

Run: `npm run test`
Expected: FAIL — module `./render` has no such exports.

- [ ] **Step 3: Implement `render.ts`**

```ts
const TURQUOISE = "#6cf0d0";
const GOLD = "#f2c14e";
const ORANGE = "#ff9d3c";
const RED = "#ff4d4d";

export function bandColor(band: string): string {
  switch (band) {
    case "95%": return GOLD;
    case "99%": return ORANGE;
    case "99.9%": return RED;
    default: return TURQUOISE;
  }
}
export function verdictColor(v: string): string {
  switch (v) {
    case "pass": return TURQUOISE;
    case "warn": return GOLD;
    case "fail": return RED;
    default: return "#888";
  }
}
export function envelopePoints(maxK: number, z: number): [number, number][] {
  const pts: [number, number][] = [];
  for (let k = 1; k <= maxK; k++) pts.push([k, z * Math.sqrt(k)]);
  return pts;
}

// --- Canvas painters (not unit-tested; verified manually in Task 8) ---
type Dto = any;

export function drawCoherence(ctx: CanvasRenderingContext2D, dto: Dto, w: number, h: number) {
  ctx.clearRect(0, 0, w, h);
  const walk: [number, number][] = dto.walk;
  const maxK = Math.max(50, dto.trial_count || 50);
  const yMax = 3.29052673 * Math.sqrt(maxK) * 1.2 || 10;
  const sx = (k: number) => (k / maxK) * w;
  const sy = (c: number) => h / 2 - (c / yMax) * (h / 2);
  // envelopes
  for (const [z, color] of [[1.95996398, GOLD], [2.5758293, ORANGE], [3.29052673, RED]] as [number,string][]) {
    for (const sign of [1, -1]) {
      ctx.strokeStyle = color; ctx.globalAlpha = 0.5; ctx.beginPath();
      envelopePoints(maxK, z).forEach(([k, c], i) => {
        const x = sx(k), y = sy(sign * c);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;
  // the walk
  ctx.strokeStyle = bandColor(dto.coherence_band); ctx.lineWidth = 1.5; ctx.beginPath();
  walk.forEach(([k, c], i) => { const x = sx(k), y = sy(c); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
  ctx.stroke();
}

export function drawHistogram(ctx: CanvasRenderingContext2D, dto: Dto, w: number, h: number) {
  ctx.clearRect(0, 0, w, h);
  const hist: number[] = dto.histogram;
  const max = Math.max(1, ...hist);
  const bw = w / 256;
  ctx.fillStyle = TURQUOISE;
  hist.forEach((c, i) => { const bh = (c / max) * h; ctx.fillRect(i * bw, h - bh, Math.max(1, bw - 0.5), bh); });
}

export function drawBitstream(ctx: CanvasRenderingContext2D, dto: Dto, w: number, h: number) {
  ctx.clearRect(0, 0, w, h);
  const bytes: number[] = dto.recent;
  const cols = Math.floor(w / 6);
  bytes.slice(-cols * 8).forEach((b, i) => {
    const x = (i % cols) * 6, y = Math.floor(i / cols) * 6;
    ctx.fillStyle = `hsl(${(b / 255) * 360}, 70%, 55%)`;
    ctx.fillRect(x, y, 5, 5);
  });
}
```

- [ ] **Step 4: Run to verify pass**

Run: `npm run test`
Expected: PASS.

- [ ] **Step 5: Build the layout (`index.html`, `style.css`) and wire events (`main.ts`)**

`src/index.html` body:
```html
<main>
  <header>
    <span id="label">connecting…</span>
    <span id="status"></span>
    <span id="rate"></span>
    <select id="source">
      <option value="auto">auto-detect</option>
      <option value="simulate">simulate</option>
      <option value="bad">bad rng</option>
    </select>
    <button id="pause">pause</button>
    <button id="reset">reset</button>
  </header>
  <section class="grid">
    <div class="panel"><h2>Coherence</h2><canvas id="coherence"></canvas></div>
    <div class="panel"><h2>Randomness</h2><ul id="metrics"></ul></div>
    <div class="panel"><h2>Byte distribution</h2><canvas id="histogram"></canvas></div>
    <div class="panel"><h2>Bitstream</h2><canvas id="bitstream"></canvas></div>
  </section>
</main>
```

`src/main.ts`:
```ts
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { drawCoherence, drawHistogram, drawBitstream, verdictColor } from "./render";
import "./style.css";

function canvas(id: string): [CanvasRenderingContext2D, number, number] {
  const el = document.getElementById(id) as HTMLCanvasElement;
  const w = el.clientWidth, h = el.clientHeight;
  el.width = w; el.height = h;
  return [el.getContext("2d")!, w, h];
}

let paused = false;

listen<any>("snapshot", (e) => {
  const dto = e.payload;
  document.getElementById("label")!.textContent = dto.label;
  document.getElementById("status")!.textContent = dto.status;
  document.getElementById("rate")!.textContent =
    `${(dto.throughput_bps / 1024).toFixed(0)} KiB/s · ${(dto.total_bytes / 1048576).toFixed(1)} MiB`;

  const [cc, cw, ch] = canvas("coherence"); drawCoherence(cc, dto, cw, ch);
  const [hc, hw, hh] = canvas("histogram"); drawHistogram(hc, dto, hw, hh);
  const [bc, bw, bh] = canvas("bitstream"); drawBitstream(bc, dto, bw, bh);

  const rows = [
    ["Shannon", dto.shannon], ["Min-entropy", dto.min_entropy], ["Monobit", dto.monobit],
    ["Chi-square", dto.chi_square], ["Serial corr", dto.serial_corr],
  ] as [string, any][];
  document.getElementById("metrics")!.innerHTML = rows.map(([name, m]) =>
    `<li style="color:${verdictColor(m.verdict)}">${name}: ${m.value == null ? "…" : m.value.toFixed(4)} <b>${m.verdict.toUpperCase()}</b></li>`
  ).join("");
});

document.getElementById("source")!.addEventListener("change", (ev) =>
  invoke("start_source", { kind: (ev.target as HTMLSelectElement).value }));
document.getElementById("reset")!.addEventListener("click", () => invoke("reset"));
document.getElementById("pause")!.addEventListener("click", () => {
  paused = !paused; invoke("set_paused", { paused });
  document.getElementById("pause")!.textContent = paused ? "resume" : "pause";
});
```

`src/style.css`:
```css
:root { --bg:#0a0e0d; --turq:#6cf0d0; --gold:#f2c14e; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--turq); font:14px ui-monospace, monospace; }
main { height:100vh; display:flex; flex-direction:column; }
header { display:flex; gap:12px; align-items:center; padding:8px 12px; border-bottom:1px solid #1d2422; }
header select, header button { background:#111614; color:var(--turq); border:1px solid #243029; padding:4px 8px; font:inherit; }
.grid { flex:1; display:grid; grid-template-columns:1fr 1fr; grid-template-rows:1fr 1fr; gap:10px; padding:10px; }
.panel { border:1px solid #1d2422; border-radius:6px; padding:8px; display:flex; flex-direction:column; min-height:0; }
.panel h2 { margin:0 0 6px; font-size:12px; color:var(--gold); font-weight:600; }
.panel canvas { flex:1; width:100%; min-height:0; }
#metrics { list-style:none; margin:0; padding:0; }
#metrics li { padding:3px 0; }
```

- [ ] **Step 6: Type-check and test**

Run: `npm run test && npx tsc --noEmit`
Expected: tests PASS, no type errors. (Tauri API import resolution requires `@tauri-apps/api` — installed by the scaffold.)

- [ ] **Step 7: Commit**

```bash
git add src/ package.json package-lock.json
git commit -m "feat: observatory frontend — coherence/histogram/bitstream/metrics"
```

---

## Task 8: End-to-end smoke + README

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: the full app.

- [ ] **Step 1: Run the app against the simulator**

Run: `npm run tauri dev`
Expected: a window opens; within ~2 s the metrics list shows PASS verdicts, the histogram is roughly flat, the bitstream ribbon animates, and the coherence walk wanders inside the gold envelope. Switch the source dropdown to **bad rng** and confirm verdicts turn red and the coherence walk bolts out of the band. Switch to **auto-detect**; if a TrueRNG is plugged in, the label shows the `/dev/cu.usbmodem*` path.

- [ ] **Step 2: Capture the result**

Confirm (and note in the commit) the three states observed: simulate→green, bad→red, auto→device label (or simulate fallback if no device).

- [ ] **Step 3: Write `README.md`**

Document: what it is (entropy observatory), requirements (Rust, Node, optional TrueRNG), `npm install`, `npm run tauri dev`, the source switch, and a one-line pointer that diffusion-driven text generation is Plans 2–3. Credit the ported core from ghostty-rng.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: add README and verify end-to-end observatory"
```

---

## Self-Review

**Spec coverage (Plan 1 scope):**
- Tauri desktop app ✓ (Task 1) · TrueRNG reader + simulate/bad + autodetect ✓ (Task 4) · entropy/NIST stats ✓ (Task 3) · coherence walk + envelopes + anomalies ✓ (Tasks 3, 7) · bitstream + histogram ✓ (Task 7) · palette ✓ (Task 7) · auto-detect→simulate fallback ✓ (Task 6) · source switch UI ✓ (Task 7) · ported tests preserved ✓ (Tasks 2–4). Diffusion (`TrngRng`, candle, generation UI) is intentionally deferred to Plans 2–3.

**Placeholder scan:** No "TBD"/"add error handling"-style steps; every code step shows complete code, every test step shows the command and expected result.

**Type consistency:** `SnapshotDto` field names mirror `stats::Snapshot` exactly (verified against `stats.rs` lines 96–129). `metric.value` is `Option<f64>` in the DTO and the frontend checks `m.value == null`. Engine `ControlMsg` variants match the command senders in `lib.rs`. `resolve_kind` is the single source-construction path used by both the command and the setup loop.

**Risk note:** Task 1's non-interactive scaffold into a non-empty dir is the one fragile step — the fallback (scaffold in temp, move files) is spelled out.
```
