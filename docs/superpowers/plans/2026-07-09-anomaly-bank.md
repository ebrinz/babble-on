# Anomaly Bank + Plaid Quality Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bytes captured while the coherence walk is outside its 95% envelope are banked (64 KiB FIFO) and destructively spent to seed each generation's initial diffusion latent, with per-generation provenance ("seed: 72% anomaly bank — +3.2σ @ 14:32") and a live fill meter — plus a documented sampler-parity audit and a new "ultra" quality preset.

**Architecture:** The bank is a new `AnomalyBank` module owned by the single-threaded stats `Engine` loop (`src-tauri/src/engine.rs`) — no mutex; all access flows through the existing `ControlMsg` channel. The existing `ControlMsg::GetEntropy` is replaced by `GetSeed`, which withdraws bank-first and tops up from `Stats::fresh_entropy`. Provenance rides the reply into `run_generation`, which emits a new `seeded` event on the existing `diffusion` event channel. Frontend adds a gold fill meter in the anomaly-log panel and a stamp under the generation output.

**Tech Stack:** Rust (Tauri 2, candle), TypeScript + Vite frontend, vitest, cargo test.

**Spec:** `docs/superpowers/specs/2026-07-09-anomaly-bank-design.md`

## Global Constraints

- Bank capacity: 64 KiB (`64 * 1024`), drop-oldest.
- Bank seeds the **initial latent only**; spending is destructive FIFO; shortfall tops up from the live stream silently (never blocks, never errors).
- Deposit trigger: `snapshot.coherence_band != Band::Inside` (beyond the 95% envelope), same classification as the anomaly log.
- `reset` clears the bank along with stats. Bank is session-scoped (no persistence).
- Palette: gold `#f2c14e` is `var(--gold)` in `src/style.css` — use the CSS var.
- Existing behavior to preserve, not "fix": per-step sampler noise is software PRNG (`Tensor::randn` in `diffusion-rs/src/lib.rs`); only the initial latent is entropy-fed. This plan does not change that.
- Rust commands run from `src-tauri/` (`cargo test`, `cargo build`); the `diffusion-rs` crate has its own manifest (`cargo test --manifest-path ../diffusion-rs/Cargo.toml` from `src-tauri/`, or run from `diffusion-rs/`). Frontend: `npm run test`, `npm run build` from the repo root.
- Commit after every task.

---

### Task 1: `AnomalyBank` module

**Files:**
- Create: `src-tauri/src/bank.rs`
- Modify: `src-tauri/src/lib.rs:8-13` (add `mod bank;`)
- Test: inline `#[cfg(test)]` in `src-tauri/src/bank.rs`

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces (Task 2+ relies on these exact signatures):
  - `pub const BANK_CAPACITY: usize = 64 * 1024;`
  - `pub struct ProvenanceTag { pub at_secs: f64, pub peak_sigma: f64, pub band: &'static str }` (derives `Clone, Copy, Debug, serde::Serialize`)
  - `AnomalyBank::new(capacity: usize) -> Self`
  - `AnomalyBank::deposit(&mut self, bytes: &[u8], tag: ProvenanceTag)`
  - `AnomalyBank::withdraw(&mut self, n: usize) -> (Vec<u8>, Vec<ProvenanceTag>)`
  - `AnomalyBank::fill(&self) -> (usize, usize)` (current, capacity)
  - `AnomalyBank::clear(&mut self)`

- [ ] **Step 1: Write the failing tests**

Create `src-tauri/src/bank.rs` containing only the test module for now (plus the imports the tests need — the tests reference types that don't exist yet, so this fails to compile, which is the failing state for a new module):

```rust
//! Anomaly bank: raw entropy bytes harvested while the coherence walk is
//! outside its 95% envelope, spent destructively to seed diffusion latents.

use std::collections::VecDeque;

use serde::Serialize;

#[cfg(test)]
mod tests {
    use super::*;

    fn tag(at: f64, sigma: f64) -> ProvenanceTag {
        ProvenanceTag { at_secs: at, peak_sigma: sigma, band: "95%" }
    }

    #[test]
    fn withdraw_is_fifo_and_destructive() {
        let mut b = AnomalyBank::new(BANK_CAPACITY);
        b.deposit(&[1, 2, 3], tag(1.0, 2.1));
        b.deposit(&[4, 5], tag(9.0, -2.6));
        let (bytes, tags) = b.withdraw(4);
        assert_eq!(bytes, vec![1, 2, 3, 4]);
        assert_eq!(tags.len(), 2);
        assert_eq!(tags[0].at_secs, 1.0);
        assert_eq!(tags[1].at_secs, 9.0);
        assert_eq!(b.fill().0, 1);
        let (bytes, tags) = b.withdraw(99);
        assert_eq!(bytes, vec![5]);
        assert_eq!(tags.len(), 1);
        assert_eq!(b.fill().0, 0);
    }

    #[test]
    fn same_event_deposits_merge_and_update_tag() {
        let mut b = AnomalyBank::new(BANK_CAPACITY);
        // Same event (same at_secs) deposited across two ticks; peak grows.
        b.deposit(&[1, 2], tag(1.0, 2.1));
        b.deposit(&[3, 4], tag(1.0, 3.4));
        let (bytes, tags) = b.withdraw(4);
        assert_eq!(bytes, vec![1, 2, 3, 4]);
        assert_eq!(tags.len(), 1, "one event → one tag");
        assert_eq!(tags[0].peak_sigma, 3.4, "latest tag wins (running peak)");
    }

    #[test]
    fn capacity_evicts_oldest() {
        let mut b = AnomalyBank::new(4);
        b.deposit(&[1, 2, 3], tag(1.0, 2.1));
        b.deposit(&[4, 5, 6], tag(9.0, 2.9));
        assert_eq!(b.fill(), (4, 4));
        let (bytes, tags) = b.withdraw(4);
        assert_eq!(bytes, vec![3, 4, 5, 6], "oldest bytes dropped");
        assert_eq!(tags.len(), 2, "partially-evicted run keeps its tag");
    }

    #[test]
    fn empty_and_clear() {
        let mut b = AnomalyBank::new(BANK_CAPACITY);
        let (bytes, tags) = b.withdraw(8);
        assert!(bytes.is_empty() && tags.is_empty());
        b.deposit(&[7; 10], tag(1.0, 2.1));
        b.clear();
        assert_eq!(b.fill().0, 0);
        assert!(b.withdraw(1).0.is_empty());
    }

    #[test]
    fn empty_deposit_is_a_noop() {
        let mut b = AnomalyBank::new(BANK_CAPACITY);
        b.deposit(&[], tag(1.0, 2.1));
        assert_eq!(b.fill().0, 0);
        assert!(b.withdraw(1).1.is_empty(), "no phantom tag");
    }
}
```

Also add `mod bank;` to the module list in `src-tauri/src/lib.rs` (after `mod stats;`).

- [ ] **Step 2: Run tests to verify they fail**

Run (from `src-tauri/`): `cargo test bank`
Expected: COMPILE ERROR — `cannot find type ProvenanceTag` / `AnomalyBank`.

- [ ] **Step 3: Write the implementation**

Insert above the test module in `src-tauri/src/bank.rs`:

```rust
/// Bank capacity in bytes. A full 256-token latent needs 16 KiB
/// (seq_len 256 × embed_dim 16 × 4 bytes/gaussian), so this holds ~4 seeds.
pub const BANK_CAPACITY: usize = 64 * 1024;

/// Provenance of a banked run: the anomaly event whose bytes these are.
/// `at_secs`/`peak_sigma`/`band` mirror `stats::AnomalyEvent` (band as label).
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ProvenanceTag {
    pub at_secs: f64,
    pub peak_sigma: f64,
    pub band: &'static str,
}

/// Bounded FIFO byte pool with per-event provenance. Deposits append; when
/// over capacity the oldest bytes fall off. Withdrawals are destructive and
/// return the tags of every event represented in the bytes handed out.
pub struct AnomalyBank {
    buf: VecDeque<u8>,
    /// Per-event runs, FIFO: (bytes remaining in run, tag). Invariant:
    /// the run lengths always sum to `buf.len()`.
    runs: VecDeque<(usize, ProvenanceTag)>,
    capacity: usize,
}

impl AnomalyBank {
    pub fn new(capacity: usize) -> Self {
        AnomalyBank { buf: VecDeque::new(), runs: VecDeque::new(), capacity }
    }

    pub fn deposit(&mut self, bytes: &[u8], tag: ProvenanceTag) {
        if bytes.is_empty() {
            return;
        }
        self.buf.extend(bytes.iter().copied());
        match self.runs.back_mut() {
            // Same ongoing event across ticks: extend the run; the latest tag
            // carries the event's running peak/strongest band, so it wins.
            Some((len, last)) if last.at_secs == tag.at_secs => {
                *len += bytes.len();
                *last = tag;
            }
            _ => self.runs.push_back((bytes.len(), tag)),
        }
        if self.buf.len() > self.capacity {
            let mut excess = self.buf.len() - self.capacity;
            self.buf.drain(..excess);
            while excess > 0 {
                let (len, _) = self.runs.front_mut().expect("runs cover buf");
                if *len <= excess {
                    excess -= *len;
                    self.runs.pop_front();
                } else {
                    *len -= excess;
                    excess = 0;
                }
            }
        }
    }

    /// Destructive FIFO read of up to `n` bytes + the tags they came from.
    pub fn withdraw(&mut self, n: usize) -> (Vec<u8>, Vec<ProvenanceTag>) {
        let take = n.min(self.buf.len());
        let bytes: Vec<u8> = self.buf.drain(..take).collect();
        let mut tags = Vec::new();
        let mut left = take;
        while left > 0 {
            let (len, tag) = self.runs.front_mut().expect("runs cover buf");
            tags.push(*tag);
            if *len <= left {
                left -= *len;
                self.runs.pop_front();
            } else {
                *len -= left;
                left = 0;
            }
        }
        (bytes, tags)
    }

    /// (current fill, capacity) in bytes — for the UI meter.
    pub fn fill(&self) -> (usize, usize) {
        (self.buf.len(), self.capacity)
    }

    pub fn clear(&mut self) {
        self.buf.clear();
        self.runs.clear();
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run (from `src-tauri/`): `cargo test bank`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src-tauri/src/bank.rs src-tauri/src/lib.rs
git commit -m "feat(bank): AnomalyBank — FIFO byte pool with event provenance"
```

---

### Task 2: Engine wiring — deposits, GetSeed, reset, DTO fill fields

**Files:**
- Modify: `src-tauri/src/engine.rs` (bank ownership, deposit on tick, `GetSeed`, reset)
- Modify: `src-tauri/src/dto.rs:9-31,42-68` (add `bank_fill`/`bank_capacity`)
- Modify: `src-tauri/src/lib.rs:66-74` (switch `generate` to `GetSeed` so the crate still compiles — provenance is *used* in Task 3)
- Test: inline tests in `src-tauri/src/engine.rs`

**Interfaces:**
- Consumes: `AnomalyBank`, `ProvenanceTag`, `BANK_CAPACITY` from Task 1; `Stats::fresh_entropy`, `Band`, `AnomalyEvent` (existing).
- Produces (Task 3 relies on):
  - `pub struct SeedReply { pub bytes: Vec<u8>, pub bank_fraction: f64, pub tags: Vec<ProvenanceTag> }` in `engine.rs`
  - `ControlMsg::GetSeed(usize, std::sync::mpsc::Sender<SeedReply>)` — **replaces** `ControlMsg::GetEntropy` (only caller is `lib.rs::generate`)
  - `pub fn deposit_if_anomalous(bank: &mut AnomalyBank, bytes: &[u8], band: Band, sigma: f64, event: Option<&AnomalyEvent>)`
  - `SnapshotDto` gains `pub bank_fill: usize, pub bank_capacity: usize`

- [ ] **Step 1: Write the failing tests**

Add to the existing `mod tests` in `src-tauri/src/engine.rs`:

```rust
    use crate::bank::{AnomalyBank, ProvenanceTag, BANK_CAPACITY};
    use crate::stats::{AnomalyEvent, Band};

    #[test]
    fn deposits_only_when_out_of_band() {
        let mut bank = AnomalyBank::new(BANK_CAPACITY);
        deposit_if_anomalous(&mut bank, &[1, 2, 3], Band::Inside, 0.4, None);
        assert_eq!(bank.fill().0, 0, "in-band bytes are not banked");

        let ev = AnomalyEvent { at_secs: 12.5, peak_sigma: 3.1, band: Band::P99, duration_secs: 2.0, ongoing: true };
        deposit_if_anomalous(&mut bank, &[1, 2, 3], Band::P99, 3.0, Some(&ev));
        assert_eq!(bank.fill().0, 3);
        let (_, tags) = bank.withdraw(3);
        assert_eq!(tags[0].at_secs, 12.5);
        assert_eq!(tags[0].band, "99%");
    }

    #[test]
    fn get_seed_prefers_bank_and_tops_up_live() {
        let mut e = Engine::new(SourceKind::Simulate);
        std::thread::sleep(Duration::from_millis(120));
        e.tick(); // accumulate live bytes for the top-up
        e.bank.deposit(&[9, 9, 9, 9], ProvenanceTag { at_secs: 1.0, peak_sigma: 2.5, band: "95%" });

        let (tx, rx) = std::sync::mpsc::channel();
        e.apply(ControlMsg::GetSeed(8, tx));
        let reply = rx.recv().unwrap();
        assert_eq!(&reply.bytes[..4], &[9, 9, 9, 9], "bank bytes come first");
        assert_eq!(reply.bytes.len(), 8, "topped up from the live stream");
        assert!((reply.bank_fraction - 0.5).abs() < 1e-9);
        assert_eq!(reply.tags.len(), 1);
        assert_eq!(e.bank.fill().0, 0, "spend is destructive");
    }

    #[test]
    fn reset_clears_the_bank_and_snapshot_reports_fill() {
        let mut e = Engine::new(SourceKind::Simulate);
        e.bank.deposit(&[7; 100], ProvenanceTag { at_secs: 1.0, peak_sigma: 2.5, band: "95%" });
        let dto = e.tick();
        assert_eq!(dto.bank_fill, 100);
        assert_eq!(dto.bank_capacity, BANK_CAPACITY);
        e.apply(ControlMsg::Reset);
        assert_eq!(e.bank.fill().0, 0);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run (from `src-tauri/`): `cargo test engine`
Expected: COMPILE ERROR — `deposit_if_anomalous` / `GetSeed` / `bank` field not found.

- [ ] **Step 3: Implement the engine changes**

In `src-tauri/src/engine.rs`:

Replace the imports and `ControlMsg` (lines 1–13) with:

```rust
use crate::source::{Source, SourceKind, SourceStatus, autodetect};
use crate::stats::{AnomalyEvent, Band, Stats};
use crate::bank::{AnomalyBank, ProvenanceTag, BANK_CAPACITY};
use crate::dto::SnapshotDto;

/// Reply to a `GetSeed` request: seed bytes drawn bank-first, plus how much
/// of the seed came from the bank and which anomaly events it spent.
pub struct SeedReply {
    pub bytes: Vec<u8>,
    pub bank_fraction: f64,
    pub tags: Vec<ProvenanceTag>,
}

pub enum ControlMsg {
    SetSource(SourceKind),
    Reset,
    SetWindow(usize),
    SetPaused(bool),
    /// Request `n` seed bytes for a generation, replied over the sender.
    /// Drawn destructively from the anomaly bank first, topped up with the
    /// most recent live-stream bytes.
    GetSeed(usize, std::sync::mpsc::Sender<SeedReply>),
}
```

(Note: `drain_into` is no longer imported here — it stays in `source.rs`, still used by its own tests. Since test-only usage doesn't count for the dead-code lint, if `cargo build` now warns about it, annotate it `#[allow(dead_code)]` rather than deleting it.)

Add the bank to `Engine` and its constructor:

```rust
pub struct Engine {
    source: Source,
    stats: Stats,
    bank: AnomalyBank,
    paused: bool,
    status: String,
}
```

In `Engine::new`, add `bank: AnomalyBank::new(BANK_CAPACITY),` to the struct literal.

In `Engine::apply`, replace the `Reset` and `GetEntropy` arms:

```rust
            ControlMsg::Reset => {
                self.stats.reset();
                self.bank.clear();
            }
            ControlMsg::GetSeed(n, reply) => {
                let (mut bytes, tags) = self.bank.withdraw(n);
                let banked = bytes.len();
                if banked < n {
                    bytes.extend_from_slice(&self.stats.fresh_entropy(n - banked));
                }
                let bank_fraction = if n == 0 { 0.0 } else { banked as f64 / n as f64 };
                let _ = reply.send(SeedReply { bytes, bank_fraction, tags });
            }
```

Replace `Engine::tick` with (the drain now collects bytes so out-of-band ticks can bank them; the snapshot — which carries the band — is computed before depositing):

```rust
    pub fn tick(&mut self) -> SnapshotDto {
        while let Ok(st) = self.source.status_rx.try_recv() {
            self.status = match st {
                SourceStatus::Streaming => "streaming".into(),
                SourceStatus::Reconnecting(m) => format!("reconnecting: {m}"),
                SourceStatus::Error(m) => format!("error: {m}"),
            };
        }
        let mut drained: Vec<u8> = Vec::new();
        if !self.paused {
            while let Ok(chunk) = self.source.data_rx.try_recv() {
                drained.extend_from_slice(&chunk);
            }
            self.stats.push(&drained);
            self.stats.tick_trials();
        } else {
            // Keep the unbounded channel from growing while paused, without
            // feeding the bytes into the (frozen) stats window.
            while self.source.data_rx.try_recv().is_ok() {}
        }
        let snap = self.stats.snapshot();
        if !self.paused {
            self.stats.tick_curve(snap.shannon.value);
            deposit_if_anomalous(
                &mut self.bank,
                &drained,
                snap.coherence_band,
                snap.coherence_sigma,
                snap.anomalies.first().filter(|a| a.ongoing),
            );
        }
        let mut dto = SnapshotDto::from_snapshot(&snap, self.source.label.clone(), self.status.clone());
        let (fill, cap) = self.bank.fill();
        dto.bank_fill = fill;
        dto.bank_capacity = cap;
        dto
    }
```

Add the free function after `resolve_kind`:

```rust
/// Bank a tick's drained bytes when the coherence walk is outside the 95%
/// envelope, tagged with the ongoing anomaly event. The event's `peak_sigma`
/// and `band` are running values, so the latest deposit's tag supersedes.
pub fn deposit_if_anomalous(
    bank: &mut AnomalyBank,
    bytes: &[u8],
    band: Band,
    sigma: f64,
    event: Option<&AnomalyEvent>,
) {
    if band == Band::Inside || bytes.is_empty() {
        return;
    }
    let tag = match event {
        Some(a) => ProvenanceTag { at_secs: a.at_secs, peak_sigma: a.peak_sigma, band: a.band.label() },
        // Excursion frame before the event is logged: tag with live values.
        None => ProvenanceTag { at_secs: -1.0, peak_sigma: sigma, band: band.label() },
    };
    bank.deposit(bytes, tag);
}
```

In `src-tauri/src/dto.rs`, add two fields to the `SnapshotDto` struct (after `pub status: String,`):

```rust
    /// Anomaly-bank fill state (bytes), stamped by the engine each tick.
    pub bank_fill: usize,
    pub bank_capacity: usize,
```

and initialize both to `0` at the end of the struct literal in `from_snapshot`:

```rust
            label, status,
            bank_fill: 0,
            bank_capacity: 0,
```

In `src-tauri/src/lib.rs`, update `generate` so the crate compiles against the new message (full provenance use comes in Task 3). Replace lines 66–74 (`// Pull the most recent...` through `let entropy = ...`) with:

```rust
    // Seed bytes for the initial latent: anomaly bank first, live-stream top-up.
    let need = n_samples * seq_len * EMBED_DIM * 4;
    let (etx, erx) = channel::<SeedReply>();
    let _ = ctrl
        .0
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .send(ControlMsg::GetSeed(need, etx));
    let seed = erx.recv_timeout(Duration::from_secs(2)).unwrap_or(SeedReply {
        bytes: Vec::new(),
        bank_fraction: 0.0,
        tags: Vec::new(),
    });
    let entropy = seed.bytes;
```

and change the engine import to `use engine::{Engine, ControlMsg, SeedReply, resolve_kind};`.

- [ ] **Step 4: Run the full backend suite**

Run (from `src-tauri/`): `cargo test`
Expected: all pass (existing 15+ plus the 5 bank tests and 3 new engine tests). If `fresh_entropy` returns fewer than requested in `get_seed_prefers_bank_and_tops_up_live` (warmup), the sleep already covers it — the simulator produces KiB within 120 ms.

- [ ] **Step 5: Commit**

```bash
git add src-tauri/src/engine.rs src-tauri/src/dto.rs src-tauri/src/lib.rs
git commit -m "feat(engine): bank out-of-band bytes on tick; GetSeed draws bank-first"
```

---

### Task 3: Seed provenance through the generation events

**Files:**
- Modify: `src-tauri/src/lib.rs:76-83` (pass the whole `SeedReply` down)
- Modify: `src-tauri/src/diffusion.rs:49-86` (`run_generation` takes `SeedReply`, emits a `seeded` event)

**Interfaces:**
- Consumes: `SeedReply` (Task 2), `ProvenanceTag: Serialize` (Task 1).
- Produces (Task 4 relies on): a `diffusion` event `{"type": "seeded", "bank_fraction": f64, "tags": [{"at_secs", "peak_sigma", "band"}...]}` emitted once per generation, after model load and before sampling begins.

- [ ] **Step 1: Change `run_generation` to take the seed**

In `src-tauri/src/diffusion.rs`, add the import `use crate::engine::SeedReply;`, change the last parameter of `run_generation` from `entropy: Vec<u8>` to `seed: SeedReply`, and replace the body between the engine load block and `let t0 = ...`:

```rust
    let eng = guard.as_ref().unwrap();

    let _ = app.emit(
        "diffusion",
        json!({"type": "seeded", "bank_fraction": seed.bank_fraction, "tags": seed.tags}),
    );

    let need = seq_len * EMBED_DIM * 4;
    let entropy = seed.bytes;
    let entropy_opt = if entropy.len() >= need { Some(&entropy[..need]) } else { None };
    let preview_every = (steps / 24).max(1); // dense cadence for a smooth "boil"
```

- [ ] **Step 2: Pass the seed from the command**

In `src-tauri/src/lib.rs::generate`, delete the `let entropy = seed.bytes;` line added in Task 2 and pass `seed` directly:

```rust
        if let Err(e) = run_generation(&app2, &diff2, steps, seq_len, n_samples, temperature, noise_scale, ddim, prompt, seed) {
```

- [ ] **Step 3: Verify it builds and tests pass**

Run (from `src-tauri/`): `cargo build && cargo test`
Expected: clean build, all tests pass.

- [ ] **Step 4: Commit**

```bash
git add src-tauri/src/lib.rs src-tauri/src/diffusion.rs
git commit -m "feat(diffusion): emit seeded event with bank fraction + anomaly provenance"
```

---

### Task 4: Frontend — bank meter and seed stamp

**Files:**
- Modify: `index.html:27` (meter in the anomaly-log panel), `index.html:48-49` (stamp element)
- Modify: `src/render.ts` (add `seedStamp` pure helper)
- Modify: `src/main.ts` (meter update on `snapshot`; `seeded` case; clear stamp on new run)
- Modify: `src/style.css` (meter + stamp styles)
- Test: `src/render.test.ts`

**Interfaces:**
- Consumes: `dto.bank_fill` / `dto.bank_capacity` on `snapshot` events (Task 2); the `seeded` diffusion event (Task 3).
- Produces: `export function seedStamp(bankFraction: number, tags: {at_secs: number; peak_sigma: number; band: string}[], fmtTime: (s: number) => string): string` in `src/render.ts`.

- [ ] **Step 1: Write the failing test**

Add to `src/render.test.ts` (extend the import line with `seedStamp`, add inside the existing `describe` or as a sibling):

```ts
  it("seed stamp: live stream when nothing came from the bank", () => {
    expect(seedStamp(0, [], (s) => `T${s}`)).toBe("seed: live stream");
  });
  it("seed stamp: bank fraction with signed sigma per anomaly", () => {
    const fmt = (s: number) => `T${s}`;
    const tags = [
      { at_secs: 10, peak_sigma: 3.21, band: "99%" },
      { at_secs: 40, peak_sigma: -2.84, band: "95%" },
    ];
    expect(seedStamp(0.72, tags, fmt)).toBe(
      "seed: 72% anomaly bank — +3.2σ @ T10, −2.8σ @ T40",
    );
  });
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm run test`
Expected: FAIL — `seedStamp` is not exported.

- [ ] **Step 3: Implement `seedStamp`**

Add to `src/render.ts`:

```ts
/** Provenance stamp for a generation's seed, e.g.
 *  "seed: 72% anomaly bank — +3.2σ @ 14:32, −2.8σ @ 15:01". */
export function seedStamp(
  bankFraction: number,
  tags: { at_secs: number; peak_sigma: number; band: string }[],
  fmtTime: (s: number) => string,
): string {
  if (bankFraction <= 0 || !tags.length) return "seed: live stream";
  const pct = Math.round(bankFraction * 100);
  const parts = tags.map((t) => {
    const sign = t.peak_sigma >= 0 ? "+" : "−";
    return `${sign}${Math.abs(t.peak_sigma).toFixed(1)}σ @ ${fmtTime(t.at_secs)}`;
  });
  return `seed: ${pct}% anomaly bank — ${parts.join(", ")}`;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm run test`
Expected: PASS (all vitest tests green).

- [ ] **Step 5: Wire the meter and stamp**

`index.html` — replace line 27 (the anomaly-log panel) with:

```html
        <div class="panel log"><h2>Anomaly log · band excursions</h2>
          <div class="bank" title="entropy harvested while the walk is out-of-band; spent to seed generations">
            <div class="bank-meter"><div id="bank-fill"></div></div>
            <span id="bank-label">bank —</span>
          </div>
          <div id="anomalies"></div></div>
```

and insert the stamp element between `.gen-params` and `#gen-output` (after line 48):

```html
        <div id="gen-seed" class="gen-seed"></div>
```

`src/main.ts` — extend the render import with `seedStamp`; in the `snapshot` listener (after `renderAnomalies(...)`) add:

```ts
  const cap = dto.bank_capacity || 0;
  const pct = cap ? (dto.bank_fill / cap) * 100 : 0;
  (document.getElementById("bank-fill") as HTMLElement).style.width = `${pct.toFixed(1)}%`;
  document.getElementById("bank-label")!.textContent =
    `bank ${(dto.bank_fill / 1024).toFixed(1)}/${(cap / 1024).toFixed(0)} KiB`;
```

In the `diffusion` listener add a case:

```ts
    case "seeded":
      document.getElementById("gen-seed")!.textContent =
        seedStamp(m.bank_fraction, m.tags || [], fmtClock);
      break;
```

In the `genBtn` click handler (next to `genOutput.innerHTML = "";`) add:

```ts
  document.getElementById("gen-seed")!.textContent = "";
```

`src/style.css` — add after the `.anom` block:

```css
/* Anomaly bank: gold reservoir of out-of-band entropy, spent to seed latents */
.bank { display:flex; align-items:center; gap:8px; margin:0 0 6px; }
.bank-meter { flex:1; height:8px; background:#111614; border:1px solid #243029; border-radius:4px; overflow:hidden; }
#bank-fill { height:100%; width:0%; background:var(--gold); transition:width 0.3s; }
#bank-label { font-size:11px; color:var(--gold); white-space:nowrap; }
.gen-seed { font-size:11px; color:var(--gold); margin-bottom:6px; min-height:14px; }
```

- [ ] **Step 6: Verify build + tests**

Run: `npm run build && npm run test`
Expected: vite build succeeds, tests pass.

- [ ] **Step 7: Commit**

```bash
git add index.html src/main.ts src/render.ts src/render.test.ts src/style.css
git commit -m "feat(ui): anomaly-bank fill meter + seed provenance stamp"
```

---

### Task 5: Quality pass — parity audit record + "ultra" preset

The candle sampler was audited against the validated reference (`sidecar/sidecar.py`) during planning and found line-for-line faithful (step spacing `s = t − 1/steps` over `linspace(1,0,steps)`, self-conditioning on the raw `x_reconst`, `score_temp` low-temperature trick, ancestral `−expm1` update, final decode at the last γ). Per the spec, nothing non-divergent is touched; the deliverables are the written audit and a higher-step preset.

**Files:**
- Create: `docs/2026-07-09-sampler-parity-audit.md`
- Modify: `index.html:34-38` (ultra option), `src/main.ts:84-88` (MODES entry)

**Interfaces:**
- Consumes: nothing new.
- Produces: nothing other tasks rely on.

- [ ] **Step 1: Write the audit record**

Create `docs/2026-07-09-sampler-parity-audit.md`:

```markdown
# Sampler parity audit — candle port vs. reference (2026-07-09)

Scope: `diffusion-rs/src/sampler.rs::generate` vs. `sidecar/sidecar.py` (the
torch pipeline previously validated against upstream Plaid).

| Aspect | Reference (sidecar.py) | Candle port | Verdict |
|---|---|---|---|
| Timesteps | `ts = linspace(1,0,steps)`; `s = t − 1/steps` | `t = 1 − i/(steps−1)`; `s = t − 1/steps` | identical |
| Schedule | `γ = γ0 + (γ1−γ0)·sched(t)` | same (`g` closure) | identical |
| Self-conditioning | `x_selfcond = x_reconst` (raw model output, pre-temp) | same | identical |
| Score temp | `eps = (z − a_t·xr)/s_t/temp; xr = (z − s_t·eps)/a_t` | same | identical |
| Ancestral update | `c = −expm1(γs−γt)`; three-coef update | same | identical |
| Final decode | one forward at last γ, argmax | same | identical |
| Forward pass | — | validated numerically (`cargo run` validate: max\|Δ\| < 1e-2) | validated |

Conclusion: no sampling-fidelity gap. Text quality is bounded by Plaid-1B
itself (≈GPT-2-small likelihood). Remaining lever exercised here: more reverse
steps ("ultra" preset). RePlaid (arXiv:2605.18530) remains the upgrade path —
weights still unpublished as of 2026-07-07.
```

- [ ] **Step 2: Add the ultra preset**

`src/main.ts` — add to `MODES`:

```ts
  ultra: { steps: 768, seqLen: 256 },
```

`index.html` — add after the quality option:

```html
            <option value="ultra">ultra · 768 steps</option>
```

- [ ] **Step 3: Sanity-run the engine at high step count (CLI, no UI needed)**

Run (from `diffusion-rs/`): `cargo run --release -- generate 768 96`
Expected: completes without error; note ms/step — 768 steps at balanced-mode ms/step should stay in low minutes on Metal. Paste the output text into the audit doc under a "768-step sample" heading (evidence, not a benchmark).

- [ ] **Step 4: Verify frontend build**

Run: `npm run build && npm run test`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add docs/2026-07-09-sampler-parity-audit.md index.html src/main.ts
git commit -m "feat(quality): sampler parity audit (no divergence) + ultra 768-step preset"
```

---

### Task 6: README + end-to-end verification

**Files:**
- Modify: `README.md`

**Interfaces:** none.

- [ ] **Step 1: Full automated verification**

Run all four, from the repo root and `src-tauri/`:

```bash
npm run build && npm run test
cd src-tauri && cargo build --release && cargo test
```

Expected: all green. Record actual counts for the README table.

- [ ] **Step 2: Update the README**

Add a section after "What Each Panel Shows":

```markdown
## Anomaly Bank

Bytes that arrive while the coherence walk is outside its 95% envelope are
harvested into a 64 KiB FIFO **anomaly bank** (gold meter in the anomaly-log
panel). Each generation's initial latent is seeded from the bank first —
destructively, so every anomaly's bytes seed exactly one text — topped up from
the live stream when the bank runs short. The stamp under the output records
the provenance, e.g. `seed: 72% anomaly bank — +3.2σ @ 14:32`.

Demo arc: switch the source to **bad rng** → the walk escapes the gold band →
the bank floods → hit **Generate** → the text is stamped with the anomaly that
birthed it. **reset** clears the bank along with the stats.
```

Update the generation-modes copy if the README mentions presets (add `ultra · 768 steps`), refresh the verification table with the Step 1 results, and update the stale "Plan 1 … Plans 2–3 not yet implemented" framing (line 5) to reflect that diffusion generation is live.

- [ ] **Step 3: Manual visual smoke (user-run)**

Document in the README's user-run section (matching the existing convention, since headless CI can't do this):

1. `npm run tauri dev`, source **simulate** → bank meter present, near-empty (healthy source banks at ~5% duty cycle).
2. Switch to **bad rng** → walk escapes; bank meter visibly fills gold.
3. **Generate** → stamp shows a bank percentage and σ/time tags; bank meter drops by ~16 KiB (one 256-token latent).
4. Generate again with an empty bank → stamp reads `seed: live stream`.
5. **reset** → meter returns to zero.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: anomaly bank section + refreshed verification status"
```
