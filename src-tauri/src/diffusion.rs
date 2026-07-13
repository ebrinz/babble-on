//! All-Rust diffusion backend. Loads the candle Plaid-1B engine (resident after
//! first use) and runs generations seeded with live TrueRNG entropy, streaming
//! the crystallizing-text updates to the webview as `diffusion` events.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use diffusion_rs::Engine;
use serde_json::json;
use tauri::{AppHandle, Emitter};
use crate::engine::SeedReply;

/// Plaid's token-embedding dim; the initial latent is `seq_len * EMBED_DIM`
/// Gaussians, 4 entropy bytes each.
pub const EMBED_DIM: usize = 16;

fn rel(parts: &[&str]) -> String {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("..");
    for part in parts {
        p.push(part);
    }
    p.to_string_lossy().into_owned()
}

pub struct Diffusion {
    engine: Mutex<Option<Engine>>,
    busy: AtomicBool,
}

impl Diffusion {
    pub fn new() -> Self {
        Diffusion { engine: Mutex::new(None), busy: AtomicBool::new(false) }
    }

    /// Claim the engine for one generation (it serves one at a time).
    pub fn try_acquire(&self) -> bool {
        !self.busy.swap(true, Ordering::SeqCst)
    }

    pub fn release(&self) {
        self.busy.store(false, Ordering::SeqCst);
    }
}

/// Run one generation to completion, streaming `diffusion` events. Loads the
/// model on first call (emitting a `loading` event while it does).
#[allow(clippy::too_many_arguments)]
pub fn run_generation(
    app: &AppHandle,
    diffusion: &Diffusion,
    steps: usize,
    seq_len: usize,
    _n_samples: usize, // the candle engine generates a single sample
    temperature: f64,
    noise_scale: f64,
    ddim: bool,
    prompt: Option<String>,
    seed: SeedReply,
) -> Result<(), String> {
    let mut guard = diffusion.engine.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        let _ = app.emit("diffusion", json!({"type": "loading"}));
        let eng = Engine::load(&rel(&["models", "plaid1b"]), &rel(&["sidecar", "misc", "owt2_tokenizer.json"]), true)
            .map_err(|e| format!("load model: {e}"))?;
        *guard = Some(eng);
    }
    let eng = guard.as_ref().unwrap();

    let _ = app.emit(
        "diffusion",
        json!({"type": "seeded", "bank_fraction": seed.bank_fraction, "tags": seed.tags}),
    );

    let need = seq_len * EMBED_DIM * 4;
    let entropy = seed.bytes;
    let entropy_opt = if entropy.len() >= need { Some(&entropy[..need]) } else { None };
    let preview_every = (steps / 24).max(1); // dense cadence for a smooth "boil"

    let t0 = std::time::Instant::now();
    let tokens = eng
        .generate(steps, seq_len, temperature, noise_scale, ddim, preview_every, entropy_opt, prompt.as_deref(), |i, total, toks| {
            let _ = app.emit("diffusion", json!({"type": "step", "i": i, "total": total, "tokens": toks}));
        })
        .map_err(|e| format!("generate: {e}"))?;
    let elapsed = (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0;
    let _ = app.emit(
        "diffusion",
        json!({"type": "done", "i": steps, "total": steps, "tokens": tokens, "elapsed": elapsed}),
    );
    Ok(())
}
