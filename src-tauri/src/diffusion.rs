//! All-Rust diffusion backend. Loads the candle Plaid-1B engine (resident after
//! first use) and runs generations seeded with live TrueRNG entropy, streaming
//! the crystallizing-text updates to the webview as `diffusion` events.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use diffusion_rs::Engine;
use serde_json::json;
use tauri::path::BaseDirectory;
use tauri::{AppHandle, Emitter, Manager};
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

/// First candidate directory that actually holds the converted model
/// (`plaid1b.safetensors`), or an error naming every location checked so a
/// bundled app can tell the user exactly where to put the weights.
fn resolve_model_dir(candidates: &[PathBuf]) -> Result<PathBuf, String> {
    for dir in candidates {
        if dir.join("plaid1b.safetensors").is_file() {
            return Ok(dir.clone());
        }
    }
    let list = candidates
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join(" or ");
    Err(format!(
        "model not found — put the converted weights (plaid1b.safetensors + meta.json) in {list}; see the README's Text Generation Setup"
    ))
}

/// Where the model may live: the dev checkout (running from the repo), then
/// the per-user app-data dir (running a bundled build).
fn model_candidates(app: &AppHandle) -> Vec<PathBuf> {
    let mut v = vec![PathBuf::from(rel(&["models", "plaid1b"]))];
    if let Ok(data) = app.path().app_data_dir() {
        v.push(data.join("models").join("plaid1b"));
    }
    v
}

/// The tokenizer ships with the app: dev checkout path first, then the copy
/// bundled as a Tauri resource.
fn resolve_tokenizer(app: &AppHandle) -> Result<PathBuf, String> {
    let dev = PathBuf::from(rel(&["sidecar", "misc", "owt2_tokenizer.json"]));
    if dev.is_file() {
        return Ok(dev);
    }
    if let Ok(res) = app.path().resolve("owt2_tokenizer.json", BaseDirectory::Resource) {
        if res.is_file() {
            return Ok(res);
        }
    }
    Err("bundled tokenizer missing (owt2_tokenizer.json)".into())
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
    draw_seed: impl FnOnce(usize) -> SeedReply,
) -> Result<(), String> {
    let mut guard = diffusion.engine.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        let _ = app.emit("diffusion", json!({"type": "loading"}));
        let model_dir = resolve_model_dir(&model_candidates(app))?;
        let tokenizer = resolve_tokenizer(app)?;
        let eng = Engine::load(&model_dir.to_string_lossy(), &tokenizer.to_string_lossy(), true)
            .map_err(|e| format!("load model: {e}"))?;
        *guard = Some(eng);
    }
    let eng = guard.as_ref().unwrap();

    // Spend the bank only once the model is resident (a load failure above
    // must not consume anomaly bytes).
    let need = seq_len * EMBED_DIM * 4;
    let seed = draw_seed(need);
    let _ = app.emit(
        "diffusion",
        json!({"type": "seeded", "bank_fraction": seed.bank_fraction, "tags": seed.tags}),
    );
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_model_dir_picks_first_candidate_with_weights() {
        let tmp = std::env::temp_dir().join("babble-resolve-test");
        let empty = tmp.join("empty");
        let stocked = tmp.join("stocked");
        std::fs::create_dir_all(&empty).unwrap();
        std::fs::create_dir_all(&stocked).unwrap();
        std::fs::write(stocked.join("plaid1b.safetensors"), b"x").unwrap();

        let got = resolve_model_dir(&[empty.clone(), stocked.clone()]).unwrap();
        assert_eq!(got, stocked);

        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn resolve_model_dir_error_names_every_checked_path() {
        let a = PathBuf::from("/nonexistent/dev/models/plaid1b");
        let b = PathBuf::from("/nonexistent/appdata/models/plaid1b");
        let err = resolve_model_dir(&[a, b]).unwrap_err();
        assert!(err.contains("/nonexistent/dev/models/plaid1b"), "err: {err}");
        assert!(err.contains("/nonexistent/appdata/models/plaid1b"), "err: {err}");
        assert!(err.contains("plaid1b.safetensors"), "err: {err}");
    }
}
