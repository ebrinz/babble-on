//! Manages the Python diffusion sidecar (Plaid-1B).
//!
//! The sidecar is spawned on first use (loading the model is slow, so we keep
//! the process resident) and driven over its newline-JSON stdio protocol. A
//! generation request is seeded with fresh TrueRNG entropy pulled from the live
//! stats engine, and each crystallizing-text update the sidecar streams back is
//! re-emitted to the webview as a `diffusion` event.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use serde_json::json;
use tauri::{AppHandle, Emitter};

/// Plaid's token-embedding dimension; the initial latent is
/// `n_samples * seq_len * EMBED_DIM` Gaussians, 4 entropy bytes each.
pub const EMBED_DIM: usize = 16;

fn sidecar_dir() -> PathBuf {
    if let Ok(d) = std::env::var("BABBLE_SIDECAR_DIR") {
        return PathBuf::from(d);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("sidecar")
}

struct Proc {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Drop for Proc {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub struct Diffusion {
    proc: Mutex<Option<Proc>>,
    busy: AtomicBool,
}

impl Diffusion {
    pub fn new() -> Self {
        Diffusion { proc: Mutex::new(None), busy: AtomicBool::new(false) }
    }

    /// Try to claim the sidecar for one generation. Returns false if a
    /// generation is already running (the sidecar serves one at a time).
    pub fn try_acquire(&self) -> bool {
        !self.busy.swap(true, Ordering::SeqCst)
    }

    pub fn release(&self) {
        self.busy.store(false, Ordering::SeqCst);
    }
}

fn spawn_sidecar() -> Result<Proc, String> {
    let dir = sidecar_dir();
    let python = dir.join(".venv").join("bin").join("python");
    let mut child = Command::new(&python)
        .arg("sidecar.py")
        .current_dir(&dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit()) // sidecar's own logs flow to the app's stderr
        .spawn()
        .map_err(|e| format!("spawn {} failed: {e}", python.display()))?;
    let stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // Block until the model is resident and the sidecar announces readiness.
    let mut line = String::new();
    loop {
        line.clear();
        match stdout.read_line(&mut line) {
            Ok(0) => return Err("sidecar exited before becoming ready".into()),
            Ok(_) => {
                if line.contains("\"ready\"") {
                    break;
                }
            }
            Err(e) => return Err(format!("reading sidecar startup: {e}")),
        }
    }
    Ok(Proc { child, stdin, stdout })
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

/// Run one generation to completion, streaming `diffusion` events. Holds the
/// sidecar process lock for the whole run (the caller serializes via `busy`).
pub fn run_generation(
    app: &AppHandle,
    diffusion: &Diffusion,
    steps: usize,
    seq_len: usize,
    n_samples: usize,
    entropy: Vec<u8>,
) -> Result<(), String> {
    let need = n_samples * seq_len * EMBED_DIM * 4;
    let entropy_hex = if entropy.len() >= need { Some(hex(&entropy[..need])) } else { None };

    let mut guard = diffusion.proc.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        let _ = app.emit("diffusion", json!({"type": "loading"}));
        *guard = Some(spawn_sidecar()?);
    }
    let proc = guard.as_mut().unwrap();

    let req = json!({
        "steps": steps,
        "seq_len": seq_len,
        "n_samples": n_samples,
        "preview_every": (steps / 12).max(1),
        "entropy_hex": entropy_hex,
    });
    writeln!(proc.stdin, "{req}").map_err(|e| format!("write request: {e}"))?;
    proc.stdin.flush().ok();

    let mut line = String::new();
    loop {
        line.clear();
        match proc.stdout.read_line(&mut line) {
            Ok(0) => {
                *guard = None; // sidecar died mid-generation — respawn next time
                return Err("sidecar closed unexpectedly".into());
            }
            Ok(_) => {}
            Err(e) => return Err(format!("reading sidecar: {e}")),
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let val: serde_json::Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => return Err(format!("bad sidecar json: {e}")),
        };
        let kind = val.get("type").and_then(|t| t.as_str()).unwrap_or("").to_string();
        let _ = app.emit("diffusion", val);
        if kind == "done" || kind == "error" {
            break;
        }
    }
    Ok(())
}
