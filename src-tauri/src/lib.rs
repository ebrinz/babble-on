use std::path::PathBuf;
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use serde_json::json;
use tauri::{Emitter, Manager};
use engine::{Engine, ControlMsg, SeedReply, resolve_kind};
use diffusion::{Diffusion, run_generation, EMBED_DIM};
use bbrec::{SeedBundle, SeedTag};

/// Default seed export size: one DiffusionGemma canvas at the reference
/// settings, `4·256·(1 + 2·48)` bytes (see docs/superpowers/specs/2026-10-02).
pub const GEMMA_SEED_BYTES: usize = 4 * 256 * (1 + 2 * 48);

mod math;
mod stats;
mod bank;
mod source;
mod dto;
mod engine;
mod diffusion;
mod sidecar;

/// The resident DiffusionGemma sidecar, spawned on first use.
struct GemmaSidecar(Mutex<Option<sidecar::Sidecar>>);

struct Control(Mutex<Sender<ControlMsg>>);

#[tauri::command]
fn start_source(kind: String, path: Option<String>, baud: Option<u32>, ctrl: tauri::State<Control>) {
    let k = resolve_kind(&kind, path, baud.unwrap_or(9600));
    let _ = ctrl.0.lock().unwrap_or_else(|e| e.into_inner()).send(ControlMsg::SetSource(k));
}

#[tauri::command]
fn reset(ctrl: tauri::State<Control>) {
    let _ = ctrl.0.lock().unwrap_or_else(|e| e.into_inner()).send(ControlMsg::Reset);
}

#[tauri::command]
fn set_window(n: usize, ctrl: tauri::State<Control>) {
    let _ = ctrl.0.lock().unwrap_or_else(|e| e.into_inner()).send(ControlMsg::SetWindow(n));
}

#[tauri::command]
fn set_paused(paused: bool, ctrl: tauri::State<Control>) {
    let _ = ctrl.0.lock().unwrap_or_else(|e| e.into_inner()).send(ControlMsg::SetPaused(paused));
}

/// Kick off a diffusion generation. Pulls fresh hardware entropy from the live
/// stats engine (anomaly bank first, live-stream top-up), then runs the chosen
/// engine on a background thread, streaming `diffusion` events to the webview.
///
/// `engine`: `"plaid"` (default) runs the in-process candle Plaid-1B engine
/// seeded with a 16-KiB initial latent; `"gemma"` runs DiffusionGemma through
/// the harness sidecar, which draws every random choice of its masked
/// diffusion sampler from a `4·256·(1+2·steps)`-byte tape.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
fn generate(
    steps: Option<usize>,
    seq_len: Option<usize>,
    n_samples: Option<usize>,
    temperature: Option<f64>,
    noise_scale: Option<f64>,
    ddim: Option<bool>,
    prompt: Option<String>,
    engine: Option<String>,
    app: tauri::AppHandle,
    ctrl: tauri::State<Control>,
    diff: tauri::State<Arc<Diffusion>>,
    gemma: tauri::State<Arc<GemmaSidecar>>,
) -> Result<(), String> {
    let steps = steps.unwrap_or(256);
    let seq_len = seq_len.unwrap_or(256);
    let n_samples = n_samples.unwrap_or(1);
    let temperature = temperature.unwrap_or(0.9);
    let noise_scale = noise_scale.unwrap_or(1.0);
    let ddim = ddim.unwrap_or(false);
    let use_gemma = engine.as_deref() == Some("gemma");

    if !diff.try_acquire() {
        return Err("a generation is already in progress".into());
    }

    let need = if use_gemma {
        sidecar::budget_bytes(sidecar::CANVAS_LENGTH, steps)
    } else {
        n_samples * seq_len * EMBED_DIM * 4
    };
    // The seed is drawn (destructively, bank first) only once the engine is
    // ready, so a model/sidecar failure never spends the anomaly bank.
    let sender = ctrl.0.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let draw_seed = move |n: usize| -> SeedReply {
        let (etx, erx) = channel::<SeedReply>();
        let _ = sender.send(ControlMsg::GetSeed(n, etx));
        erx.recv_timeout(Duration::from_secs(2)).unwrap_or(SeedReply {
            bytes: Vec::new(),
            bank_fraction: 0.0,
            tags: Vec::new(),
        })
    };

    let app2 = app.clone();
    let diff2: Arc<Diffusion> = diff.inner().clone();
    let gemma2: Arc<GemmaSidecar> = gemma.inner().clone();
    std::thread::spawn(move || {
        let result = if use_gemma {
            run_gemma_generation(&app2, &gemma2, steps, prompt, need, draw_seed)
        } else {
            run_generation(&app2, &diff2, steps, seq_len, n_samples, temperature, noise_scale, ddim, prompt, draw_seed)
        };
        if let Err(e) = result {
            let _ = app2.emit("diffusion", serde_json::json!({"type": "error", "message": e}));
        }
        diff2.release();
    });
    Ok(())
}

/// Where `harness/` may live at run time (see `sidecar::candidate_roots`).
fn sidecar_roots(app: &tauri::AppHandle) -> Vec<PathBuf> {
    let compile_time = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    sidecar::candidate_roots(&compile_time, app.path().app_data_dir().ok())
}

/// One DiffusionGemma generation through the sidecar, forwarding its
/// `step`/`done`/`error` messages as `diffusion` events. Spawns the sidecar
/// on first use (emitting `loading` while the model becomes resident).
fn run_gemma_generation(
    app: &tauri::AppHandle,
    gemma: &GemmaSidecar,
    steps: usize,
    prompt: Option<String>,
    need: usize,
    draw_seed: impl FnOnce(usize) -> SeedReply,
) -> Result<(), String> {
    let mut guard = gemma.0.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        let _ = app.emit("diffusion", json!({"type": "loading"}));
        let cfg = sidecar::resolve_config(&sidecar_roots(app))?;
        let mut sc = sidecar::Sidecar::spawn(cfg)?;
        sc.wait_ready(Duration::from_secs(1800))?; // the 26B load can take minutes
        *guard = Some(sc);
    }

    // Only now spend the bank. A short seed is refused rather than silently
    // replaced by a PRNG tape: the text must be a function of the bytes.
    let seed = draw_seed(need);
    if seed.bytes.len() < need {
        return Err(format!(
            "only {} of {need} seed bytes available yet — let the stream run a few seconds, then generate again",
            seed.bytes.len()
        ));
    }
    let _ = app.emit(
        "diffusion",
        json!({"type": "seeded", "bank_fraction": seed.bank_fraction, "tags": seed.tags}),
    );
    let mut req = json!({"steps": steps, "seq_len": sidecar::CANVAS_LENGTH, "preview_every": 1,
                         "entropy_hex": sidecar::hex(&seed.bytes[..need])});
    if let Some(p) = prompt.as_deref().map(str::trim).filter(|p| !p.is_empty()) {
        req["prompt"] = json!(p);
    }
    let sc = guard.as_mut().unwrap();
    if let Err(e) = sc.request(&req) {
        *guard = None; // transport error: drop the dead process, respawn next time
        return Err(e);
    }
    loop {
        match sc.next_event(Duration::from_secs(600)) {
            Ok(sidecar::Event::Message(v)) => {
                let kind = v.get("type").and_then(serde_json::Value::as_str).unwrap_or("");
                let _ = app.emit("diffusion", &v);
                if kind == "done" || kind == "error" {
                    return Ok(());
                }
            }
            Ok(sidecar::Event::Ready { .. }) => continue,
            Ok(sidecar::Event::Exit) => {
                *guard = None;
                return Err("DiffusionGemma sidecar exited (see the terminal for its stderr)".into());
            }
            Err(e) => {
                *guard = None;
                return Err(e);
            }
        }
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Draw seed bytes exactly as `generate` would (anomaly bank first, live
/// top-up) and write them as a seed bundle the harness can load. Spending is
/// destructive, as for a generation. Returns the `.seed.bin` path.
#[tauri::command]
fn export_seed(
    n_bytes: Option<usize>,
    app: tauri::AppHandle,
    ctrl: tauri::State<Control>,
) -> Result<String, String> {
    let need = n_bytes.unwrap_or(GEMMA_SEED_BYTES);
    let (etx, erx) = channel::<SeedReply>();
    let _ = ctrl
        .0
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .send(ControlMsg::GetSeed(need, etx));
    let seed = erx.recv_timeout(Duration::from_secs(2)).map_err(|_| "engine did not reply".to_string())?;
    if seed.bytes.len() < need {
        return Err(format!("only {} of {need} bytes available yet — let the stream run", seed.bytes.len()));
    }
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("app data dir: {e}"))?
        .join("exports");
    let stamp = now_ms();
    let bundle = SeedBundle {
        bytes: seed.bytes.len(),
        bank_fraction: seed.bank_fraction,
        tags: seed.tags.iter().map(|t| SeedTag { at_secs: t.at_secs, peak_sigma: t.peak_sigma, band: t.band.to_string() }).collect(),
        created_at_ms: stamp,
        source: "babble-on".into(),
        label: SeedBundle::label_for(seed.bank_fraction).into(),
    };
    let (bin, _json) = bundle
        .write(&dir.join(format!("seed-{stamp}")), &seed.bytes)
        .map_err(|e| format!("write seed bundle: {e}"))?;
    Ok(bin.to_string_lossy().into_owned())
}

/// Start recording the raw stream to a `.bbrec`. `path` defaults to
/// `<app-data>/recordings/<timestamp>.bbrec`. Returns the path.
#[tauri::command]
fn start_recording(
    path: Option<String>,
    app: tauri::AppHandle,
    ctrl: tauri::State<Control>,
) -> Result<String, String> {
    let path = match path {
        Some(p) => PathBuf::from(p),
        None => app
            .path()
            .app_data_dir()
            .map_err(|e| format!("app data dir: {e}"))?
            .join("recordings")
            .join(format!("stream-{}.bbrec", now_ms())),
    };
    let (tx, rx) = channel();
    let _ = ctrl
        .0
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .send(ControlMsg::StartRecording(path, tx));
    rx.recv_timeout(Duration::from_secs(2)).map_err(|_| "engine did not reply".to_string())?
}

#[tauri::command]
fn stop_recording(ctrl: tauri::State<Control>) {
    let _ = ctrl.0.lock().unwrap_or_else(|e| e.into_inner()).send(ControlMsg::StopRecording);
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let (tx, rx) = channel::<ControlMsg>();
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(Control(Mutex::new(tx)))
        .manage(Arc::new(Diffusion::new()))
        .manage(Arc::new(GemmaSidecar(Mutex::new(None))))
        .invoke_handler(tauri::generate_handler![
            start_source, reset, set_window, set_paused, generate,
            export_seed, start_recording, stop_recording
        ])
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
