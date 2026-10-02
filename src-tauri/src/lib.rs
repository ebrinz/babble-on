use std::path::PathBuf;
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;
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
/// stats engine to seed the latent, then runs the sidecar on a background
/// thread, streaming `diffusion` events to the webview. (`prompt` is accepted
/// for forward-compat but unused in this unconditional first version.)
#[tauri::command]
fn generate(
    steps: Option<usize>,
    seq_len: Option<usize>,
    n_samples: Option<usize>,
    temperature: Option<f64>,
    noise_scale: Option<f64>,
    ddim: Option<bool>,
    prompt: Option<String>,
    app: tauri::AppHandle,
    ctrl: tauri::State<Control>,
    diff: tauri::State<Arc<Diffusion>>,
) -> Result<(), String> {
    let steps = steps.unwrap_or(256);
    let seq_len = seq_len.unwrap_or(256);
    let n_samples = n_samples.unwrap_or(1);
    let temperature = temperature.unwrap_or(0.9);
    let noise_scale = noise_scale.unwrap_or(1.0);
    let ddim = ddim.unwrap_or(false);

    if !diff.try_acquire() {
        return Err("a generation is already in progress".into());
    }

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

    let app2 = app.clone();
    let diff2: Arc<Diffusion> = diff.inner().clone();
    std::thread::spawn(move || {
        if let Err(e) = run_generation(&app2, &diff2, steps, seq_len, n_samples, temperature, noise_scale, ddim, prompt, seed) {
            let _ = app2.emit("diffusion", serde_json::json!({"type": "error", "message": e}));
        }
        diff2.release();
    });
    Ok(())
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
