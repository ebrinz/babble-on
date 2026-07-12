use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri::Emitter;
use engine::{Engine, ControlMsg, SeedReply, resolve_kind};
use diffusion::{Diffusion, run_generation, EMBED_DIM};

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
    let entropy = seed.bytes;

    let app2 = app.clone();
    let diff2: Arc<Diffusion> = diff.inner().clone();
    std::thread::spawn(move || {
        if let Err(e) = run_generation(&app2, &diff2, steps, seq_len, n_samples, temperature, noise_scale, ddim, prompt, entropy) {
            let _ = app2.emit("diffusion", serde_json::json!({"type": "error", "message": e}));
        }
        diff2.release();
    });
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let (tx, rx) = channel::<ControlMsg>();
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(Control(Mutex::new(tx)))
        .manage(Arc::new(Diffusion::new()))
        .invoke_handler(tauri::generate_handler![
            start_source, reset, set_window, set_paused, generate
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
