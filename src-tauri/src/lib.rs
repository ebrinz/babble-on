use std::sync::mpsc::{channel, Sender};
use std::sync::Mutex;
use std::time::Duration;
use tauri::Emitter;
use engine::{Engine, ControlMsg, resolve_kind};

mod math;
mod stats;
mod source;
mod dto;
mod engine;

struct Control(Mutex<Sender<ControlMsg>>);

#[tauri::command]
fn start_source(kind: String, path: Option<String>, baud: Option<u32>, ctrl: tauri::State<Control>) {
    let k = resolve_kind(&kind, path, baud.unwrap_or(9600));
    let _ = ctrl.0.lock().unwrap().send(ControlMsg::SetSource(k));
}

#[tauri::command]
fn reset(ctrl: tauri::State<Control>) {
    let _ = ctrl.0.lock().unwrap().send(ControlMsg::Reset);
}

#[tauri::command]
fn set_window(n: usize, ctrl: tauri::State<Control>) {
    let _ = ctrl.0.lock().unwrap().send(ControlMsg::SetWindow(n));
}

#[tauri::command]
fn set_paused(paused: bool, ctrl: tauri::State<Control>) {
    let _ = ctrl.0.lock().unwrap().send(ControlMsg::SetPaused(paused));
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let (tx, rx) = channel::<ControlMsg>();
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
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
