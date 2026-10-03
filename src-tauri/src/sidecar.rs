//! DiffusionGemma sidecar: the harness's `serve` mode as a child process,
//! speaking newline-delimited JSON on stdio (see `harness/babble_harness/serve.py`).
//!
//! Resolution order for the command:
//!   1. `BABBLE_SIDECAR_CMD` — a full command line, whitespace-split
//!      (e.g. `/path/to/python -m babble_harness.cli serve --model diffusion_gemma --quant nvfp4`)
//!   2. the first candidate root holding `harness/.venv/bin/python`: the dev
//!      checkout (compile-time path), the ancestors of the running executable,
//!      and the app-data directory — run as `… -m babble_harness.cli serve
//!      --model $BABBLE_SIDECAR_MODEL` (default `diffusion_gemma`; `tiny` and
//!      `stub` are the weight-free models for development)
//!
//! The process stays resident after the first generation (the model load is
//! the expensive part); it is dropped and respawned if it exits or errors at
//! the transport level.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

use serde_json::Value;

/// Entropy one DiffusionGemma canvas may consume: the random initial canvas,
/// then one categorical and one renoise draw per position per step (4 bytes
/// each). Mirrors `babble_harness.noise.budget_bytes`.
pub const fn budget_bytes(seq_len: usize, steps: usize) -> usize {
    4 * seq_len * (1 + 2 * steps)
}

/// DiffusionGemma's canvas; the sidecar ignores any other `seq_len` for a real model.
pub const CANVAS_LENGTH: usize = 256;

#[derive(Clone, Debug, PartialEq)]
pub struct SidecarConfig {
    pub program: String,
    pub args: Vec<String>,
    pub cwd: Option<PathBuf>,
}

/// Candidate roots that may hold `harness/`: the compile-time checkout, the
/// running executable's ancestors (a bundle placed beside the repo, or a
/// `harness` folder shipped next to the binary), and the app-data dir.
pub fn candidate_roots(compile_time_root: &std::path::Path, app_data: Option<PathBuf>) -> Vec<PathBuf> {
    let mut v = vec![compile_time_root.to_path_buf()];
    if let Ok(exe) = std::env::current_exe() {
        let mut d = exe.parent().map(|p| p.to_path_buf());
        for _ in 0..6 {
            match d {
                Some(p) => {
                    v.push(p.clone());
                    d = p.parent().map(|q| q.to_path_buf());
                }
                None => break,
            }
        }
    }
    if let Some(a) = app_data {
        v.push(a);
    }
    v
}

/// Build the sidecar command from the environment or the first root that
/// holds a harness venv.
pub fn resolve_config(roots: &[PathBuf]) -> Result<SidecarConfig, String> {
    if let Ok(cmd) = std::env::var("BABBLE_SIDECAR_CMD") {
        let mut parts = cmd.split_whitespace().map(str::to_string);
        let program = parts.next().ok_or("BABBLE_SIDECAR_CMD is empty")?;
        return Ok(SidecarConfig { program, args: parts.collect(), cwd: None });
    }
    let model = std::env::var("BABBLE_SIDECAR_MODEL").unwrap_or_else(|_| "diffusion_gemma".into());
    for root in roots {
        let harness = root.join("harness");
        let python = harness.join(".venv").join("bin").join("python");
        if python.is_file() {
            return Ok(SidecarConfig {
                program: python.to_string_lossy().into_owned(),
                args: vec!["-m".into(), "babble_harness.cli".into(), "serve".into(), "--model".into(), model],
                cwd: Some(harness),
            });
        }
    }
    let looked = roots.iter().map(|r| r.join("harness").display().to_string()).collect::<Vec<_>>().join(", ");
    Err(format!(
        "DiffusionGemma sidecar not set up: no harness/.venv/bin/python under {looked} — create it per harness/README.md, or set BABBLE_SIDECAR_CMD"
    ))
}

/// One protocol message from the sidecar.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    Ready { model: String },
    /// A `step`, `done` or `error` message, forwarded verbatim (the UI reads
    /// the same fields the Plaid engine emits).
    Message(Value),
    /// stdout closed: the process exited.
    Exit,
}

/// Parse one stdout line. Non-JSON lines (stray prints) are ignored.
pub fn parse_line(line: &str) -> Option<Event> {
    let v: Value = serde_json::from_str(line.trim()).ok()?;
    match v.get("type").and_then(Value::as_str)? {
        "ready" => Some(Event::Ready { model: v.get("model").and_then(Value::as_str).unwrap_or("?").into() }),
        "step" | "done" | "error" => Some(Event::Message(v)),
        _ => None,
    }
}

pub fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(DIGITS[(b >> 4) as usize] as char);
        s.push(DIGITS[(b & 15) as usize] as char);
    }
    s
}

pub struct Sidecar {
    child: Child,
    stdin: ChildStdin,
    rx: Receiver<Event>,
    pub model: Option<String>,
}

impl Sidecar {
    pub fn spawn(config: SidecarConfig) -> Result<Self, String> {
        let mut cmd = Command::new(&config.program);
        cmd.args(&config.args).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit());
        if let Some(cwd) = &config.cwd {
            cmd.current_dir(cwd);
        }
        let mut child = cmd.spawn().map_err(|e| format!("spawn sidecar `{}`: {e}", config.program))?;
        let stdin = child.stdin.take().ok_or("sidecar stdin")?;
        let stdout = child.stdout.take().ok_or("sidecar stdout")?;
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                match line {
                    Ok(l) => {
                        if let Some(ev) = parse_line(&l) {
                            if tx.send(ev).is_err() {
                                break;
                            }
                        }
                    }
                    Err(_) => break,
                }
            }
            let _ = tx.send(Event::Exit);
        });
        Ok(Sidecar { child, stdin, rx, model: None })
    }

    /// Block until the `ready` line (the model is resident) or `timeout`.
    pub fn wait_ready(&mut self, timeout: Duration) -> Result<(), String> {
        let deadline = Instant::now() + timeout;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match self.rx.recv_timeout(left) {
                Ok(Event::Ready { model }) => {
                    self.model = Some(model);
                    return Ok(());
                }
                Ok(Event::Message(v)) if v.get("type").and_then(Value::as_str) == Some("error") => {
                    return Err(v.get("message").and_then(Value::as_str).unwrap_or("sidecar error").into());
                }
                Ok(Event::Message(_)) => continue,
                Ok(Event::Exit) => return Err("sidecar exited before becoming ready (see its stderr)".into()),
                Err(RecvTimeoutError::Timeout) => return Err(format!("sidecar not ready after {timeout:?}")),
                Err(RecvTimeoutError::Disconnected) => return Err("sidecar reader died".into()),
            }
        }
    }

    pub fn request(&mut self, req: &Value) -> Result<(), String> {
        let line = serde_json::to_string(req).map_err(|e| e.to_string())?;
        self.stdin
            .write_all(line.as_bytes())
            .and_then(|_| self.stdin.write_all(b"\n"))
            .and_then(|_| self.stdin.flush())
            .map_err(|e| format!("write to sidecar: {e}"))
    }

    pub fn next_event(&self, timeout: Duration) -> Result<Event, String> {
        match self.rx.recv_timeout(timeout) {
            Ok(ev) => Ok(ev),
            Err(RecvTimeoutError::Timeout) => Err(format!("no reply from sidecar within {timeout:?}")),
            Err(RecvTimeoutError::Disconnected) => Ok(Event::Exit),
        }
    }
}

impl Drop for Sidecar {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_matches_python() {
        assert_eq!(budget_bytes(256, 48), 99_328);
        assert_eq!(budget_bytes(16, 5), 4 * 16 * 11);
    }

    #[test]
    fn hex_encodes() {
        assert_eq!(hex(&[0, 1, 0xab, 0xff]), "0001abff");
    }

    #[test]
    fn parses_protocol_lines_and_ignores_noise() {
        assert_eq!(parse_line(r#"{"type":"ready","model":"stub"}"#), Some(Event::Ready { model: "stub".into() }));
        assert!(matches!(parse_line(r#"{"type":"step","i":1,"total":5,"tokens":["a"]}"#), Some(Event::Message(_))));
        assert!(matches!(parse_line(r#"{"type":"done","i":5,"total":5,"tokens":[]}"#), Some(Event::Message(_))));
        assert!(matches!(parse_line(r#"{"type":"error","message":"x"}"#), Some(Event::Message(_))));
        assert_eq!(parse_line("loading model ..."), None);
        assert_eq!(parse_line(r#"{"type":"unknown"}"#), None);
        assert_eq!(parse_line(""), None);
    }

    #[test]
    fn resolve_config_prefers_env_then_dev_checkout() {
        // env override (set/unset serially; cargo runs tests in threads, so scope it tightly)
        std::env::set_var("BABBLE_SIDECAR_CMD", "/usr/bin/env python3 -m x serve --model tiny");
        let c = resolve_config(&[PathBuf::from("/nonexistent")]).unwrap();
        std::env::remove_var("BABBLE_SIDECAR_CMD");
        assert_eq!(c.program, "/usr/bin/env");
        assert_eq!(c.args, vec!["python3", "-m", "x", "serve", "--model", "tiny"]);
        assert!(c.cwd.is_none());
        // no venv anywhere → actionable error naming every root looked at
        let err = resolve_config(&[PathBuf::from("/nonexistent"), PathBuf::from("/also-not")]).unwrap_err();
        assert!(err.contains("/nonexistent/harness") && err.contains("/also-not/harness"), "{err}");
        assert!(err.contains("harness/README.md") && err.contains("BABBLE_SIDECAR_CMD"), "{err}");
        // the second root wins when the first has no venv
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
        if root.join("harness/.venv/bin/python").is_file() {
            let c = resolve_config(&[PathBuf::from("/nonexistent"), root.clone()]).unwrap();
            assert_eq!(c.cwd, Some(root.join("harness")));
        }
        let roots = candidate_roots(std::path::Path::new("/ct"), Some(PathBuf::from("/data")));
        assert_eq!(roots.first(), Some(&PathBuf::from("/ct")));
        assert_eq!(roots.last(), Some(&PathBuf::from("/data")));
        assert!(roots.len() >= 3, "exe ancestors included: {roots:?}");
    }

    /// Full round trip against the harness's stub model. Skipped when the
    /// harness venv is not set up (CI without Python).
    #[test]
    fn stub_sidecar_round_trip() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
        let python = root.join("harness/.venv/bin/python");
        if !python.is_file() {
            eprintln!("skipping: {} not found", python.display());
            return;
        }
        let cfg = SidecarConfig {
            program: python.to_string_lossy().into_owned(),
            args: vec!["-m".into(), "babble_harness.cli".into(), "serve".into(), "--model".into(), "stub".into()],
            cwd: Some(root.join("harness")),
        };
        let mut sc = Sidecar::spawn(cfg).unwrap();
        sc.wait_ready(Duration::from_secs(60)).unwrap();
        assert_eq!(sc.model.as_deref(), Some("stub"));

        let (steps, seq_len) = (5usize, 16usize);
        let entropy = vec![7u8; budget_bytes(seq_len, steps)];
        sc.request(&serde_json::json!({
            "steps": steps, "seq_len": seq_len, "entropy_hex": hex(&entropy), "preview_every": 2, "prompt": "sea"
        }))
        .unwrap();
        let mut steps_seen = 0;
        let done = loop {
            match sc.next_event(Duration::from_secs(60)).unwrap() {
                Event::Message(v) => match v["type"].as_str() {
                    Some("step") => steps_seen += 1,
                    Some("done") => break v,
                    Some(t) => panic!("unexpected {t}: {v}"),
                    None => unreachable!(),
                },
                Event::Ready { .. } => continue,
                Event::Exit => panic!("sidecar exited"),
            }
        };
        assert!(steps_seen >= 1);
        assert_eq!(done["tokens"].as_array().unwrap().len(), seq_len);
        assert_eq!(done["seed"], "entropy");
        assert_eq!(done["total"], steps);

        // too little entropy → protocol error, process stays alive
        sc.request(&serde_json::json!({"steps": 5, "seq_len": 16, "entropy_hex": "00"})).unwrap();
        match sc.next_event(Duration::from_secs(30)).unwrap() {
            Event::Message(v) => assert_eq!(v["type"], "error"),
            other => panic!("{other:?}"),
        }
    }
}
