//! Entropy sources. A background thread reads bytes and ships chunks over an
//! `mpsc` channel to the UI thread, which never blocks on I/O.
//!
//! Ported verbatim from ghostty-rng. `SourceStatus::Error` is reserved for
//! fatal-source reporting wired up in a later plan, so it is currently unused.
#![allow(dead_code)]

use std::io::Read;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;

use anyhow::Result;

/// A status line the reader thread can hand back to the UI.
#[derive(Clone)]
pub enum SourceStatus {
    Streaming,
    Reconnecting(String),
    Error(String),
}

pub enum SourceKind {
    /// Real serial device, e.g. /dev/cu.usbmodem*.
    Serial { path: String, baud: u32 },
    /// High-quality software RNG (xoshiro256**), for demos without hardware.
    Simulate,
    /// Deliberately biased LCG — makes the entropy charts visibly fail.
    BadRng,
}

pub struct Source {
    pub data_rx: Receiver<Vec<u8>>,
    pub status_rx: Receiver<SourceStatus>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
    pub label: String,
}

impl Source {
    pub fn spawn(kind: SourceKind) -> Self {
        let (data_tx, data_rx) = std::sync::mpsc::channel();
        let (status_tx, status_rx) = std::sync::mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_thread = stop.clone();

        let label = match &kind {
            SourceKind::Serial { path, baud } => format!("{path} @ {baud}"),
            SourceKind::Simulate => "simulator · xoshiro256**".to_string(),
            SourceKind::BadRng => "BAD RNG · biased LCG".to_string(),
        };

        let handle = std::thread::spawn(move || {
            let _ = run(kind, data_tx, status_tx, stop_thread);
        });

        Source {
            data_rx,
            status_rx,
            stop,
            handle: Some(handle),
            label,
        }
    }
}

impl Drop for Source {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn run(
    kind: SourceKind,
    data_tx: Sender<Vec<u8>>,
    status_tx: Sender<SourceStatus>,
    stop: Arc<AtomicBool>,
) -> Result<()> {
    match kind {
        SourceKind::Serial { path, baud } => run_serial(&path, baud, data_tx, status_tx, stop),
        SourceKind::Simulate => run_software(false, data_tx, status_tx, stop),
        SourceKind::BadRng => run_software(true, data_tx, status_tx, stop),
    }
}

fn run_serial(
    path: &str,
    baud: u32,
    data_tx: Sender<Vec<u8>>,
    status_tx: Sender<SourceStatus>,
    stop: Arc<AtomicBool>,
) -> Result<()> {
    let mut buf = vec![0u8; 16 * 1024];
    loop {
        if stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        let port = serialport::new(path, baud)
            .timeout(Duration::from_millis(200))
            .open();
        let mut port = match port {
            Ok(p) => p,
            Err(e) => {
                let _ = status_tx.send(SourceStatus::Reconnecting(format!("open failed: {e}")));
                sleep_interruptible(&stop, 500);
                continue;
            }
        };
        // TrueRNG starts streaming once DTR is asserted.
        let _ = port.write_data_terminal_ready(true);
        let _ = status_tx.send(SourceStatus::Streaming);

        loop {
            if stop.load(Ordering::Relaxed) {
                return Ok(());
            }
            match port.read(&mut buf) {
                Ok(0) => {}
                Ok(n) => {
                    if data_tx.send(buf[..n].to_vec()).is_err() {
                        return Ok(());
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::TimedOut => {}
                Err(e) => {
                    let _ = status_tx
                        .send(SourceStatus::Reconnecting(format!("read error: {e}")));
                    break; // reopen
                }
            }
        }
        sleep_interruptible(&stop, 400);
    }
}

/// Software source. `bad = true` emits a low-entropy biased stream.
fn run_software(
    bad: bool,
    data_tx: Sender<Vec<u8>>,
    status_tx: Sender<SourceStatus>,
    stop: Arc<AtomicBool>,
) -> Result<()> {
    // Software sources are always "live" — report it so the header doesn't sit
    // on the initial "connecting…" state forever.
    let _ = status_tx.send(SourceStatus::Streaming);
    let mut rng = Xoshiro256::seeded();
    let mut lcg: u32 = 0x1234_5678;
    loop {
        if stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        let mut chunk = vec![0u8; 8 * 1024];
        if bad {
            for b in chunk.iter_mut() {
                // A weak LCG, then squashed toward a few values to crush entropy.
                lcg = lcg.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                let v = (lcg >> 16) as u8;
                *b = v & 0b0011_1111; // only 6 bits, biased distribution
            }
        } else {
            for b in chunk.iter_mut() {
                *b = (rng.next_u64() >> 24) as u8;
            }
        }
        if data_tx.send(chunk).is_err() {
            return Ok(());
        }
        // Throttle the simulator to a believable hardware-RNG rate (~400 KiB/s).
        sleep_interruptible(&stop, 20);
    }
}

fn sleep_interruptible(stop: &Arc<AtomicBool>, ms: u64) {
    let step = 20;
    let mut left = ms;
    while left > 0 {
        if stop.load(Ordering::Relaxed) {
            return;
        }
        let s = step.min(left);
        std::thread::sleep(Duration::from_millis(s));
        left -= s;
    }
}

/// Auto-detect a likely TrueRNG / USB-CDC serial device on macOS or Linux.
pub fn autodetect() -> Option<String> {
    let ports = serialport::available_ports().ok()?;
    // Prefer cu.usbmodem* (macOS callout devices), then any usbmodem/ttyACM.
    let mut candidates: Vec<String> = ports
        .into_iter()
        .map(|p| p.port_name)
        .filter(|n| {
            n.contains("usbmodem")
                || n.contains("ttyACM")
                || n.to_lowercase().contains("truerng")
        })
        .collect();
    candidates.sort_by_key(|n| !n.contains("cu.")); // cu.* first on macOS
    candidates.into_iter().next()
}

// --- xoshiro256** : fast, high-quality PRNG for the simulator ---------------

struct Xoshiro256 {
    s: [u64; 4],
}

impl Xoshiro256 {
    fn seeded() -> Self {
        // SplitMix64-expand a time-derived seed.
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E37_79B9_7F4A_7C15)
            ^ (std::process::id() as u64).wrapping_mul(0x2545_F491_4F6C_DD1D);
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Xoshiro256 {
            s: [next(), next(), next(), next()],
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }
}

/// Pull bytes from the channel without blocking; returns total bytes fed.
pub fn drain_into(rx: &Receiver<Vec<u8>>, sink: &mut crate::stats::Stats) -> usize {
    let mut total = 0;
    while let Ok(chunk) = rx.try_recv() {
        total += chunk.len();
        sink.push(&chunk);
    }
    total
}

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
