use std::path::PathBuf;
use std::time::Instant;

use bbrec::{RecordingHeader, RecordingWriter};

use crate::source::{Source, SourceKind, SourceStatus, autodetect};
use crate::stats::{AnomalyEvent, Band, Stats, TRIAL_INTERVAL, TRIAL_MIN_BITS};
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
    /// Start writing every tick's drained bytes to a `.bbrec` at this path
    /// (replaces any recording in progress). Replies with the result.
    StartRecording(PathBuf, std::sync::mpsc::Sender<Result<String, String>>),
    StopRecording,
}

/// An open `.bbrec` plus the clock its frames are stamped against.
struct Recorder {
    writer: RecordingWriter<std::fs::File>,
    path: PathBuf,
    started: Instant,
}

pub struct Engine {
    source: Source,
    stats: Stats,
    bank: AnomalyBank,
    paused: bool,
    status: String,
    recorder: Option<Recorder>,
}

impl Engine {
    pub fn new(initial: SourceKind) -> Self {
        let source = Source::spawn(initial);
        Engine { source, stats: Stats::new(1 << 16), bank: AnomalyBank::new(BANK_CAPACITY), paused: false, status: "connecting".into(), recorder: None }
    }

    fn start_recording(&mut self, path: PathBuf) -> Result<String, String> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
        }
        let ws = self.stats.walk_state();
        let header = RecordingHeader {
            source: self.source.label.clone(),
            started_at_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            trial_interval_ms: TRIAL_INTERVAL.as_millis() as u64,
            trial_min_bits: TRIAL_MIN_BITS,
            notes: format!("babble-on {}", env!("CARGO_PKG_VERSION")),
            walk_cum: ws.cum,
            walk_k: ws.k,
            trial_ones: ws.trial_ones,
            trial_bits: ws.trial_bits,
            since_last_trial_ns: ws.since_last_trial.as_nanos() as u64,
        };
        // Open the new file first so a failure leaves the current recording running.
        let writer = RecordingWriter::create(&path, &header).map_err(|e| format!("open {}: {e}", path.display()))?;
        self.stop_recording();
        let shown = path.to_string_lossy().into_owned();
        self.recorder = Some(Recorder { writer, path, started: Instant::now() });
        Ok(shown)
    }

    fn stop_recording(&mut self) {
        if let Some(r) = self.recorder.take() {
            let _ = r.writer.finish();
        }
    }

    /// (path, bytes written) of the recording in progress.
    pub fn recording(&self) -> Option<(String, u64)> {
        self.recorder.as_ref().map(|r| (r.path.to_string_lossy().into_owned(), r.writer.bytes))
    }

    pub fn apply(&mut self, msg: ControlMsg) {
        match msg {
            // A recording is one source's stream with one continuous walk, so a
            // source switch or a reset (which restarts the walk) ends it.
            ControlMsg::SetSource(kind) => {
                self.stop_recording();
                self.source = Source::spawn(kind);
                self.status = "connecting".into();
            }
            ControlMsg::Reset => {
                self.stop_recording();
                self.stats.reset();
                self.bank.clear();
            }
            ControlMsg::SetWindow(n) => self.stats.set_capacity(n),
            ControlMsg::SetPaused(p) => self.paused = p,
            ControlMsg::GetSeed(n, reply) => {
                let (mut bytes, tags) = self.bank.withdraw(n);
                let banked = bytes.len();
                if banked < n {
                    bytes.extend_from_slice(&self.stats.fresh_entropy(n - banked));
                }
                let bank_fraction = if n == 0 { 0.0 } else { banked as f64 / n as f64 };
                let _ = reply.send(SeedReply { bytes, bank_fraction, tags });
            }
            ControlMsg::StartRecording(path, reply) => {
                let _ = reply.send(self.start_recording(path));
            }
            ControlMsg::StopRecording => self.stop_recording(),
        }
    }

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
            // One frame per tick: exactly the bytes the walk just saw, stamped
            // on the recording's own clock. Flushed per tick so a crash loses
            // at most one frame (the reader tolerates a torn tail).
            if let Some(r) = self.recorder.as_mut() {
                let t_ns = r.started.elapsed().as_nanos() as u64;
                if r.writer.frame(t_ns, &drained).and_then(|_| r.writer.flush()).is_err() {
                    self.status = "error: recording write failed".into();
                    self.recorder = None;
                }
            }
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
        if let Some((path, bytes)) = self.recording() {
            dto.recording = Some(path);
            dto.recording_bytes = bytes;
        }
        dto
    }
}

/// Resolve a UI request string into a concrete source kind.
pub fn resolve_kind(kind: &str, path: Option<String>, baud: u32) -> SourceKind {
    match kind {
        "bad" => SourceKind::BadRng,
        "serial" => match path.or_else(autodetect) {
            Some(p) => SourceKind::Serial { path: p, baud },
            None => SourceKind::Simulate,
        },
        "auto" => match autodetect() {
            Some(p) => SourceKind::Serial { path: p, baud },
            None => SourceKind::Simulate,
        },
        _ => SourceKind::Simulate,
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
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
        e.stats.push(&[1, 2, 3, 4]); // deterministic live bytes for the top-up
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

    #[test]
    fn recording_writes_one_frame_per_tick_and_skips_paused() {
        let dir = std::env::temp_dir().join(format!("babble-rec-{}", std::process::id()));
        let path = dir.join("nested").join("s.bbrec");
        let mut e = Engine::new(SourceKind::Simulate);
        let (tx, rx) = std::sync::mpsc::channel();
        e.apply(ControlMsg::StartRecording(path.clone(), tx));
        assert_eq!(rx.recv().unwrap().unwrap(), path.to_string_lossy());
        std::thread::sleep(Duration::from_millis(120));
        let before = e.tick().total_bytes;
        std::thread::sleep(Duration::from_millis(120));
        let dto = e.tick();
        assert!(dto.total_bytes > before && dto.total_bytes > 0);
        assert_eq!(dto.recording.as_deref(), Some(&*path.to_string_lossy()));
        assert_eq!(dto.recording_bytes, dto.total_bytes, "every drained byte is recorded");
        e.apply(ControlMsg::SetPaused(true));
        std::thread::sleep(Duration::from_millis(120));
        let paused = e.tick();
        assert_eq!(paused.recording_bytes, dto.recording_bytes, "paused ticks write nothing");
        e.apply(ControlMsg::StopRecording);
        assert!(e.tick().recording.is_none());

        let r = bbrec::RecordingReader::open(&path).unwrap();
        assert!(r.header.source.starts_with("simulator"), "source label: {}", r.header.source);
        assert_eq!(r.header.trial_min_bits, TRIAL_MIN_BITS);
        assert_eq!(r.header.walk_k, 0, "fresh engine: no trials yet");
        let frames = r.frames().unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames.iter().map(|f| f.1.len() as u64).sum::<u64>(), dto.total_bytes);
        assert!(frames[0].0 < frames[1].0);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn recording_header_carries_walk_state_and_reset_stops_it() {
        let dir = std::env::temp_dir().join(format!("babble-rec2-{}", std::process::id()));
        let path = dir.join("s.bbrec");
        let mut e = Engine::new(SourceKind::Simulate);
        // Accumulate some trials first so the walk has history.
        for _ in 0..3 {
            std::thread::sleep(Duration::from_millis(110));
            e.tick();
        }
        let ws = e.stats.walk_state();
        assert!(ws.k >= 1);
        let (tx, rx) = std::sync::mpsc::channel();
        e.apply(ControlMsg::StartRecording(path.clone(), tx));
        rx.recv().unwrap().unwrap();
        let r = bbrec::RecordingReader::open(&path).unwrap();
        assert_eq!(r.header.walk_k, ws.k);
        assert!((r.header.walk_cum - ws.cum).abs() < 1e-12);
        // a bad path leaves the current recording running
        let (tx2, rx2) = std::sync::mpsc::channel();
        e.apply(ControlMsg::StartRecording(PathBuf::from("/proc/nonexistent-dir/x.bbrec"), tx2));
        assert!(rx2.recv().unwrap().is_err());
        assert!(e.recording().is_some(), "failed start must not stop the running recording");
        e.apply(ControlMsg::Reset);
        assert!(e.recording().is_none(), "reset ends the recording");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn engine_accumulates_bytes_over_ticks() {
        let mut e = Engine::new(SourceKind::Simulate);
        std::thread::sleep(Duration::from_millis(120));
        let a = e.tick();
        std::thread::sleep(Duration::from_millis(120));
        let b = e.tick();
        assert!(b.total_bytes >= a.total_bytes);
        assert!(b.total_bytes > 0);
    }

    #[test]
    fn pause_stops_accumulation() {
        let mut e = Engine::new(SourceKind::Simulate);
        std::thread::sleep(Duration::from_millis(120));
        let a0 = e.tick();
        assert!(a0.total_bytes > 0);
        e.apply(ControlMsg::SetPaused(true));
        let a = e.tick();
        std::thread::sleep(Duration::from_millis(120));
        let b = e.tick();
        assert_eq!(a.total_bytes, b.total_bytes);
    }

    #[test]
    fn resolve_kind_falls_back_to_simulate() {
        assert!(matches!(resolve_kind("bad", None, 9600), SourceKind::BadRng));
        // "auto" returns Serial if a device is present, else Simulate — both valid.
        assert!(matches!(resolve_kind("auto", None, 9600), SourceKind::Simulate | SourceKind::Serial { .. }));
        // "serial" with an explicit path returns Serial with that path.
        assert!(matches!(resolve_kind("serial", Some("/dev/null".into()), 9600), SourceKind::Serial { .. }));
    }
}
