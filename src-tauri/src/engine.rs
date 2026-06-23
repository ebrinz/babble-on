use crate::source::{Source, SourceKind, SourceStatus, autodetect, drain_into};
use crate::stats::Stats;
use crate::dto::SnapshotDto;

pub enum ControlMsg {
    SetSource(SourceKind),
    Reset,
    SetWindow(usize),
    SetPaused(bool),
}

pub struct Engine {
    source: Source,
    stats: Stats,
    paused: bool,
    status: String,
}

impl Engine {
    pub fn new(initial: SourceKind) -> Self {
        let source = Source::spawn(initial);
        Engine { source, stats: Stats::new(1 << 16), paused: false, status: "connecting".into() }
    }

    pub fn apply(&mut self, msg: ControlMsg) {
        match msg {
            ControlMsg::SetSource(kind) => { self.source = Source::spawn(kind); self.status = "connecting".into(); }
            ControlMsg::Reset => self.stats.reset(),
            ControlMsg::SetWindow(n) => self.stats.set_capacity(n),
            ControlMsg::SetPaused(p) => self.paused = p,
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
        if !self.paused {
            drain_into(&self.source.data_rx, &mut self.stats);
            self.stats.tick_trials();
        } else {
            // Keep the unbounded channel from growing while paused, without
            // feeding the bytes into the (frozen) stats window.
            while self.source.data_rx.try_recv().is_ok() {}
        }
        let snap = self.stats.snapshot();
        if !self.paused {
            self.stats.tick_curve(snap.shannon.value);
        }
        SnapshotDto::from_snapshot(&snap, self.source.label.clone(), self.status.clone())
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

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
