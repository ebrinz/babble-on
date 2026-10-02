//! Seed bundles: the bytes a generation would have been seeded with, drawn
//! bank-first (destructively) exactly as `generate` draws them, plus the
//! provenance of every anomaly event those bytes came from.
//!
//! Two files side by side: `<stem>.seed.bin` (raw bytes) and
//! `<stem>.seed.json` (this struct, minus the bytes). The harness's
//! `EntropyTape.from_seed(path)` loads both.

use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Mirrors `src-tauri::bank::ProvenanceTag`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SeedTag {
    pub at_secs: f64,
    pub peak_sigma: f64,
    pub band: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SeedBundle {
    /// Byte count in the `.seed.bin` sibling.
    pub bytes: usize,
    /// Share of the bytes that came from the anomaly bank (0..=1).
    pub bank_fraction: f64,
    pub tags: Vec<SeedTag>,
    pub created_at_ms: u64,
    pub source: String,
    /// `"in_band"`, `"out_band"`, `"mixed"` or `"live"` — what the app knew
    /// about the bytes when it exported them (the harness may relabel from a
    /// recording when one covers the same session).
    pub label: String,
}

impl SeedBundle {
    /// Label from the bank fraction: all-bank ⇒ out_band, none ⇒ live
    /// (in-band stream bytes), anything else ⇒ mixed.
    pub fn label_for(bank_fraction: f64) -> &'static str {
        if bank_fraction >= 0.999 {
            "out_band"
        } else if bank_fraction <= 0.0 {
            "live"
        } else {
            "mixed"
        }
    }

    pub fn paths(stem: &Path) -> (PathBuf, PathBuf) {
        let s = stem.to_string_lossy();
        (PathBuf::from(format!("{s}.seed.bin")), PathBuf::from(format!("{s}.seed.json")))
    }

    /// Write both files. `stem` is the path without the `.seed.*` suffix.
    pub fn write(&self, stem: &Path, data: &[u8]) -> io::Result<(PathBuf, PathBuf)> {
        if data.len() != self.bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("bundle says {} bytes, got {}", self.bytes, data.len()),
            ));
        }
        let (bin, json) = Self::paths(stem);
        if let Some(dir) = bin.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(&bin, data)?;
        std::fs::write(&json, serde_json::to_vec_pretty(self).map_err(io::Error::other)?)?;
        Ok((bin, json))
    }

    pub fn read(stem: &Path) -> io::Result<(Self, Vec<u8>)> {
        let (bin, json) = Self::paths(stem);
        let meta: SeedBundle = serde_json::from_slice(&std::fs::read(&json)?)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let data = std::fs::read(&bin)?;
        if data.len() != meta.bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{} holds {} bytes, json says {}", bin.display(), data.len(), meta.bytes),
            ));
        }
        Ok((meta, data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_then_read() {
        let dir = std::env::temp_dir().join(format!("bbrec-seed-{}", std::process::id()));
        let stem = dir.join("exports").join("anomaly-01");
        let b = SeedBundle {
            bytes: 5,
            bank_fraction: 0.6,
            tags: vec![SeedTag { at_secs: 12.5, peak_sigma: 3.1, band: "99%".into() }],
            created_at_ms: 1,
            source: "simulate".into(),
            label: SeedBundle::label_for(0.6).into(),
        };
        let (bin, json) = b.write(&stem, &[9, 8, 7, 6, 5]).unwrap();
        assert!(bin.ends_with("anomaly-01.seed.bin") && json.ends_with("anomaly-01.seed.json"));
        let (meta, data) = SeedBundle::read(&stem).unwrap();
        assert_eq!(meta, b);
        assert_eq!(data, vec![9, 8, 7, 6, 5]);
        assert_eq!(meta.label, "mixed");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn labels() {
        assert_eq!(SeedBundle::label_for(1.0), "out_band");
        assert_eq!(SeedBundle::label_for(0.0), "live");
        assert_eq!(SeedBundle::label_for(0.3), "mixed");
    }

    #[test]
    fn length_mismatch_is_an_error() {
        let b = SeedBundle { bytes: 2, bank_fraction: 0.0, tags: vec![], created_at_ms: 0, source: "x".into(), label: "live".into() };
        assert!(b.write(Path::new("/nonexistent-dir-should-not-be-created/x"), &[1]).is_err());
    }
}
