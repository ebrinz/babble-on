//! `.bbrec` stream recordings: the raw byte stream the observatory saw, framed
//! per engine tick with a monotonic timestamp, so the harness can replay the
//! coherence walk offline and label every byte with the band it arrived in.
//!
//! Layout (all integers little-endian):
//!   magic   8 bytes  "BBREC001"
//!   hlen    u32      header JSON length
//!   header  hlen bytes of JSON (`RecordingHeader`)
//!   frames  repeated: t_ns u64 | len u32 | len bytes
//!
//! `t_ns` is nanoseconds since the recording started. A frame holds exactly
//! the bytes one engine tick drained, so frames align with the walk's trial
//! clock (`trial_interval_ms` / `trial_min_bits` in the header are the
//! engine's constants at recording time, for a faithful replay).

use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

pub const MAGIC: &[u8; 8] = b"BBREC001";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RecordingHeader {
    /// Source label as the UI shows it (device path, "simulate", "bad", "anu").
    pub source: String,
    /// Wall-clock start, ms since the Unix epoch.
    pub started_at_ms: u64,
    /// Engine coherence-trial constants in force while recording.
    pub trial_interval_ms: u64,
    pub trial_min_bits: u64,
    /// Free-form: app version, device, notes.
    #[serde(default)]
    pub notes: String,
}

pub struct RecordingWriter<W: Write> {
    w: BufWriter<W>,
    pub frames: u64,
    pub bytes: u64,
}

impl RecordingWriter<std::fs::File> {
    pub fn create(path: &Path, header: &RecordingHeader) -> io::Result<Self> {
        Self::new(std::fs::File::create(path)?, header)
    }
}

impl<W: Write> RecordingWriter<W> {
    pub fn new(inner: W, header: &RecordingHeader) -> io::Result<Self> {
        let mut w = BufWriter::new(inner);
        let h = serde_json::to_vec(header).map_err(io::Error::other)?;
        w.write_all(MAGIC)?;
        w.write_all(&(h.len() as u32).to_le_bytes())?;
        w.write_all(&h)?;
        Ok(RecordingWriter { w, frames: 0, bytes: 0 })
    }

    /// Append one tick's bytes. Empty frames are skipped (they carry nothing
    /// the replay needs; the walk only advances on data).
    pub fn frame(&mut self, t_ns: u64, data: &[u8]) -> io::Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        self.w.write_all(&t_ns.to_le_bytes())?;
        self.w.write_all(&(data.len() as u32).to_le_bytes())?;
        self.w.write_all(data)?;
        self.frames += 1;
        self.bytes += data.len() as u64;
        Ok(())
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.w.flush()
    }

    pub fn finish(mut self) -> io::Result<W> {
        self.w.flush()?;
        self.w.into_inner().map_err(|e| e.into_error())
    }
}

pub struct RecordingReader<R: Read> {
    r: BufReader<R>,
    pub header: RecordingHeader,
}

impl RecordingReader<std::fs::File> {
    pub fn open(path: &Path) -> io::Result<Self> {
        Self::new(std::fs::File::open(path)?)
    }
}

impl<R: Read> RecordingReader<R> {
    pub fn new(inner: R) -> io::Result<Self> {
        let mut r = BufReader::new(inner);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "not a .bbrec file (bad magic)"));
        }
        let mut len = [0u8; 4];
        r.read_exact(&mut len)?;
        let mut h = vec![0u8; u32::from_le_bytes(len) as usize];
        r.read_exact(&mut h)?;
        let header: RecordingHeader = serde_json::from_slice(&h)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(RecordingReader { r, header })
    }

    /// Next `(t_ns, bytes)` frame, or `None` at a clean end of file. A
    /// truncated trailing frame (app killed mid-write) is treated as the end.
    pub fn next_frame(&mut self) -> io::Result<Option<(u64, Vec<u8>)>> {
        if self.r.fill_buf()?.is_empty() {
            return Ok(None);
        }
        let mut t = [0u8; 8];
        if self.r.read_exact(&mut t).is_err() {
            return Ok(None);
        }
        let mut len = [0u8; 4];
        if self.r.read_exact(&mut len).is_err() {
            return Ok(None);
        }
        let mut data = vec![0u8; u32::from_le_bytes(len) as usize];
        if self.r.read_exact(&mut data).is_err() {
            return Ok(None);
        }
        Ok(Some((u64::from_le_bytes(t), data)))
    }

    pub fn frames(mut self) -> io::Result<Vec<(u64, Vec<u8>)>> {
        let mut out = Vec::new();
        while let Some(f) = self.next_frame()? {
            out.push(f);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header() -> RecordingHeader {
        RecordingHeader {
            source: "simulate".into(),
            started_at_ms: 1_700_000_000_000,
            trial_interval_ms: 100,
            trial_min_bits: 2048,
            notes: String::new(),
        }
    }

    #[test]
    fn round_trip_in_memory() {
        let mut w = RecordingWriter::new(Vec::new(), &header()).unwrap();
        w.frame(10, &[1, 2, 3]).unwrap();
        w.frame(20, &[]).unwrap(); // skipped
        w.frame(30, &[4]).unwrap();
        assert_eq!((w.frames, w.bytes), (2, 4));
        let buf = w.finish().unwrap();
        assert_eq!(&buf[..8], MAGIC);
        let r = RecordingReader::new(&buf[..]).unwrap();
        assert_eq!(r.header, header());
        assert_eq!(r.frames().unwrap(), vec![(10, vec![1, 2, 3]), (30, vec![4])]);
    }

    #[test]
    fn truncated_tail_is_end_of_file() {
        let mut w = RecordingWriter::new(Vec::new(), &header()).unwrap();
        w.frame(10, &[1, 2, 3]).unwrap();
        w.frame(30, &[4, 5, 6, 7]).unwrap();
        let mut buf = w.finish().unwrap();
        buf.truncate(buf.len() - 2);
        let got = RecordingReader::new(&buf[..]).unwrap().frames().unwrap();
        assert_eq!(got, vec![(10, vec![1, 2, 3])]);
    }

    #[test]
    fn rejects_bad_magic() {
        assert!(RecordingReader::new(&b"NOPE0001\0\0\0\0"[..]).is_err());
    }

    /// Reads the fixture the Python side wrote (harness/tests/fixtures).
    #[test]
    fn reads_python_fixture() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../harness/tests/fixtures/tiny.bbrec");
        let r = RecordingReader::open(Path::new(path)).unwrap();
        assert_eq!(r.header.source, "fixture");
        assert_eq!(r.header.trial_min_bits, 2048);
        let frames = r.frames().unwrap();
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0], (0, vec![0xde, 0xad]));
        assert_eq!(frames[2].0, 200_000_000);
        assert_eq!(frames[2].1.len(), 256);
    }
}
