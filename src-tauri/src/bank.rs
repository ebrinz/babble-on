//! Anomaly bank: raw entropy bytes harvested while the coherence walk is
//! outside its 95% envelope, spent destructively to seed diffusion latents.

use std::collections::VecDeque;

use serde::Serialize;

/// Bank capacity in bytes. A full 256-token latent needs 16 KiB
/// (seq_len 256 × embed_dim 16 × 4 bytes/gaussian), so this holds ~4 seeds.
pub const BANK_CAPACITY: usize = 64 * 1024;

/// Provenance of a banked run: the anomaly event whose bytes these are.
/// `at_secs`/`peak_sigma`/`band` mirror `stats::AnomalyEvent` (band as label).
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ProvenanceTag {
    pub at_secs: f64,
    pub peak_sigma: f64,
    pub band: &'static str,
}

/// Bounded FIFO byte pool with per-event provenance. Deposits append; when
/// over capacity the oldest bytes fall off. Withdrawals are destructive and
/// return the tags of every event represented in the bytes handed out.
pub struct AnomalyBank {
    buf: VecDeque<u8>,
    /// Per-event runs, FIFO: (bytes remaining in run, tag). Invariant:
    /// the run lengths always sum to `buf.len()`.
    runs: VecDeque<(usize, ProvenanceTag)>,
    capacity: usize,
}

impl AnomalyBank {
    pub fn new(capacity: usize) -> Self {
        AnomalyBank { buf: VecDeque::new(), runs: VecDeque::new(), capacity }
    }

    pub fn deposit(&mut self, bytes: &[u8], tag: ProvenanceTag) {
        if bytes.is_empty() {
            return;
        }
        self.buf.extend(bytes.iter().copied());
        match self.runs.back_mut() {
            // Same ongoing event across ticks: extend the run; the latest tag
            // carries the event's running peak/strongest band, so it wins.
            Some((len, last)) if last.at_secs == tag.at_secs => {
                *len += bytes.len();
                *last = tag;
            }
            _ => self.runs.push_back((bytes.len(), tag)),
        }
        if self.buf.len() > self.capacity {
            let mut excess = self.buf.len() - self.capacity;
            self.buf.drain(..excess);
            while excess > 0 {
                let (len, _) = self.runs.front_mut().expect("runs cover buf");
                if *len <= excess {
                    excess -= *len;
                    self.runs.pop_front();
                } else {
                    *len -= excess;
                    excess = 0;
                }
            }
        }
    }

    /// Destructive FIFO read of up to `n` bytes + the tags they came from.
    pub fn withdraw(&mut self, n: usize) -> (Vec<u8>, Vec<ProvenanceTag>) {
        let take = n.min(self.buf.len());
        let bytes: Vec<u8> = self.buf.drain(..take).collect();
        let mut tags = Vec::new();
        let mut left = take;
        while left > 0 {
            let (len, tag) = self.runs.front_mut().expect("runs cover buf");
            tags.push(*tag);
            if *len <= left {
                left -= *len;
                self.runs.pop_front();
            } else {
                *len -= left;
                left = 0;
            }
        }
        (bytes, tags)
    }

    /// (current fill, capacity) in bytes — for the UI meter.
    pub fn fill(&self) -> (usize, usize) {
        (self.buf.len(), self.capacity)
    }

    pub fn clear(&mut self) {
        self.buf.clear();
        self.runs.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tag(at: f64, sigma: f64) -> ProvenanceTag {
        ProvenanceTag { at_secs: at, peak_sigma: sigma, band: "95%" }
    }

    #[test]
    fn withdraw_is_fifo_and_destructive() {
        let mut b = AnomalyBank::new(BANK_CAPACITY);
        b.deposit(&[1, 2, 3], tag(1.0, 2.1));
        b.deposit(&[4, 5], tag(9.0, -2.6));
        let (bytes, tags) = b.withdraw(4);
        assert_eq!(bytes, vec![1, 2, 3, 4]);
        assert_eq!(tags.len(), 2);
        assert_eq!(tags[0].at_secs, 1.0);
        assert_eq!(tags[1].at_secs, 9.0);
        assert_eq!(b.fill().0, 1);
        let (bytes, tags) = b.withdraw(99);
        assert_eq!(bytes, vec![5]);
        assert_eq!(tags.len(), 1);
        assert_eq!(b.fill().0, 0);
    }

    #[test]
    fn same_event_deposits_merge_and_update_tag() {
        let mut b = AnomalyBank::new(BANK_CAPACITY);
        // Same event (same at_secs) deposited across two ticks; peak grows.
        b.deposit(&[1, 2], tag(1.0, 2.1));
        b.deposit(&[3, 4], tag(1.0, 3.4));
        let (bytes, tags) = b.withdraw(4);
        assert_eq!(bytes, vec![1, 2, 3, 4]);
        assert_eq!(tags.len(), 1, "one event → one tag");
        assert_eq!(tags[0].peak_sigma, 3.4, "latest tag wins (running peak)");
    }

    #[test]
    fn capacity_evicts_oldest() {
        let mut b = AnomalyBank::new(4);
        b.deposit(&[1, 2, 3], tag(1.0, 2.1));
        b.deposit(&[4, 5, 6], tag(9.0, 2.9));
        assert_eq!(b.fill(), (4, 4));
        let (bytes, tags) = b.withdraw(4);
        assert_eq!(bytes, vec![3, 4, 5, 6], "oldest bytes dropped");
        assert_eq!(tags.len(), 2, "partially-evicted run keeps its tag");
    }

    #[test]
    fn empty_and_clear() {
        let mut b = AnomalyBank::new(BANK_CAPACITY);
        let (bytes, tags) = b.withdraw(8);
        assert!(bytes.is_empty() && tags.is_empty());
        b.deposit(&[7; 10], tag(1.0, 2.1));
        b.clear();
        assert_eq!(b.fill().0, 0);
        assert!(b.withdraw(1).0.is_empty());
    }

    #[test]
    fn empty_deposit_is_a_noop() {
        let mut b = AnomalyBank::new(BANK_CAPACITY);
        b.deposit(&[], tag(1.0, 2.1));
        assert_eq!(b.fill().0, 0);
        assert!(b.withdraw(1).1.is_empty(), "no phantom tag");
    }
}
