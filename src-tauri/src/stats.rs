//! Sliding-window statistics engine.
//!
//! Bytes are pushed in as they arrive from the entropy source. A fixed-size
//! window (`VecDeque<u8>`) of the most recent bytes is maintained, along with
//! an incrementally-updated 256-bin histogram and a running popcount. Each
//! frame the UI calls [`Stats::snapshot`] to compute the derived metrics.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::math::{chi_square_sf, normal_sf, normal_two_sided};

// Two-sided significance thresholds (σ) for the coherence envelopes.
pub const Z95: f64 = 1.959_963_98;
pub const Z99: f64 = 2.575_829_30;
pub const Z999: f64 = 3.290_526_73;

/// How often a coherence "trial" is finalized. Decoupling from throughput keeps
/// the random walk readable regardless of how fast the device streams.
const TRIAL_INTERVAL: Duration = Duration::from_millis(100);
const TRIAL_MIN_BITS: u64 = 2048;

/// Rolling buffer kept for the in-TUI authenticity audit (press `a`).
const AUDIT_CAP: usize = 2 * 1024 * 1024;

/// One test result, ready to render with a verdict.
#[derive(Clone, Copy)]
pub struct Metric {
    pub value: f64,
    /// 0.0 = worst, 1.0 = ideal — drives the gauge fill and color.
    pub quality: f64,
    pub verdict: Verdict,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Verdict {
    Pass,
    Warn,
    Fail,
    Warmup,
}

/// Which significance envelope the cumulative deviation has crossed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Band {
    /// Inside the 95% envelope — ordinary random walk.
    Inside,
    /// Beyond 95% (p < 0.05).
    P95,
    /// Beyond 99% (p < 0.01).
    P99,
    /// Beyond 99.9% (p < 0.001) — strong anomalous coherence.
    P999,
}

impl Band {
    /// Classify a signed σ deviation into a band.
    pub fn of(sigma: f64) -> Band {
        let a = sigma.abs();
        if a >= Z999 {
            Band::P999
        } else if a >= Z99 {
            Band::P99
        } else if a >= Z95 {
            Band::P95
        } else {
            Band::Inside
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Band::Inside => "in-band",
            Band::P95 => "95%",
            Band::P99 => "99%",
            Band::P999 => "99.9%",
        }
    }
}

/// A logged excursion of the cumulative-deviation walk outside the 95% band.
#[derive(Clone, Copy, Debug)]
pub struct AnomalyEvent {
    /// Seconds since session start when the excursion began.
    pub at_secs: f64,
    /// Peak signed σ reached during the excursion.
    pub peak_sigma: f64,
    /// Strongest band reached.
    pub band: Band,
    /// Duration in seconds (grows while `ongoing`).
    pub duration_secs: f64,
    /// True while the walk is still outside the band.
    pub ongoing: bool,
}

/// A full computed view of the current window, produced once per frame.
pub struct Snapshot {
    pub window_len: usize,
    pub total_bytes: u64,
    pub throughput_bps: f64,

    pub shannon: Metric,    // bits / byte, ideal 8.0
    pub min_entropy: Metric, // bits / byte, ideal 8.0
    pub monobit: Metric,    // proportion of 1-bits, ideal 0.5
    pub chi_square: Metric, // goodness of fit vs uniform
    pub serial_corr: Metric, // lag-1 serial correlation, ideal 0.0

    pub monobit_p: f64,
    pub chi_square_p: f64,

    /// 256-bin byte histogram (counts) for the distribution panel.
    pub histogram: [u32; 256],
    pub hist_expected: f64,

    /// Most-recent bytes (newest last) for the bitstream visualizer.
    pub recent: Vec<u8>,

    // --- Coherence (cumulative-deviation random walk) ---
    /// Current normalized deviation σ = C_k / √k (signed). ~N(0,1) under null.
    pub coherence_sigma: f64,
    /// Band the current σ falls in.
    pub coherence_band: Band,
    /// Number of finalized trials so far (k).
    pub trial_count: u64,
    /// Recent walk points as (k, C_k) for the coherence chart.
    pub walk: Vec<(f64, f64)>,
    /// Recent anomaly events, most-recent first (includes the ongoing one).
    pub anomalies: Vec<AnomalyEvent>,
}

pub struct Stats {
    window: VecDeque<u8>,
    capacity: usize,
    hist: [u32; 256],
    ones: u64, // popcount across the window
    pub total_bytes: u64,

    // Throughput tracking.
    recent_counts: VecDeque<(std::time::Instant, u64)>,

    // Entropy-vs-time curve (Shannon bits/byte history).
    pub curve: VecDeque<f64>,
    curve_cap: usize,

    // Coherence: GCP-style cumulative-deviation random walk of monobit-Z.
    start: Instant,
    trial_ones: u64,
    trial_bits: u64,
    trial_last: Instant,
    cum: f64,          // cumulative sum of per-trial z-scores (the walk)
    trial_count: u64,  // k
    walk: VecDeque<(f64, f64)>,
    walk_cap: usize,
    current_event: Option<AnomalyEvent>,
    events: VecDeque<AnomalyEvent>,
    events_cap: usize,
    /// Set true on the frame a new ≥99.9% excursion begins (drives the bell).
    pub alert_pending: bool,

    // Rolling sample for the authenticity audit.
    audit_buf: VecDeque<u8>,
}

impl Stats {
    pub fn new(capacity: usize) -> Self {
        let now = Instant::now();
        Stats {
            window: VecDeque::with_capacity(capacity),
            capacity,
            hist: [0; 256],
            ones: 0,
            total_bytes: 0,
            recent_counts: VecDeque::new(),
            curve: VecDeque::new(),
            curve_cap: 512,
            start: now,
            trial_ones: 0,
            trial_bits: 0,
            trial_last: now,
            cum: 0.0,
            trial_count: 0,
            walk: VecDeque::new(),
            walk_cap: 600,
            current_event: None,
            events: VecDeque::new(),
            events_cap: 64,
            alert_pending: false,
            audit_buf: VecDeque::with_capacity(AUDIT_CAP),
        }
    }

    /// Bytes currently available for an audit.
    pub fn audit_len(&self) -> usize {
        self.audit_buf.len()
    }

    /// Contiguous copy of the rolling audit sample (newest bytes last).
    pub fn audit_sample(&self) -> Vec<u8> {
        self.audit_buf.iter().copied().collect()
    }

    /// Resize the window, discarding the oldest bytes if shrinking.
    pub fn set_capacity(&mut self, capacity: usize) {
        self.capacity = capacity.max(256);
        while self.window.len() > self.capacity {
            if let Some(b) = self.window.pop_front() {
                self.hist[b as usize] -= 1;
                self.ones -= b.count_ones() as u64;
            }
        }
    }

    pub fn reset(&mut self) {
        let now = Instant::now();
        self.window.clear();
        self.hist = [0; 256];
        self.ones = 0;
        self.total_bytes = 0;
        self.recent_counts.clear();
        self.curve.clear();
        self.start = now;
        self.trial_ones = 0;
        self.trial_bits = 0;
        self.trial_last = now;
        self.cum = 0.0;
        self.trial_count = 0;
        self.walk.clear();
        self.current_event = None;
        self.events.clear();
        self.alert_pending = false;
        self.audit_buf.clear();
    }

    /// Ingest a freshly-read chunk of bytes.
    pub fn push(&mut self, data: &[u8]) {
        for &b in data {
            if self.window.len() == self.capacity {
                if let Some(old) = self.window.pop_front() {
                    self.hist[old as usize] -= 1;
                    self.ones -= old.count_ones() as u64;
                }
            }
            self.window.push_back(b);
            self.hist[b as usize] += 1;
            self.ones += b.count_ones() as u64;
        }
        self.total_bytes += data.len() as u64;

        // Accumulate this chunk into the in-progress coherence trial.
        let mut chunk_ones = 0u64;
        for &b in data {
            chunk_ones += b.count_ones() as u64;
        }
        self.trial_ones += chunk_ones;
        self.trial_bits += (data.len() as u64) * 8;

        // Maintain the rolling audit sample.
        self.audit_buf.extend(data.iter().copied());
        let over = self.audit_buf.len().saturating_sub(AUDIT_CAP);
        if over > 0 {
            self.audit_buf.drain(0..over);
        }

        let now = std::time::Instant::now();
        self.recent_counts.push_back((now, self.total_bytes));
        while let Some(&(t, _)) = self.recent_counts.front() {
            if now.duration_since(t).as_secs_f64() > 3.0 {
                self.recent_counts.pop_front();
            } else {
                break;
            }
        }
    }

    /// Finalize a coherence trial if the interval has elapsed and enough bits
    /// have accumulated. Each trial contributes a standard-normal z-score to the
    /// cumulative-deviation random walk, and excursions past the 95/99/99.9%
    /// envelopes are tracked as anomaly events. Call once per frame.
    pub fn tick_trials(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.trial_last) < TRIAL_INTERVAL {
            return;
        }
        if self.trial_bits < TRIAL_MIN_BITS {
            return; // wait for more data; keep accumulating
        }

        let n = self.trial_bits as f64;
        // Monobit z for this trial: mean n/2, variance n/4.
        let z = (self.trial_ones as f64 - n / 2.0) / (n / 4.0).sqrt();
        self.cum += z;
        self.trial_count += 1;
        self.trial_ones = 0;
        self.trial_bits = 0;
        self.trial_last = now;

        let k = self.trial_count as f64;
        self.walk.push_back((k, self.cum));
        while self.walk.len() > self.walk_cap {
            self.walk.pop_front();
        }

        // Normalized deviation: σ = C_k / √k  (~N(0,1) under pure randomness).
        let sigma = self.cum / k.sqrt();
        let band = Band::of(sigma);
        let elapsed = self.start.elapsed().as_secs_f64();

        if band == Band::Inside {
            // Excursion ends (if one was active) — finalize and log it.
            if let Some(mut ev) = self.current_event.take() {
                ev.duration_secs = elapsed - ev.at_secs;
                ev.ongoing = false;
                self.events.push_front(ev);
                while self.events.len() > self.events_cap {
                    self.events.pop_back();
                }
            }
        } else {
            match &mut self.current_event {
                // Excursion begins.
                None => {
                    if band == Band::P999 {
                        self.alert_pending = true;
                    }
                    self.current_event = Some(AnomalyEvent {
                        at_secs: elapsed,
                        peak_sigma: sigma,
                        band,
                        duration_secs: 0.0,
                        ongoing: true,
                    });
                }
                // Excursion continues — update peak / band / duration.
                Some(ev) => {
                    if sigma.abs() > ev.peak_sigma.abs() {
                        ev.peak_sigma = sigma;
                    }
                    if band_rank(band) > band_rank(ev.band) {
                        if band == Band::P999 && ev.band != Band::P999 {
                            self.alert_pending = true;
                        }
                        ev.band = band;
                    }
                    ev.duration_secs = elapsed - ev.at_secs;
                }
            }
        }
    }

    /// Take the pending-bell flag, clearing it.
    pub fn take_alert(&mut self) -> bool {
        std::mem::take(&mut self.alert_pending)
    }

    /// Append the current Shannon entropy to the time-series curve. Called once
    /// per frame so the curve advances at a steady cadence.
    pub fn tick_curve(&mut self, shannon_bits: f64) {
        if self.window.len() < 256 {
            return;
        }
        self.curve.push_back(shannon_bits);
        while self.curve.len() > self.curve_cap {
            self.curve.pop_front();
        }
    }

    fn throughput(&self) -> f64 {
        if self.recent_counts.len() < 2 {
            return 0.0;
        }
        let (t0, c0) = *self.recent_counts.front().unwrap();
        let (t1, c1) = *self.recent_counts.back().unwrap();
        let dt = t1.duration_since(t0).as_secs_f64();
        if dt <= 0.0 {
            0.0
        } else {
            (c1 - c0) as f64 / dt
        }
    }

    pub fn snapshot(&self) -> Snapshot {
        let n = self.window.len();
        let nf = n as f64;
        let warming = n < 4096;

        // --- Shannon entropy (bits / byte) ---
        let mut shannon = 0.0;
        let mut max_p = 0.0_f64;
        if n > 0 {
            for &c in self.hist.iter() {
                if c > 0 {
                    let p = c as f64 / nf;
                    shannon -= p * p.log2();
                    if p > max_p {
                        max_p = p;
                    }
                }
            }
        }
        let min_entropy = if max_p > 0.0 { -max_p.log2() } else { 0.0 };

        // --- Monobit: proportion of 1-bits ---
        let total_bits = nf * 8.0;
        let prop_ones = if total_bits > 0.0 {
            self.ones as f64 / total_bits
        } else {
            0.5
        };
        // S_obs = |ones - zeros| / sqrt(nbits); p = erfc(S_obs / sqrt2)
        let monobit_p = if total_bits > 0.0 {
            let s = (2.0 * self.ones as f64 - total_bits) / total_bits.sqrt();
            normal_two_sided(s)
        } else {
            1.0
        };

        // --- Chi-square goodness of fit vs uniform (256 bins) ---
        let expected = nf / 256.0;
        let mut chi = 0.0;
        if expected > 0.0 {
            for &c in self.hist.iter() {
                let d = c as f64 - expected;
                chi += d * d / expected;
            }
        }
        let chi_p = chi_square_sf(chi, 255.0);

        // --- Lag-1 serial correlation coefficient ---
        let serial = self.serial_correlation();

        // --- Recent bytes for the bitstream rain ---
        let take = 1024.min(n);
        let recent: Vec<u8> = self.window.iter().rev().take(take).rev().copied().collect();

        // --- Coherence (cumulative-deviation walk) ---
        let k = self.trial_count as f64;
        let coherence_sigma = if k > 0.0 { self.cum / k.sqrt() } else { 0.0 };
        let coherence_band = Band::of(coherence_sigma);
        let walk: Vec<(f64, f64)> = self.walk.iter().copied().collect();
        let mut anomalies: Vec<AnomalyEvent> = Vec::new();
        if let Some(ev) = &self.current_event {
            anomalies.push(*ev);
        }
        anomalies.extend(self.events.iter().copied());

        Snapshot {
            window_len: n,
            total_bytes: self.total_bytes,
            throughput_bps: self.throughput(),
            shannon: metric_entropy(shannon, warming),
            min_entropy: metric_min_entropy(max_p, n, min_entropy, warming),
            monobit: metric_monobit(prop_ones, monobit_p, warming),
            chi_square: metric_chi(chi, chi_p, warming),
            serial_corr: metric_serial(serial, n, warming),
            monobit_p,
            chi_square_p: chi_p,
            histogram: self.hist,
            hist_expected: expected,
            recent,
            coherence_sigma,
            coherence_band,
            trial_count: self.trial_count,
            walk,
            anomalies,
        }
    }

    fn serial_correlation(&self) -> f64 {
        let n = self.window.len();
        if n < 2 {
            return 0.0;
        }
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut sum_prod = 0.0f64;
        let mut prev = self.window[0] as f64;
        sum += prev;
        sum_sq += prev * prev;
        for i in 1..n {
            let cur = self.window[i] as f64;
            sum += cur;
            sum_sq += cur * cur;
            sum_prod += prev * cur;
            prev = cur;
        }
        let nf = n as f64;
        let num = nf * sum_prod - sum * sum;
        let den = nf * sum_sq - sum * sum;
        if den.abs() < 1e-9 {
            0.0
        } else {
            num / den
        }
    }
}

fn band_rank(b: Band) -> u8 {
    match b {
        Band::Inside => 0,
        Band::P95 => 1,
        Band::P99 => 2,
        Band::P999 => 3,
    }
}

// --- Metric → quality/verdict mapping --------------------------------------

fn warmup() -> Metric {
    Metric {
        value: f64::NAN,
        quality: 0.0,
        verdict: Verdict::Warmup,
    }
}

fn metric_entropy(bits: f64, warming: bool) -> Metric {
    if warming {
        return Metric { value: bits, quality: (bits / 8.0).clamp(0.0, 1.0), verdict: Verdict::Warmup };
    }
    let deficit = 8.0 - bits;
    let verdict = if deficit < 0.02 {
        Verdict::Pass
    } else if deficit < 0.1 {
        Verdict::Warn
    } else {
        Verdict::Fail
    };
    Metric { value: bits, quality: (bits / 8.0).clamp(0.0, 1.0), verdict }
}

/// Min-entropy verdict. Unlike Shannon, min-entropy is *expected* to sit below
/// 8.0: it's `-log2(max pᵢ)`, and the busiest of 256 bins naturally rises a few
/// σ above the mean count purely by chance. So we don't compare to 8.0 — we ask
/// whether the tallest bin is improbably tall given the window size. Under
/// uniformity each bin ≈ Normal(λ, λ) with λ = N/256, and the max over 256 bins
/// sits near z ≈ 3.3; only z well past that signals real bias.
fn metric_min_entropy(max_p: f64, n: usize, bits: f64, warming: bool) -> Metric {
    if warming || n < 256 {
        return Metric {
            value: bits,
            quality: (bits / 8.0).clamp(0.0, 1.0),
            verdict: Verdict::Warmup,
        };
    }
    let nf = n as f64;
    let lambda = nf / 256.0; // expected count per byte value
    let sd = lambda.sqrt();
    let max_count = max_p * nf;
    let z = if sd > 0.0 { (max_count - lambda) / sd } else { 0.0 };
    // Max of 256 ~N(0,1) bins typically peaks near z≈3.3; >4.5 is unlikely,
    // >5.5 indicates a genuinely over-represented value.
    let verdict = if z < 4.5 {
        Verdict::Pass
    } else if z < 5.5 {
        Verdict::Warn
    } else {
        Verdict::Fail
    };
    let quality = (1.0 - z / 6.0).clamp(0.0, 1.0);
    Metric { value: bits, quality, verdict }
}

fn metric_monobit(prop: f64, p: f64, warming: bool) -> Metric {
    if warming {
        return Metric { value: prop, quality: 1.0 - (prop - 0.5).abs() * 4.0, verdict: Verdict::Warmup };
    }
    let verdict = if p >= 0.01 { Verdict::Pass } else if p >= 0.001 { Verdict::Warn } else { Verdict::Fail };
    // quality: how close prop is to 0.5 (within +/-0.01 is essentially perfect)
    let quality = (1.0 - (prop - 0.5).abs() / 0.01).clamp(0.0, 1.0);
    Metric { value: prop, quality, verdict }
}

fn metric_chi(chi: f64, p: f64, warming: bool) -> Metric {
    if warming {
        return warmup();
    }
    // For 255 df, expected chi ~= 255. A healthy p-value lives in (0.01, 0.99).
    let verdict = if p > 0.01 && p < 0.99 {
        Verdict::Pass
    } else if p > 0.001 && p < 0.999 {
        Verdict::Warn
    } else {
        Verdict::Fail
    };
    // quality peaks when p ~ 0.5 (chi near its expected value).
    let quality = (1.0 - (p - 0.5).abs() * 2.0).clamp(0.0, 1.0);
    let _ = chi;
    Metric { value: chi, quality, verdict }
}

fn metric_serial(corr: f64, n: usize, warming: bool) -> Metric {
    if warming {
        return Metric { value: corr, quality: 1.0 - corr.abs() * 20.0, verdict: Verdict::Warmup };
    }
    // Under H0 the correlation is ~N(0, 1/n). Convert to a z-score.
    let z = corr.abs() * (n as f64).sqrt();
    let p = normal_sf(z) * 2.0;
    let verdict = if p >= 0.01 { Verdict::Pass } else if p >= 0.001 { Verdict::Warn } else { Verdict::Fail };
    let quality = (1.0 - corr.abs() / 0.02).clamp(0.0, 1.0);
    Metric { value: corr, quality, verdict }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Perfectly uniform stream (every byte value equally often) must give
    /// the ideal 8.0 bits/byte, zero chi-square, and a balanced monobit.
    #[test]
    fn uniform_stream_is_ideal() {
        let mut s = Stats::new(1 << 16);
        let mut block = Vec::with_capacity(256 * 64);
        for _ in 0..64 {
            for b in 0u16..256 {
                block.push(b as u8);
            }
        }
        s.push(&block);
        let snap = s.snapshot();
        assert!((snap.shannon.value - 8.0).abs() < 1e-9, "shannon={}", snap.shannon.value);
        assert!((snap.min_entropy.value - 8.0).abs() < 1e-9);
        assert!(snap.chi_square.value < 1e-6, "chi={}", snap.chi_square.value);
        assert!((snap.monobit.value - 0.5).abs() < 1e-9);
        assert_eq!(snap.shannon.verdict, Verdict::Pass);
    }

    /// A constant stream has zero entropy and a maxed-out chi-square.
    #[test]
    fn constant_stream_is_dead() {
        let mut s = Stats::new(1 << 16);
        s.push(&[0xAA; 50_000]);
        let snap = s.snapshot();
        assert!(snap.shannon.value < 1e-9, "shannon={}", snap.shannon.value);
        assert!(snap.min_entropy.value < 1e-9);
        assert_eq!(snap.shannon.verdict, Verdict::Fail);
        assert_eq!(snap.min_entropy.verdict, Verdict::Fail);
    }

    /// A healthy RNG window has min-entropy below 8.0 (the busiest of 256 bins
    /// always rises a few σ by chance) yet must still PASS — that was the bug
    /// behind a red "Min-entropy FAIL" on real hardware.
    #[test]
    fn min_entropy_passes_for_healthy_window() {
        let mut s = Stats::new(1 << 16);
        let mut x: u64 = 0x1234_5678_9abc_def0;
        let data: Vec<u8> = (0..70_000)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 24) as u8
            })
            .collect();
        s.push(&data);
        let snap = s.snapshot();
        assert!(
            snap.min_entropy.value < 8.0 && snap.min_entropy.value > 7.4,
            "min-entropy {} outside expected healthy range",
            snap.min_entropy.value
        );
        assert_eq!(snap.min_entropy.verdict, Verdict::Pass);
    }

    /// The biased LCG used by `--bad` must measurably fail the entropy bar.
    #[test]
    fn biased_source_fails() {
        let mut s = Stats::new(1 << 16);
        let mut lcg: u32 = 0x1234_5678;
        let data: Vec<u8> = (0..60_000)
            .map(|_| {
                lcg = lcg.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                ((lcg >> 16) as u8) & 0b0011_1111
            })
            .collect();
        s.push(&data);
        let snap = s.snapshot();
        // Only 6 usable bits and a skewed distribution → well under 8.
        assert!(snap.shannon.value < 7.0, "shannon={}", snap.shannon.value);
        assert_eq!(snap.shannon.verdict, Verdict::Fail);
    }

    #[test]
    fn band_classification() {
        assert_eq!(Band::of(0.0), Band::Inside);
        assert_eq!(Band::of(1.5), Band::Inside);
        assert_eq!(Band::of(2.0), Band::P95);
        assert_eq!(Band::of(-2.0), Band::P95);
        assert_eq!(Band::of(2.7), Band::P99);
        assert_eq!(Band::of(-3.5), Band::P999);
    }

    /// A strongly biased stream must drive the cumulative-deviation walk out of
    /// the band and register an anomalous-coherence excursion.
    #[test]
    fn biased_stream_triggers_coherence_anomaly() {
        let mut s = Stats::new(1 << 16);
        let mut lcg: u32 = 0x1234_5678;
        // A few trials, each well past TRIAL_MIN_BITS, spaced past TRIAL_INTERVAL.
        for _ in 0..3 {
            let data: Vec<u8> = (0..1024)
                .map(|_| {
                    lcg = lcg.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    ((lcg >> 16) as u8) & 0b0011_1111 // top two bits always 0 → bit-biased
                })
                .collect();
            s.push(&data);
            std::thread::sleep(TRIAL_INTERVAL + Duration::from_millis(15));
            s.tick_trials();
        }
        let snap = s.snapshot();
        assert!(snap.trial_count >= 1, "no trials finalized");
        assert!(
            snap.coherence_sigma.abs() > Z95,
            "sigma {} did not exceed the 95% band",
            snap.coherence_sigma
        );
        assert!(!snap.anomalies.is_empty(), "no anomaly recorded");
    }

    /// A uniform stream keeps every per-trial z at exactly zero, so the walk
    /// never leaves the band.
    #[test]
    fn uniform_stream_stays_in_band() {
        let mut s = Stats::new(1 << 16);
        for _ in 0..3 {
            // Equal 0s and 1s per byte (0x0F) → trial z is exactly 0.
            s.push(&[0x0F; 1024]);
            std::thread::sleep(TRIAL_INTERVAL + Duration::from_millis(15));
            s.tick_trials();
        }
        let snap = s.snapshot();
        assert!(snap.trial_count >= 1);
        assert_eq!(snap.coherence_band, Band::Inside);
        assert!(snap.anomalies.is_empty());
    }

    #[test]
    fn window_eviction_keeps_counts_consistent() {
        let mut s = Stats::new(1000);
        s.push(&[1u8; 2000]); // overflow the window twice over
        let snap = s.snapshot();
        assert_eq!(snap.window_len, 1000);
        assert_eq!(snap.histogram[1], 1000);
        assert_eq!(snap.histogram.iter().sum::<u32>(), 1000);
    }
}
