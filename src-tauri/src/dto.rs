use serde::Serialize;
use crate::stats::{Snapshot, Metric, Verdict};

#[derive(Serialize)]
pub struct MetricDto { pub value: Option<f64>, pub quality: f64, pub verdict: &'static str }
#[derive(Serialize)]
pub struct AnomalyDto { pub at_secs: f64, pub peak_sigma: f64, pub band: &'static str, pub duration_secs: f64, pub ongoing: bool }

#[derive(Serialize)]
pub struct SnapshotDto {
    pub window_len: usize,
    pub total_bytes: u64,
    pub throughput_bps: f64,
    pub shannon: MetricDto,
    pub min_entropy: MetricDto,
    pub monobit: MetricDto,
    pub chi_square: MetricDto,
    pub serial_corr: MetricDto,
    pub monobit_p: Option<f64>,
    pub chi_square_p: Option<f64>,
    pub histogram: Vec<u32>,
    pub hist_expected: f64,
    pub recent: Vec<u8>,
    pub coherence_sigma: f64,
    pub coherence_band: &'static str,
    pub trial_count: u64,
    pub walk: Vec<(f64, f64)>,
    pub anomalies: Vec<AnomalyDto>,
    pub label: String,
    pub status: String,
}

fn nz(x: f64) -> Option<f64> { if x.is_finite() { Some(x) } else { None } }

/// Decimate the cumulative-deviation walk to at most `max` points for transport,
/// preserving the first and last points so the chart still spans vertex→now.
fn decimate_walk(walk: &[(f64, f64)], max: usize) -> Vec<(f64, f64)> {
    if walk.len() <= max {
        return walk.to_vec();
    }
    let stride = walk.len().div_ceil(max);
    let mut out: Vec<(f64, f64)> = walk.iter().step_by(stride).copied().collect();
    if let Some(&last) = walk.last() {
        if out.last() != Some(&last) {
            out.push(last);
        }
    }
    out
}

fn verdict_str(v: Verdict) -> &'static str {
    match v { Verdict::Pass => "pass", Verdict::Warn => "warn", Verdict::Fail => "fail", Verdict::Warmup => "warmup" }
}
fn metric_dto(m: Metric) -> MetricDto {
    MetricDto { value: nz(m.value), quality: m.quality, verdict: verdict_str(m.verdict) }
}

impl SnapshotDto {
    pub fn from_snapshot(s: &Snapshot, label: String, status: String) -> Self {
        SnapshotDto {
            window_len: s.window_len,
            total_bytes: s.total_bytes,
            throughput_bps: s.throughput_bps,
            shannon: metric_dto(s.shannon),
            min_entropy: metric_dto(s.min_entropy),
            monobit: metric_dto(s.monobit),
            chi_square: metric_dto(s.chi_square),
            serial_corr: metric_dto(s.serial_corr),
            monobit_p: nz(s.monobit_p),
            chi_square_p: nz(s.chi_square_p),
            histogram: s.histogram.to_vec(),
            hist_expected: s.hist_expected,
            recent: s.recent.clone(),
            coherence_sigma: s.coherence_sigma,
            coherence_band: s.coherence_band.label(),
            trial_count: s.trial_count,
            walk: decimate_walk(&s.walk, 1500),
            anomalies: s.anomalies.iter().map(|a| AnomalyDto {
                at_secs: a.at_secs, peak_sigma: a.peak_sigma, band: a.band.label(),
                duration_secs: a.duration_secs, ongoing: a.ongoing,
            }).collect(),
            label, status,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::Stats;

    #[test]
    fn warmup_metric_serializes_nan_as_null() {
        // Few bytes → warming → shannon.value may be finite but chi_square is NaN at warmup.
        let mut s = Stats::new(1 << 16);
        s.push(&[1u8; 10]);
        let snap = s.snapshot();
        let dto = SnapshotDto::from_snapshot(&snap, "test".into(), "streaming".into());
        let json = serde_json::to_string(&dto).unwrap();
        assert!(!json.contains("NaN"), "JSON must not contain NaN: {json}");
        assert!(json.contains("\"warmup\""), "expected a warmup verdict");
    }

    #[test]
    fn decimate_caps_length_and_keeps_endpoints() {
        let walk: Vec<(f64, f64)> = (0..5000).map(|k| (k as f64, (k as f64) * 0.5)).collect();
        let out = decimate_walk(&walk, 1500);
        assert!(out.len() <= 1501, "decimated to {} points", out.len());
        assert_eq!(out.first(), walk.first(), "must keep the vertex point");
        assert_eq!(out.last(), walk.last(), "must keep the newest point");
    }

    #[test]
    fn decimate_is_identity_below_cap() {
        let walk: Vec<(f64, f64)> = (0..800).map(|k| (k as f64, 0.0)).collect();
        assert_eq!(decimate_walk(&walk, 1500), walk);
    }
}
