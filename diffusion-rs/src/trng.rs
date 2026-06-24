//! TrueRNG bytes -> N(0,1) Gaussian latent, mirroring `sidecar/trng.py`.
//! 4 bytes per sample: uint32 -> uniform (0,1) -> inverse-normal-CDF.

use candle_core::{Device, Tensor};

/// Inverse standard-normal CDF (Acklam's rational approximation, |err|<1.2e-9).
fn ndtri(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
        1.383577518672690e2, -3.066479806614716e1, 2.506628277459239e0,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
        6.680131188771972e1, -1.328068155288572e1,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0,
        -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0, 3.754408661907416e0,
    ];
    let plow = 0.02425;
    let phigh = 1.0 - plow;
    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= phigh {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Map raw bytes -> N(0,1) f64 tensor of `shape` (4 bytes per sample).
pub fn bytes_to_gaussians(raw: &[u8], shape: &[usize]) -> candle_core::Result<Tensor> {
    let n: usize = shape.iter().product();
    let need = n * 4;
    if raw.len() < need {
        return Err(candle_core::Error::Msg(format!(
            "need {need} bytes for {n} gaussians, got {}",
            raw.len()
        )));
    }
    let mut vals = Vec::with_capacity(n);
    for i in 0..n {
        let b = &raw[i * 4..i * 4 + 4];
        let u32v = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64;
        let u = (u32v + 0.5) / 2f64.powi(32); // uniform in (0,1)
        vals.push(ndtri(u));
    }
    Tensor::from_vec(vals, shape.to_vec(), &Device::Cpu)
}
