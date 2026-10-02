//! bytes → uniform (0,1) → index. The one mapping every random draw in the
//! system goes through, on both sides of the app/harness boundary.
//!
//! 4 bytes little-endian → u32 → `(u + 0.5) / 2^32`, strictly inside (0,1).
//! Mirrors `diffusion_rs::trng::bytes_to_gaussians`'s first stage and
//! `sidecar/trng.py`.

/// Map `4·n` raw bytes to `n` uniforms in (0,1). Errors if too few bytes.
pub fn bytes_to_uniforms(raw: &[u8], n: usize) -> Result<Vec<f64>, String> {
    let need = n * 4;
    if raw.len() < need {
        return Err(format!("need {need} bytes for {n} uniforms, got {}", raw.len()));
    }
    Ok((0..n)
        .map(|i| {
            let b = &raw[i * 4..i * 4 + 4];
            let v = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64;
            (v + 0.5) / 4_294_967_296.0
        })
        .collect())
}

/// Uniform (0,1) → integer in `[0, n)`: `min(floor(u·n), n−1)`.
pub fn uniform_to_index(u: f64, n: usize) -> usize {
    debug_assert!(n > 0);
    let i = (u * n as f64).floor();
    if i < 0.0 {
        0
    } else {
        (i as usize).min(n - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_is_open_interval_and_monotone() {
        let lo = bytes_to_uniforms(&[0, 0, 0, 0], 1).unwrap()[0];
        let hi = bytes_to_uniforms(&[255, 255, 255, 255], 1).unwrap()[0];
        assert!(lo > 0.0 && hi < 1.0);
        assert!((lo - 0.5 / 4_294_967_296.0).abs() < 1e-18);
        assert!(lo < hi);
    }

    #[test]
    fn little_endian_and_short_input() {
        // 0x00000001 LE → u = 1.5 / 2^32
        let u = bytes_to_uniforms(&[1, 0, 0, 0], 1).unwrap()[0];
        assert!((u - 1.5 / 4_294_967_296.0).abs() < 1e-18);
        assert!(bytes_to_uniforms(&[1, 2, 3], 1).is_err());
    }

    #[test]
    fn index_clamps() {
        assert_eq!(uniform_to_index(0.0, 10), 0);
        assert_eq!(uniform_to_index(0.999_999_999, 10), 9);
        assert_eq!(uniform_to_index(0.35, 10), 3);
        assert_eq!(uniform_to_index(1.0, 10), 9);
    }

    /// The committed cross-language vectors (see docs/contract/).
    #[test]
    fn matches_contract_vectors() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../docs/contract/noise_vectors.json");
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let raw: Vec<u8> = v["bytes_hex"].as_str().unwrap().as_bytes().chunks(2)
            .map(|c| u8::from_str_radix(std::str::from_utf8(c).unwrap(), 16).unwrap()).collect();
        let want: Vec<f64> = v["uniforms"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        let got = bytes_to_uniforms(&raw, want.len()).unwrap();
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-15, "{g} vs {w}");
        }
        let n = v["index_n"].as_u64().unwrap() as usize;
        let idx: Vec<usize> = v["indices"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        for (u, i) in got.iter().zip(&idx) {
            assert_eq!(uniform_to_index(*u, n), *i);
        }
    }
}
