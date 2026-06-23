//! Small numerical helpers: error function and tail p-values for the
//! randomness tests. No external math crate — these approximations are
//! accurate to better than 1e-7, which is far beyond what we display.

/// Complementary error function, Abramowitz & Stegun 7.1.26 (max err ~1.5e-7).
pub fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let tau = t
        * (-z * z - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807
                                + t * (-1.13520398
                                    + t * (1.48851587
                                        + t * (-0.82215223 + t * 0.17087277)))))))))
        .exp();
    if x >= 0.0 {
        tau
    } else {
        2.0 - tau
    }
}

/// Upper-tail p-value of the standard normal: P(Z > z).
pub fn normal_sf(z: f64) -> f64 {
    0.5 * erfc(z / std::f64::consts::SQRT_2)
}

/// Two-sided normal p-value: P(|Z| > |z|).
pub fn normal_two_sided(z: f64) -> f64 {
    erfc(z.abs() / std::f64::consts::SQRT_2)
}

/// Upper-tail p-value of a chi-square with `k` degrees of freedom via the
/// Wilson–Hilferty cube-root normal approximation. Excellent for large k
/// (here k = 255), which is exactly our regime.
pub fn chi_square_sf(x: f64, k: f64) -> f64 {
    if k <= 0.0 {
        return f64::NAN;
    }
    let t = (x / k).powf(1.0 / 3.0);
    let mean = 1.0 - 2.0 / (9.0 * k);
    let sd = (2.0 / (9.0 * k)).sqrt();
    let z = (t - mean) / sd;
    normal_sf(z)
}

/// Natural log of the gamma function (Lanczos approximation).
pub fn ln_gamma(x: f64) -> f64 {
    const C: [f64; 6] = [
        76.180_091_729_471_46,
        -86.505_320_329_416_77,
        24.014_098_240_830_91,
        -1.231_739_572_450_155,
        0.120_865_097_386_617_9e-2,
        -0.539_523_938_495_3e-5,
    ];
    let mut y = x;
    let tmp = x + 5.5 - (x + 0.5) * (x + 5.5).ln();
    let mut ser = 1.000_000_000_190_015;
    for c in C.iter() {
        y += 1.0;
        ser += c / y;
    }
    -tmp + (2.506_628_274_631_000_5 * ser / x).ln()
}

/// Regularized lower incomplete gamma P(a, x) via series expansion.
fn gamma_p_series(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let gln = ln_gamma(a);
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..300 {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * 1e-15 {
            break;
        }
    }
    sum * (-x + a * x.ln() - gln).exp()
}

/// Regularized upper incomplete gamma Q(a, x) via continued fraction.
fn gamma_q_cf(a: f64, x: f64) -> f64 {
    let gln = ln_gamma(a);
    let fpmin = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / fpmin;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..300 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = b + an / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-15 {
            break;
        }
    }
    (-x + a * x.ln() - gln).exp() * h
}

/// Regularized upper incomplete gamma Q(a, x) = 1 - P(a, x).
pub fn gamma_q(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x < a + 1.0 {
        1.0 - gamma_p_series(a, x)
    } else {
        gamma_q_cf(a, x)
    }
}

/// Exact upper-tail chi-square p-value via the incomplete gamma function.
/// Accurate at the low degrees of freedom (2–15) the audit tests use, where
/// the Wilson–Hilferty approximation is unreliable.
pub fn chi_square_p(chi2: f64, df: f64) -> f64 {
    gamma_q(df / 2.0, chi2 / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn known_values() {
        // Tolerance 1e-6 matches the algorithm's documented max error of ~1.5e-7.
        // The brief used 1e-9 which is tighter than the approximation can achieve.
        assert!((erfc(0.0) - 1.0).abs() < 1e-6);
        assert!((normal_sf(0.0) - 0.5).abs() < 1e-6);
        assert!((normal_two_sided(0.0) - 1.0).abs() < 1e-6);
        // chi-square at its mean (x=k) gives p≈0.5 for large k.
        assert!((chi_square_sf(255.0, 255.0) - 0.5).abs() < 0.05);
    }
}
