//! Boys function implementation
//!
//! F_n(T) = ∫₀¹ t^(2n) exp(-T t²) dt
//!
//! Stable for:
//! - small T
//! - large T
//! - n up to ~10–15 (suficiente para d/f shells)

use std::f64::consts::PI;

/// Threshold below which we use the series expansion
const T_SMALL: f64 = 1e-8;

/// Compute the Boys function F_n(T)
#[inline]
pub fn boys(n: usize, t: f64) -> f64 {
    if t < T_SMALL {
        boys_small_t(n, t)
    } else {
        boys_general(n, t)
    }
}

// =======================================================
// Small-T expansion
// =======================================================

#[inline]
fn boys_small_t(n: usize, t: f64) -> f64 {
    // F_n(0) = 1 / (2n + 1)
    // Series:
    // F_n(t) = Σ_k (-t)^k / [k! (2n + 2k + 1)]
    let mut sum = 0.0;
    let mut term = 1.0;
    let mut k = 0usize;

    loop {
        let denom = (2 * n + 2 * k + 1) as f64;
        let contrib = term / denom;
        sum += contrib;

        k += 1;
        term *= -t / (k as f64);

        if term.abs() < 1e-16 {
            break;
        }
    }

    sum
}

// =======================================================
// General evaluation using erf + recurrence
// =======================================================

#[inline]
fn boys_general(n: usize, t: f64) -> f64 {
    // Compute F_0(T) analytically
    let sqrt_t = t.sqrt();
    let mut f0 = 0.5 * (PI / t).sqrt() * erf(sqrt_t);

    if n == 0 {
        return f0;
    }

    // Upward recurrence:
    // F_{n+1} = ((2n+1) F_n - exp(-T)) / (2T)
    let exp_t = (-t).exp();

    for m in 0..n {
        f0 = ((2.0 * (m as f64) + 1.0) * f0 - exp_t)
            / (2.0 * t);
    }

    f0
}

// =======================================================
// Error function (erf)
// =======================================================

/// Error function erf(x)
///
/// Rational approximation (Abramowitz & Stegun 7.1.26)
#[inline]
fn erf(x: f64) -> f64 {
    // constants
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p  = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1)
                    * t * (-x * x).exp());

    sign * y
}

/// Boys function F_0

pub fn boys0(t: f64) -> f64 {
    if t < 1e-8 {
        1.0
    } else {
        0.5 * (std::f64::consts::PI / t).sqrt()
            * libm::erf(t.sqrt())
    }
}

