//! Vertical Recurrence Relations (VRR) for ERI
//!
//! Implementa:
//!   (ps|ss)
//!   (pp|ss)
//!   (ds|ss)
//!   (fs|ss)
//!
//! siguiendo Obara–Saika, con centros encapsulados
//! en Primitive (NO se pasan como argumentos)

use crate::basis::primitive::Primitive;
use crate::integrals::eri::eri_ssss::eri_ssss;

/// Gaussian product center P
#[inline]
fn gaussian_product_center(a: &Primitive, b: &Primitive) -> [f64; 3] {
    let alpha = a.exponent();
    let beta = b.exponent();
    let zeta = alpha + beta;
    let A = a.center();
    let B = b.center();

    [
        (alpha * A[0] + beta * B[0]) / zeta,
        (alpha * A[1] + beta * B[1]) / zeta,
        (alpha * A[2] + beta * B[2]) / zeta,
    ]
}

/// (p_i s | s s)
///
/// dir = 0(x),1(y),2(z)
pub fn eri_psss(
    a: &Primitive,
    b: &Primitive,
    c: &Primitive,
    d: &Primitive,
    dir: usize,
) -> f64 {
    let P = gaussian_product_center(a, b);
    let A = a.center();

    let ssss = eri_ssss(a, b, c, d);

    (P[dir] - A[dir]) * ssss
}

/// (p_i p_j | s s)
pub fn eri_ppss(
    a: &Primitive,
    b: &Primitive,
    c: &Primitive,
    d: &Primitive,
    i: usize,
    j: usize,
) -> f64 {
    let P = gaussian_product_center(a, b);
    let A = a.center();

    let pi = eri_psss(a, b, c, d, i);
    let pj = eri_psss(a, b, c, d, j);
    let ssss = eri_ssss(a, b, c, d);

    let alpha = a.exponent();
    let beta = b.exponent();
    let zeta = alpha + beta;

    let delta = if i == j { 1.0 } else { 0.0 };

    (P[i] - A[i]) * pj
        + delta / (2.0 * zeta) * ssss
}

/// (d_ij s | s s)
pub fn eri_dsss(
    a: &Primitive,
    b: &Primitive,
    c: &Primitive,
    d: &Primitive,
    i: usize,
    j: usize,
) -> f64 {
    let P = gaussian_product_center(a, b);
    let A = a.center();

    let pij = eri_ppss(a, b, c, d, i, j);
    let pj = eri_psss(a, b, c, d, j);
    let pi = eri_psss(a, b, c, d, i);

    let alpha = a.exponent();
    let beta = b.exponent();
    let zeta = alpha + beta;

    let delta = if i == j { 1.0 } else { 0.0 };

    (P[i] - A[i]) * pj
        + delta / (2.0 * zeta) * pi
}

/// (f_ijk s | s s)
pub fn eri_fsss(
    a: &Primitive,
    b: &Primitive,
    c: &Primitive,
    d: &Primitive,
    i: usize,
    j: usize,
    k: usize,
) -> f64 {
    let P = gaussian_product_center(a, b);
    let A = a.center();

    let dij = eri_dsss(a, b, c, d, j, k);
    let pj = eri_psss(a, b, c, d, j);

    let alpha = a.exponent();
    let beta = b.exponent();
    let zeta = alpha + beta;

    let delta = if j == k { 1.0 } else { 0.0 };

    (P[i] - A[i]) * dij
        + delta / (2.0 * zeta) * pj
}

