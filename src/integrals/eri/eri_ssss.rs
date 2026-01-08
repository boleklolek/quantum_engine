//! Primitive electron repulsion integral (ss|ss)
//!
//! (ab|cd) = ∬ exp(-α|r-A|²) exp(-β|r-B|²)
//!            1/|r-r'|
//!            exp(-γ|r'-C|²) exp(-δ|r'-D|²) dr dr'

use crate::basis::primitive::Primitive;
use crate::integrals::boys::boys0;

/// Squared distance between two points
#[inline]
fn dist2(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx*dx + dy*dy + dz*dz
}

/// Compute primitive (ss|ss) ERI
pub fn eri_ssss(
    p: &Primitive,
    q: &Primitive,
    r: &Primitive,
    s: &Primitive,
) -> f64 {

    // --------------------------------------------------
    // 1. Exponents and centers
    // --------------------------------------------------
    let a = p.exponent();
    let b = q.exponent();
    let c = r.exponent();
    let d = s.exponent();

    let A = p.center();
    let B = q.center();
    let C = r.center();
    let D = s.center();

    // --------------------------------------------------
    // 2. Gaussian product theorem
    // --------------------------------------------------
    let zeta = a + b;
    let eta  = c + d;

    let P = [
        (a*A[0] + b*B[0]) / zeta,
        (a*A[1] + b*B[1]) / zeta,
        (a*A[2] + b*B[2]) / zeta,
    ];

    let Q = [
        (c*C[0] + d*D[0]) / eta,
        (c*C[1] + d*D[1]) / eta,
        (c*C[2] + d*D[2]) / eta,
    ];

    // --------------------------------------------------
    // 3. Distances
    // --------------------------------------------------
    let rab2 = dist2(A, B);
    let rcd2 = dist2(C, D);
    let rpq2 = dist2(P, Q);

    // --------------------------------------------------
    // 4. Prefactors
    // --------------------------------------------------
    let k_ab = (-a * b / zeta * rab2).exp();
    let k_cd = (-c * d / eta  * rcd2).exp();

    let prefactor =
        2.0 * std::f64::consts::PI.powf(2.5)
        / (zeta * eta * (zeta + eta).sqrt());

    // --------------------------------------------------
    // 5. Boys argument
    // --------------------------------------------------
    let t = zeta * eta / (zeta + eta) * rpq2;

    // --------------------------------------------------
    // 6. Final value
    // --------------------------------------------------
    let value =
        prefactor * k_ab * k_cd * boys0(t);

    // Contracted coefficients
    value * p.coefficient() * q.coefficient() * r.coefficient() * s.coefficient()
}

