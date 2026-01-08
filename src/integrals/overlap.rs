//! Overlap integrals ⟨χ_a | χ_b⟩
//!
//! Implementa primitivas cartesianas gaussianas usando
//! el teorema del producto gaussiano.
//!
//! Esta versión cubre (s|s) de forma explícita y deja
//! preparado el camino para p/d mediante recursión.

use std::f64::consts::PI;

use crate::basis::primitive::Primitive;

/// Distancia al cuadrado |A - B|^2
#[inline]
fn dist2(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx*dx + dy*dy + dz*dz
}

/// Centro del producto gaussiano
#[inline]
fn gaussian_product_center(
    alpha: f64,
    a: [f64; 3],
    beta: f64,
    b: [f64; 3],
) -> [f64; 3] {
    let z = alpha + beta;
    [
        (alpha * a[0] + beta * b[0]) / z,
        (alpha * a[1] + beta * b[1]) / z,
        (alpha * a[2] + beta * b[2]) / z,
    ]
}

/// Overlap primitivo (s|s)
pub fn overlap_ss(a: &Primitive, b: &Primitive) -> f64 {

    let alpha = a.exponent();
    let beta  = b.exponent();

    let A = a.center();
    let B = b.center();

    let rab2 = dist2(A, B);
    let zeta = alpha + beta;

    let prefactor = (PI / zeta).powf(1.5);
    let k_ab = (-alpha * beta / zeta * rab2).exp();

    prefactor * k_ab * a.norm() * b.norm()
}

/// Overlap primitivo general (cartesiano)
///
/// Actualmente implementa solo (s|s).
/// Para l > 0 debe usarse VRR (Obara–Saika).
pub fn overlap_primitive(a: &Primitive, b: &Primitive) -> f64 {

    let la = a.ang();
    let lb = b.ang();

    if la == [0,0,0] && lb == [0,0,0] {
        return overlap_ss(a, b);
    }

    panic!("overlap_primitive: angular momentum > 0 not yet implemented");
}

/// Overlap entre dos shells (devuelve matriz μν)
///
/// ⟨χ_μ | χ_ν⟩ para todos los AO del shell
pub fn overlap_shell_shell(
    shell_a: &[Primitive],
    shell_b: &[Primitive],
) -> Vec<Vec<f64>> {

    let na = shell_a.len();
    let nb = shell_b.len();

    let mut s = vec![vec![0.0; nb]; na];

    for (i, pa) in shell_a.iter().enumerate() {
        for (j, pb) in shell_b.iter().enumerate() {
            s[i][j] = overlap_primitive(pa, pb);
        }
    }

    s
}

