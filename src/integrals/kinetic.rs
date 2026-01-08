//! Kinetic energy integrals ⟨χ | -½ ∇² | χ⟩
//!
//! Estructura por capas:
//! - kinetic_ss            : primitiva (s|s)
//! - kinetic_primitive     : primitiva general
//! - kinetic_contracted    : AO contraído
//! - kinetic_shell_shell   : bloque shell-shell (SCF)

use std::f64::consts::PI;

use crate::basis::primitive::Primitive;
use crate::basis::contracted::Contracted;
use crate::basis::shell::Shell;

/// |A - B|²
#[inline]
fn dist2(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

/// Kinetic energy integral (s|s)
pub fn kinetic_ss(a: &Primitive, b: &Primitive) -> f64 {
    let alpha = a.exponent();
    let beta  = b.exponent();

    let A = a.center();
    let B = b.center();

    let rab2 = dist2(A, B);
    let zeta = alpha + beta;
    let reduced = alpha * beta / zeta;

    let pref = reduced * (3.0 - 2.0 * reduced * rab2)
        * (PI / zeta).powf(1.5);

    let kab = (-reduced * rab2).exp();

    pref * kab * a.norm() * b.norm()
}

/// General primitive kinetic integral
pub fn kinetic_primitive(a: &Primitive, b: &Primitive) -> f64 {
    let la = a.ang();
    let lb = b.ang();

    if la == [0, 0, 0] && lb == [0, 0, 0] {
        return kinetic_ss(a, b);
    }

    panic!("kinetic_primitive: angular momentum > 0 not implemented yet");
}

/// Kinetic integral between two contracted AOs
pub fn kinetic_contracted(
    a: &Contracted,
    b: &Contracted,
) -> f64 {
    let mut t = 0.0;

    for pa in &a.primitives {
        for pb in &b.primitives {
            t += kinetic_primitive(pa, pb);
        }
    }

    t
}

/// Kinetic energy block between two shells
///
/// Devuelve la matriz T_{μν}
pub fn kinetic_shell_shell(
    shell_a: &Shell,
    shell_b: &Shell,
) -> Vec<Vec<f64>> {

    let comps_a = shell_a.cartesian_components();
    let comps_b = shell_b.cartesian_components();

    let na = comps_a.len();
    let nb = comps_b.len();

    let mut t = vec![vec![0.0; nb]; na];

    for (i, _) in comps_a.iter().enumerate() {
        for (j, _) in comps_b.iter().enumerate() {

            // Cada AO cartesiano comparte el mismo conjunto de primitivas
            let ca = Contracted::new(shell_a.primitives.clone());
            let cb = Contracted::new(shell_b.primitives.clone());

            t[i][j] = kinetic_contracted(&ca, &cb);
        }
    }

    t
}

