//! Nuclear attraction integrals ⟨χ | -Z / r | χ⟩
//!
//! Implementación por capas:
//! - nuclear_attraction_primitive
//! - nuclear_attraction_contracted
//! - nuclear_attraction_shell_shell
//!
//! Usa Primitive encapsulado (getters) y Shell sin orbitales explícitos.

use std::f64::consts::PI;

use crate::basis::primitive::Primitive;
use crate::basis::contracted::Contracted;
use crate::basis::shell::Shell;
use crate::system::atom::Atom;
use crate::integrals::boys::boys0;

/// |A - B|²
#[inline]
fn dist2(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

/// Primitive nuclear attraction integral (s|s)
///
/// ⟨a | -Z / r | b⟩
pub fn nuclear_attraction_primitive(
    a: &Primitive,
    b: &Primitive,
    atom: &Atom,
) -> f64 {

    let alpha = a.exponent();
    let beta  = b.exponent();

    let A = a.center();
    let B = b.center();
    let C = atom.position;

    let z = atom.atomic_number as f64;

    let zeta = alpha + beta;

    // Gaussian product center P
    let P = [
        (alpha * A[0] + beta * B[0]) / zeta,
        (alpha * A[1] + beta * B[1]) / zeta,
        (alpha * A[2] + beta * B[2]) / zeta,
    ];

    let rpc2 = dist2(P, C);
    let rab2 = dist2(A, B);

    let pref = -2.0 * PI * z / zeta;
    let kab  = (-alpha * beta / zeta * rab2).exp();

    let t = zeta * rpc2;

    pref * kab * boys0(t) * a.norm() * b.norm()
}

/// Nuclear attraction integral for contracted AOs
pub fn nuclear_attraction_contracted(
    a: &Contracted,
    b: &Contracted,
    atom: &Atom,
) -> f64 {
    let mut v = 0.0;

    for pa in &a.primitives {
        for pb in &b.primitives {
            v += nuclear_attraction_primitive(pa, pb, atom);
        }
    }

    v
}

/// Nuclear attraction block between two shells
///
/// Devuelve matriz V_{μν}
pub fn nuclear_attraction_shell_shell(
    shell_a: &Shell,
    shell_b: &Shell,
    atoms: &[Atom],
) -> Vec<Vec<f64>> {

    let na = shell_a.n_orbitals();
    let nb = shell_b.n_orbitals();

    let mut vmat = vec![vec![0.0; nb]; na];

    // Para cada núcleo
    for atom in atoms {

        let ca = Contracted::new(shell_a.primitives.clone());
        let cb = Contracted::new(shell_b.primitives.clone());

        let val = nuclear_attraction_contracted(&ca, &cb, atom);

        for i in 0..na {
            for j in 0..nb {
                vmat[i][j] += val;
            }
        }
    }

    vmat
}

