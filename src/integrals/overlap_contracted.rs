//! Overlap integrals for contracted shells

use crate::basis::contracted::Contracted;
use crate::basis::shell::Shell;
use crate::integrals::overlap::overlap_primitive;

/// Overlap between two contracted Gaussian functions
///
/// ⟨χ_c | χ_d⟩ = Σ_p Σ_q ⟨φ_p | φ_q⟩
pub fn overlap_contracted(
    a: &Contracted,
    b: &Contracted,
) -> f64 {
    let mut s = 0.0;

    for pa in &a.primitives {
        for pb in &b.primitives {
            s += overlap_primitive(pa, pb);
        }
    }

    s
}

/// Overlap matrix between two shells
///
/// Returns S_{μν} for all μ in shell A and ν in shell B
pub fn overlap_shell_shell(
    shell_a: &Shell,
    shell_b: &Shell,
) -> Vec<Vec<f64>> {

    let comps_a = shell_a.cartesian_components();
    let comps_b = shell_b.cartesian_components();

    let na = comps_a.len();
    let nb = comps_b.len();

    let mut s = vec![vec![0.0; nb]; na];

    for (i, _) in comps_a.iter().enumerate() {
        for (j, _) in comps_b.iter().enumerate() {

            // Cada AO cartesiano comparte las mismas primitivas
            // (la dependencia angular ya está en Primitive::ang)
            let ca = Contracted::new(shell_a.primitives.clone());
            let cb = Contracted::new(shell_b.primitives.clone());

            s[i][j] = overlap_contracted(&ca, &cb);
        }
    }

    s
}

