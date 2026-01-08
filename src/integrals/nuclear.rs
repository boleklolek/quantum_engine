//! Nuclear attraction integrals ⟨χ | -Z / r | χ⟩

use crate::basis::primitive::Primitive;
use crate::basis::contracted::Contracted;
use crate::basis::shell::Shell;
use crate::system::atom::Atom;
use crate::integrals::overlap::overlap_primitive;

/// Nuclear attraction integral for primitives (s|s)
///
/// ⟨a | -Z / r | b⟩
pub fn nuclear_attraction_primitive(
    a: &Primitive,
    b: &Primitive,
    atom: &Atom,
) -> f64 {
    // placeholder (Boys F0 será usado aquí)
    let z = atom.atomic_number as f64;
    -z * overlap_primitive(a, b)
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
/// Devuelve V_{μν}
pub fn nuclear_attraction_shell_shell(
    shell_a: &Shell,
    shell_b: &Shell,
    atoms: &[Atom],
) -> Vec<Vec<f64>> {

    let na = shell_a.n_orbitals();
    let nb = shell_b.n_orbitals();

    let mut v = vec![vec![0.0; nb]; na];

    for atom in atoms {
        let ca = Contracted::new(shell_a.primitives.clone());
        let cb = Contracted::new(shell_b.primitives.clone());

        let contrib = nuclear_attraction_contracted(&ca, &cb, atom);

        for i in 0..na {
            for j in 0..nb {
                v[i][j] += contrib;
            }
        }
    }

    v
}

