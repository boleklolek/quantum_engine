//! Basis loader: builds Shells and Primitives from basis-set data

use crate::system::molecule::Molecule;
use crate::basis::primitive::Primitive;
use crate::basis::shell::Shell;
use crate::basis::reader::{read_basis_set, BasisShell};

/// Build all shells for a molecule from a basis-set name
pub fn load_basis(
    molecule: &Molecule,
    basis_name: &str,
) -> Vec<Shell> {

    let mut shells: Vec<Shell> = Vec::new();
    let mut ao_offset: usize = 0;

    for atom in &molecule.atoms {
        // ----------------------------------------------
        // 1. Read basis for this atomic symbol
        // ----------------------------------------------
        let basis = read_basis_set(basis_name, &atom.symbol)
            .unwrap_or_else(|| {
                panic!(
                    "Basis '{}' not found for element {}",
                    basis_name, atom.symbol
                )
            });

        // ----------------------------------------------
        // 2. Center of all primitives on this atom
        // ----------------------------------------------
        let center: [f64; 3] = atom.position;

        // ----------------------------------------------
        // 3. Loop over basis shells (s, p, d, ...)
        // ----------------------------------------------
        for bshell in basis.shells.iter() {

            let ang: [usize; 3] = bshell.angular_momentum;

            // ------------------------------------------
            // 4. Build primitives for this shell
            // ------------------------------------------
            let mut primitives: Vec<Primitive> = Vec::new();

            for (exp, coeff) in bshell.primitives.iter() {
                primitives.push(
                    Primitive::new(
                        *exp,
                        *coeff,
                        center,
                        ang,
                    )
                );
            }

            // ------------------------------------------
            // 5. Construct shell
            // ------------------------------------------
            let shell = Shell::new(
                primitives,
                ang,
                center,
                ao_offset,
            );

            ao_offset += shell.n_orbitals();

            shells.push(shell);
        }
    }

    shells
}

