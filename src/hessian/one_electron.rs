//! Second nuclear derivatives of one-electron terms (T + V)
//!
//! Computes:
//!   H_AB^{ij} = ∂²/∂R_Ai∂R_Bj ⟨ μ | T + V_nuc | ν ⟩ · P_{μν}
//!
//! Includes:
//! - kinetic energy second derivatives
//! - nuclear attraction second derivatives
//!
//! Contracted with AO density matrix

use crate::basis::shell::Shell;
use crate::system::atom::Atom;

/// One-electron Hessian contribution
///
/// shells        : AO shells with offsets
/// shell_centers : coordinates of shell centers
/// density       : AO density matrix
/// atoms         : molecular atoms
///
/// Returns Hessian matrix (3N x 3N)
pub fn hess_one_electron(
    shells: &[Shell],
    shell_centers: &[[f64; 3]],
    density: &Vec<Vec<f64>>,
    atoms: &[Atom],
) -> Vec<Vec<f64>> {

    let natoms = atoms.len();
    let dim = 3 * natoms;
    let mut hess = vec![vec![0.0; dim]; dim];

    // Loop over shell pairs
    for (si, ci) in shells.iter().zip(shell_centers.iter()) {
        let off_i = si.offset;
        let ni = si.orbitals.len();

        for (sj, cj) in shells.iter().zip(shell_centers.iter()) {
            let off_j = sj.offset;
            let nj = sj.orbitals.len();

            // --------------------------------------------------
            // Second derivatives of kinetic integrals
            // d²T/dR_A dR_B  -> [A][B][i][j][μ][ν]
            // --------------------------------------------------
            let d2t = si.second_deriv_kinetic(sj, *ci, *cj, atoms);

            // --------------------------------------------------
            // Second derivatives of nuclear attraction
            // d²V/dR_A dR_B
            // --------------------------------------------------
            let d2v = si.second_deriv_nuclear_attraction(sj, *ci, *cj, atoms);

            for mu in 0..ni {
                for nu in 0..nj {
                    let p = density[off_i + mu][off_j + nu];
                    if p.abs() < 1e-14 {
                        continue;
                    }

                    for a in 0..natoms {
                        for b in 0..natoms {
                            for iaxis in 0..3 {
                                for jaxis in 0..3 {
                                    let idx_a = 3 * a + iaxis;
                                    let idx_b = 3 * b + jaxis;

                                    hess[idx_a][idx_b] +=
                                        p * (
                                            d2t[a][b][iaxis][jaxis][mu][nu]
                                          + d2v[a][b][iaxis][jaxis][mu][nu]
                                        );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Symmetrize Hessian
    for i in 0..dim {
        for j in 0..i {
            let avg = 0.5 * (hess[i][j] + hess[j][i]);
            hess[i][j] = avg;
            hess[j][i] = avg;
        }
    }

    hess
}

