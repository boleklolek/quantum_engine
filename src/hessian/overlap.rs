//! Second-order Pulay (overlap) contribution to the nuclear Hessian
//!
//! Implements the explicit overlap Hessian terms (no orbital response):
//!
//! H_AB^{ij} +=
//!   - Σ_{μν} P_{μν} ∂²S_{μν}/∂R_Ai∂R_Bj * F_{νμ}
//!   - 2 Σ_{μν} ∂S_{μν}/∂R_Ai * ∂F_{νμ}/∂R_Bj
//!
//! Notes:
//! - The second term couples with Fock derivatives; this implementation
//!   provides the explicit (non-response) part.
//! - Orbital-response (Z-vector) corrections are handled in hessian/cphf.rs
//!
//! Assumptions:
//! - Shell provides first/second derivatives of overlap integrals
//! - Shells know AO offsets

use crate::basis::shell::Shell;

/// Pulay second-order Hessian contribution
///
/// shells        : AO shells
/// shell_centers : coordinates of shell centers
/// density       : AO density matrix
/// fock          : AO Fock matrix
/// natoms        : number of nuclei
///
/// Returns Hessian matrix (3N x 3N)
pub fn hess_overlap_pulay(
    shells: &[Shell],
    shell_centers: &[[f64; 3]],
    density: &Vec<Vec<f64>>,
    fock: &Vec<Vec<f64>>,
    natoms: usize,
) -> Vec<Vec<f64>> {

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
            // First derivatives of overlap
            // dS/dR_A  -> [A][i][μ][ν]
            // --------------------------------------------------
            let ds =
                si.first_deriv_overlap(sj, *ci, *cj, natoms);

            // --------------------------------------------------
            // Second derivatives of overlap
            // d²S/dR_A dR_B -> [A][B][i][j][μ][ν]
            // --------------------------------------------------
            let d2s =
                si.second_deriv_overlap(sj, *ci, *cj, natoms);

            for mu in 0..ni {
                let i = off_i + mu;

                for nu in 0..nj {
                    let j = off_j + nu;

                    let p = density[i][j];
                    let f = fock[j][i]; // note transpose

                    if p.abs() < 1e-14 && f.abs() < 1e-14 {
                        continue;
                    }

                    // ------------------------------------------
                    // Term 1: -P * F * d²S
                    // ------------------------------------------
                    for a in 0..natoms {
                        for b in 0..natoms {
                            for iaxis in 0..3 {
                                for jaxis in 0..3 {

                                    let idx_a = 3*a + iaxis;
                                    let idx_b = 3*b + jaxis;

                                    hess[idx_a][idx_b] -=
                                        p * f *
                                        d2s[a][b][iaxis][jaxis][mu][nu];
                                }
                            }
                        }
                    }

                    // ------------------------------------------
                    // Term 2: -2 * (dS/dR_A) * (dF/dR_B)
                    // ------------------------------------------
                    // The explicit non-response part uses:
                    //   dF ≈ 0  (orbital response neglected here)
                    // This term is added via CPHF/Z-vector later.
                    //
                    // We keep the structure here for completeness.
                    //
                    // If you later include explicit dF/dR, add here.
                    let _ = &ds; // explicitly unused until response is added
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

