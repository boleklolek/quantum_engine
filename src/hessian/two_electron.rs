//! Second nuclear derivatives of two-electron integrals (ERI Hessian)
//!
//! Computes:
//!   H_AB^{ij} = 1/2 Σ_{μνλσ} P_{μν} P_{λσ}
//!                ∂²(μν|λσ)/∂R_Ai∂R_Bj
//!
//! This file implements ONLY the explicit (non-response) part.
//! Orbital response terms are handled via CPHF/Z-vector.
//
//! Assumes:
//! - ERI second derivatives are available at shell level
//! - Shells know their AO offsets

use crate::basis::shell::Shell;

/// Two-electron Hessian contribution (explicit ERI term)
///
/// shells  : AO shells
/// density : AO density matrix
/// natoms  : number of nuclei
///
/// Returns Hessian matrix (3N x 3N)
pub fn hess_two_electron(
    shells: &[Shell],
    density: &Vec<Vec<f64>>,
    natoms: usize,
) -> Vec<Vec<f64>> {

    let dim = 3 * natoms;
    let nao = density.len();
    let mut hess = vec![vec![0.0; dim]; dim];

    // Loop over shell quartets
    for sh_mu in shells {
        let off_mu = sh_mu.offset;
        let nmu = sh_mu.orbitals.len();

        for sh_nu in shells {
            let off_nu = sh_nu.offset;
            let nnu = sh_nu.orbitals.len();

            for sh_la in shells {
                let off_la = sh_la.offset;
                let nla = sh_la.orbitals.len();

                for sh_si in shells {
                    let off_si = sh_si.offset;
                    let nsi = sh_si.orbitals.len();

                    // --------------------------------------------------
                    // Second derivatives of ERIs
                    // d²(μν|λσ)/dR_A dR_B
                    //
                    // Tensor layout:
                    //   d2eri[A][B][iaxis][jaxis][μ][ν][λ][σ]
                    // --------------------------------------------------
                    let d2eri =
                        sh_mu.second_deriv_eri(
                            sh_nu,
                            sh_la,
                            sh_si,
                            natoms,
                        );

                    // Contract with density matrices
                    for mu in 0..nmu {
                        let i = off_mu + mu;

                        for nu in 0..nnu {
                            let j = off_nu + nu;
                            let p_ij = density[i][j];
                            if p_ij.abs() < 1e-14 {
                                continue;
                            }

                            for la in 0..nla {
                                let k = off_la + la;

                                for si in 0..nsi {
                                    let l = off_si + si;
                                    let p_kl = density[k][l];
                                    if p_kl.abs() < 1e-14 {
                                        continue;
                                    }

                                    let pref = 0.5 * p_ij * p_kl;

                                    for a in 0..natoms {
                                        for b in 0..natoms {
                                            for iaxis in 0..3 {
                                                for jaxis in 0..3 {

                                                    let idx_a = 3*a + iaxis;
                                                    let idx_b = 3*b + jaxis;

                                                    hess[idx_a][idx_b] += pref *
                                                        d2eri[a][b][iaxis][jaxis]
                                                             [mu][nu][la][si];
                                                }
                                            }
                                        }
                                    }
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

