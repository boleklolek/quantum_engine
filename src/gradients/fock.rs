//! First nuclear derivatives of the Fock matrix
//!
//! Computes the *explicit* derivative:
//!   ∂F = ∂H_core + ∂J − ∂K + ∂V_xc
//!
//! Orbital-response terms are NOT included here.

use nalgebra::DMatrix;

use crate::basis::shell::Shell;
use crate::system::atom::Atom;
use crate::gradients::overlap::overlap_derivative;
use crate::dft::vxc::build_vxc;

/// Compute explicit AO Fock derivative ∂F/∂R_Ai
pub fn fock_derivative(
    shells: &[Shell],
    shell_centers: &[[f64;3]],
    density: &DMatrix<f64>,
    fock: &DMatrix<f64>,
    atoms: &[Atom],
    atom: usize,
    axis: usize,
    is_dft: bool,
) -> DMatrix<f64> {

    let nao = fock.nrows();
    let mut dF = DMatrix::zeros(nao, nao);

    // ==================================================
    // 1. One-electron part: T + V_nuc
    // ==================================================
    for (si, ci) in shells.iter().zip(shell_centers.iter()) {
        let off_i = si.offset;
        let ni = si.orbitals.len();

        for (sj, cj) in shells.iter().zip(shell_centers.iter()) {
            let off_j = sj.offset;
            let nj = sj.orbitals.len();

            let dT =
                si.second_deriv_kinetic(sj, *ci, *cj, atoms);

            let dV =
                si.second_deriv_nuclear_attraction(sj, *ci, *cj, atoms);

            for mu in 0..ni {
                for nu in 0..nj {
                    dF[(off_i+mu, off_j+nu)] +=
                        dT[atom][atom][axis][axis][mu][nu]
                      + dV[atom][atom][axis][axis][mu][nu];
                }
            }
        }
    }

    // ==================================================
    // 2. Coulomb (J) term
    // ==================================================
    for si in shells {
        for sj in shells {
            for sk in shells {
                for sl in shells {

                    let dERI =
                        si.first_deriv_eri(sj, sk, sl, atoms.len());

                    for mu in 0..si.orbitals.len() {
                        for nu in 0..sj.orbitals.len() {
                            let i = si.offset + mu;
                            let j = sj.offset + nu;

                            let mut val = 0.0;
                            for la in 0..sk.orbitals.len() {
                                for si2 in 0..sl.orbitals.len() {
                                    let k = sk.offset + la;
                                    let l = sl.offset + si2;

                                    val += density[(k,l)]
                                        * dERI[atom][mu][nu][la][si2][axis];
                                }
                            }
                            dF[(i,j)] += 2.0 * val;
                        }
                    }
                }
            }
        }
    }

    // ==================================================
    // 3. Exchange (K) term
    // ==================================================
    for si in shells {
        for sj in shells {
            for sk in shells {
                for sl in shells {

                    let dERI =
                        si.first_deriv_eri(sj, sk, sl, atoms.len());

                    for mu in 0..si.orbitals.len() {
                        for nu in 0..sj.orbitals.len() {
                            let i = si.offset + mu;
                            let j = sj.offset + nu;

                            let mut val = 0.0;
                            for la in 0..sk.orbitals.len() {
                                for si2 in 0..sl.orbitals.len() {
                                    let k = sk.offset + la;
                                    let l = sl.offset + si2;

                                    val += density[(k,l)]
                                        * dERI[atom][mu][si2][la][nu][axis];
                                }
                            }
                            dF[(i,j)] -= val;
                        }
                    }
                }
            }
        }
    }

    // ==================================================
    // 4. XC contribution (DFT only)
    // ==================================================
    if is_dft {
        let dvxc =
            build_vxc_derivative(
                shells,
                shell_centers,
                density,
                atoms,
                atom,
                axis,
            );
        dF += dvxc;
    }

    // ==================================================
    // 5. Symmetrize
    // ==================================================
    for i in 0..nao {
        for j in 0..i {
            let avg = 0.5 * (dF[(i,j)] + dF[(j,i)]);
            dF[(i,j)] = avg;
            dF[(j,i)] = avg;
        }
    }

    dF
}

