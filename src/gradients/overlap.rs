//! First nuclear derivatives of the AO overlap matrix
//!
//! Computes:
//!   (∂S_{μν} / ∂R_Ai)

use nalgebra::DMatrix;
use crate::basis::shell::Shell;

/// Compute AO overlap derivative with respect to atom A and Cartesian axis
///
/// Returns AO matrix dS/dR_Ai
pub fn overlap_derivative(
    shells: &[Shell],
    shell_centers: &[[f64;3]],
    natoms: usize,
    atom: usize,
    axis: usize,
) -> DMatrix<f64> {

    let nao = shells.last().unwrap().offset
            + shells.last().unwrap().orbitals.len();

    let mut ds = DMatrix::zeros(nao, nao);

    for (si, ci) in shells.iter().zip(shell_centers.iter()) {
        let off_i = si.offset;
        let ni = si.orbitals.len();

        for (sj, cj) in shells.iter().zip(shell_centers.iter()) {
            let off_j = sj.offset;
            let nj = sj.orbitals.len();

            let dS =
                si.first_deriv_overlap(sj, *ci, *cj, natoms);

            for mu in 0..ni {
                for nu in 0..nj {
                    ds[(off_i+mu, off_j+nu)] =
                        dS[atom][mu][nu][axis];
                }
            }
        }
    }

    // Ensure symmetry
    for i in 0..nao {
        for j in 0..i {
            let avg = 0.5 * (ds[(i,j)] + ds[(j,i)]);
            ds[(i,j)] = avg;
            ds[(j,i)] = avg;
        }
    }

    ds
}

