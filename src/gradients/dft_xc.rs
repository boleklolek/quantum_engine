//! DFT exchange-correlation gradients (explicit part)
//!
//! Provides nuclear gradients of XC energy:
//!   ∂E_xc / ∂R_A
//!
//! Covers:
//! - LDA
//! - GGA
//!
//! Orbital-response terms are handled via CPHF and must NOT be here.

use nalgebra::DMatrix;

use crate::basis::shell::Shell;
use crate::system::atom::Atom;
use crate::dft::grid::DftGrid;
use crate::dft::density::density_at_point;
use crate::dft::libxc::LibXC;

/// Compute explicit XC gradient for LDA / GGA
///
/// Returns AO matrix contribution to ∂F/∂R_Ai
pub fn grad_xc_lda_gga(
    shells: &[Shell],
    shell_centers: &[[f64; 3]],
    density: &DMatrix<f64>,
    atoms: &[Atom],
    atom: usize,
    axis: usize,
    is_gga: bool,
) -> DMatrix<f64> {

    let nao = density.nrows();
    let mut grad = DMatrix::zeros(nao, nao);

    // Select libxc functionals (example: LDA_X + LDA_C_PZ)
    let fx = LibXC::new(if is_gga { 101 } else { 1 }, false);
    let fc = LibXC::new(if is_gga { 130 } else { 7 }, false);

    let grid = DftGrid::new(atoms, 30, 86);

    for pt in &grid.points {
        let rho_pt =
            density_at_point(shells, shell_centers, density, pt.r);

        if rho_pt.rho < 1e-12 {
            continue;
        }

        let rho = vec![rho_pt.rho];
        let sigma = vec![
            rho_pt.grad[0]*rho_pt.grad[0]
          + rho_pt.grad[1]*rho_pt.grad[1]
          + rho_pt.grad[2]*rho_pt.grad[2]
        ];

        let (_, vrho, vsigma) =
            fx.eval_gga(&rho, &sigma);

        let (_, vcrho, vcsigma) =
            fc.eval_gga(&rho, &sigma);

        let vr = vrho[0] + vcrho[0];
        let vs = if is_gga { vsigma[0] + vcsigma[0] } else { 0.0 };

        // AO loop
        for (si, ci) in shells.iter().zip(shell_centers.iter()) {
            for (sj, cj) in shells.iter().zip(shell_centers.iter()) {
                let off_i = si.offset;
                let off_j = sj.offset;

                for mu in 0..si.orbitals.len() {
                    let phi_mu = si.orbitals[mu].value(*ci, pt.r);
                    let dphi_mu = si.orbitals[mu].gradient(*ci, pt.r)[axis];

                    for nu in 0..sj.orbitals.len() {
                        let phi_nu = sj.orbitals[nu].value(*cj, pt.r);
                        let dphi_nu = sj.orbitals[nu].gradient(*cj, pt.r)[axis];

                        grad[(off_i+mu, off_j+nu)] +=
                            pt.weight * (
                                vr * (dphi_mu * phi_nu + phi_mu * dphi_nu)
                              + 2.0 * vs * rho_pt.grad[axis]
                                * (phi_mu * phi_nu)
                            );
                    }
                }
            }
        }
    }

    grad
}

