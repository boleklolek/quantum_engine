//! Electron density evaluation on DFT grid
//!
//! Provides:
//! - rho(r) and ∇rho(r) for closed-shell DFT
//! - rho_alpha(r), rho_beta(r) for spin-polarized DFT

use crate::basis::shell::Shell;
use nalgebra::DMatrix;

/// Density at a grid point (closed-shell)
pub struct DensityPoint {
    pub rho: f64,
    pub grad: [f64; 3],
}

/// Density for spin-polarized case
pub struct SpinDensityPoint {
    pub rho_a: f64,
    pub rho_b: f64,
    pub grad_a: [f64; 3],
    pub grad_b: [f64; 3],
}

// ========================================================
// Closed-shell density
// ========================================================

pub fn density_at_point(
    shells: &[Shell],
    shell_centers: &[[f64; 3]],
    density: &DMatrix<f64>,
    r: [f64; 3],
) -> DensityPoint {

    let nao = density.nrows();
    let mut rho = 0.0;
    let mut grad = [0.0; 3];

    for (si, ci) in shells.iter().zip(shell_centers.iter()) {
        let off_i = si.offset;

        for mu in 0..si.orbitals.len() {
            let phi_mu = si.orbitals[mu].value(*ci, r);
            let grad_mu = si.orbitals[mu].gradient(*ci, r);

            let i = off_i + mu;

            for (sj, cj) in shells.iter().zip(shell_centers.iter()) {
                let off_j = sj.offset;

                for nu in 0..sj.orbitals.len() {
                    let phi_nu = sj.orbitals[nu].value(*cj, r);
                    let grad_nu = sj.orbitals[nu].gradient(*cj, r);

                    let j = off_j + nu;
                    let pij = density[(i, j)];

                    rho += pij * phi_mu * phi_nu;

                    for k in 0..3 {
                        grad[k] += pij * (
                            grad_mu[k] * phi_nu
                          + phi_mu * grad_nu[k]
                        );
                    }
                }
            }
        }
    }

    DensityPoint { rho, grad }
}

// ========================================================
// Spin-polarized density (UDFT)
// ========================================================

pub fn spin_density_at_point(
    shells: &[Shell],
    shell_centers: &[[f64; 3]],
    density_alpha: &DMatrix<f64>,
    density_beta: &DMatrix<f64>,
    r: [f64; 3],
) -> SpinDensityPoint {

    let da = density_at_point(shells, shell_centers, density_alpha, r);
    let db = density_at_point(shells, shell_centers, density_beta, r);

    SpinDensityPoint {
        rho_a: da.rho,
        rho_b: db.rho,
        grad_a: da.grad,
        grad_b: db.grad,
    }
}

