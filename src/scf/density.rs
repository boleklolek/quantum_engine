//! Density matrix construction and comparison utilities
//!
//! Supports:
//! - RHF / DFT (single density)
//! - UHF / UDFT (spin densities)

/// Build density matrix (RHF / DFT)
///
/// P_μν = 2 * Σ_i^occ C_{μi} C_{νi}
///

use nalgebra::DMatrix;

pub fn build_density(
    coeff: DMatrix<f64>,
    n_electrons: usize,
) -> Vec<Vec<f64>> {
    let nao = coeff.nrows();
    let n_occ = n_electrons / 2;

    let mut p = vec![vec![0.0_f64; nao]; nao];

    for mu in 0..nao {
        for nu in 0..nao {
            let mut sum = 0.0;
            for i in 0..n_occ {
                sum += coeff[(mu,i)] * coeff[(nu,i)];
            }
            p[mu][nu] = 2.0 * sum;
        }
    }
    p
}

/// Build spin density matrix (UHF / UDFT)
///
/// P^σ_μν = Σ_i^occ C^σ_{μi} C^σ_{νi}
pub fn build_spin_density(
    coeff: &Vec<Vec<f64>>,
    n_electrons: usize,
) -> Vec<Vec<f64>> {
    let nao = coeff.len();
    let mut p = vec![vec![0.0; nao]; nao];

    for mu in 0..nao {
        for nu in 0..nao {
            let mut sum = 0.0;
            for i in 0..n_electrons {
                sum += coeff[mu][i] * coeff[nu][i];
            }
            p[mu][nu] = sum;
        }
    }
    p
}

/// RMS difference between two density matrices
///
/// Used as SCF convergence criterion
pub fn rms_density_diff(
    p_old: &Vec<Vec<f64>>,
    p_new: &Vec<Vec<f64>>,
) -> f64 {
    let nao = p_old.len();
    let mut sum = 0.0;

    for mu in 0..nao {
        for nu in 0..nao {
            let diff = p_new[mu][nu] - p_old[mu][nu];
            sum += diff * diff;
        }
    }

    (sum / (nao * nao) as f64).sqrt()
}
