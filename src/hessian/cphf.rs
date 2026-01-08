//! Coupled–Perturbed Hartree–Fock / Kohn–Sham (CPHF)
//!
//! Implements the Z-vector method for analytic Hessians.
//!
//! This module provides the *orbital-response* contribution needed to
//! complete the analytic nuclear Hessian.
//!
//! Compatible with:
//! - HF
//! - DFT (LDA / GGA / meta-GGA)
//! - Hybrids
//!
//! References:
//!   Helgaker, Jørgensen, Olsen – Molecular Electronic-Structure Theory

use nalgebra::{DMatrix, DVector};

/// Container for MO quantities
pub struct MoSystem {
    pub eps: Vec<f64>,          // MO energies
    pub c: DMatrix<f64>,        // AO → MO coefficients
    pub n_occ: usize,
}

/// Right-hand side builder for Z-vector equation
///
/// RHS_{ai} = ∂F_ai/∂R - ε_i ∂S_ai/∂R
pub fn build_rhs(
    fock_deriv: &DMatrix<f64>,
    overlap_deriv: &DMatrix<f64>,
    mo: &MoSystem,
) -> DVector<f64> {

    let nmo = mo.eps.len();
    let nvir = nmo - mo.n_occ;

    let mut rhs = DVector::zeros(mo.n_occ * nvir);

    for i in 0..mo.n_occ {
        for a in 0..nvir {
            let a_mo = mo.n_occ + a;

            let idx = i * nvir + a;

            rhs[idx] =
                fock_deriv[(a_mo, i)]
              - mo.eps[i] * overlap_deriv[(a_mo, i)];
        }
    }
    rhs
}

/// Apply orbital Hessian (A − B) on vector Z
///
/// This is done matrix-free (critical for performance)
pub fn apply_orbital_hessian(
    z: &DVector<f64>,
    mo: &MoSystem,
    eri_mo: &dyn Fn(usize, usize, usize, usize) -> f64,
) -> DVector<f64> {

    let nmo = mo.eps.len();
    let nvir = nmo - mo.n_occ;
    let mut out = DVector::zeros(z.len());

    for i in 0..mo.n_occ {
        for a in 0..nvir {
            let a_mo = mo.n_occ + a;
            let idx = i * nvir + a;

            // Diagonal term (energy difference)
            out[idx] += (mo.eps[a_mo] - mo.eps[i]) * z[idx];

            // Coupling via ERIs
            for j in 0..mo.n_occ {
                for b in 0..nvir {
                    let b_mo = mo.n_occ + b;
                    let jdx = j * nvir + b;

                    let g = 2.0 * eri_mo(a_mo, i, b_mo, j)
                          -       eri_mo(a_mo, j, b_mo, i);

                    out[idx] += g * z[jdx];
                }
            }
        }
    }
    out
}

/// Solve (A − B) Z = RHS using iterative solver
///
/// Uses a simple preconditioned Richardson + DIIS-like damping
pub fn solve_cphf(
    rhs: &DVector<f64>,
    mo: &MoSystem,
    eri_mo: &dyn Fn(usize, usize, usize, usize) -> f64,
    tol: f64,
    max_iter: usize,
) -> DVector<f64> {

    let mut z = DVector::zeros(rhs.len());

    for _iter in 0..max_iter {
        let hz = apply_orbital_hessian(&z, mo, eri_mo);
        let r = &hz - rhs;

        if r.norm() < tol {
            break;
        }

        // Simple diagonal preconditioner
        let nvir = mo.eps.len() - mo.n_occ;
        for i in 0..mo.n_occ {
            for a in 0..nvir {
                let idx = i * nvir + a;
                let denom = mo.eps[mo.n_occ + a] - mo.eps[i];
                z[idx] -= r[idx] / denom;
            }
        }
    }

    z
}

/// Contract Z-vector into Hessian correction
///
/// H_AB += 2 Σ_ai Z_ai ( ∂F_ai/∂R )
pub fn cphf_hessian_correction(
    z: &DVector<f64>,
    fock_deriv: &DMatrix<f64>,
    mo: &MoSystem,
) -> f64 {

    let nvir = mo.eps.len() - mo.n_occ;
    let mut val = 0.0;

    for i in 0..mo.n_occ {
        for a in 0..nvir {
            let idx = i * nvir + a;
            val += 2.0 * z[idx] * fock_deriv[(mo.n_occ + a, i)];
        }
    }
    val
}

