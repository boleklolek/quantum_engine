//! SCF utility routines
//!
//! Core SCF algebra:
//! - One-electron matrix (Hcore)
//! - Fock construction (scaled)
//! - Roothaan solver
//! - SCF energies
//! - DIIS helpers

use nalgebra::{DMatrix, DVector, SymmetricEigen};

use crate::basis::shell::Shell;
use crate::system::atom::Atom;

// ======================================================
// Small matrix helpers
// ======================================================

pub fn add(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    a + b
}

pub fn add_inplace(a: &mut DMatrix<f64>, b: &DMatrix<f64>) {
    *a += b;
}

// ======================================================
// One-electron matrix
// ======================================================

/// Build Hcore = T + V_nuc
pub fn build_one_electron_matrix(
    shells: &[Shell],
    shell_centers: &[[f64; 3]],
    atoms: &[Atom],
) -> DMatrix<f64> {

    let nao = shells.last().unwrap().offset + shells.last().unwrap().orbitals.len();
    let mut h = DMatrix::zeros(nao, nao);

    for (si, ci) in shells.iter().zip(shell_centers.iter()) {
        for (sj, cj) in shells.iter().zip(shell_centers.iter()) {
            let off_i = si.offset;
            let off_j = sj.offset;

            let t = si.kinetic(sj, *ci, *cj);
            let v = si.nuclear_attraction(sj, *ci, *cj, atoms);

            for mu in 0..si.orbitals.len() {
                for nu in 0..sj.orbitals.len() {
                    h[(off_i + mu, off_j + nu)] =
                        t[mu][nu] + v[mu][nu];
                }
            }
        }
    }

    // enforce symmetry
    for i in 0..nao {
        for j in 0..i {
            let avg = 0.5 * (h[(i, j)] + h[(j, i)]);
            h[(i, j)] = avg;
            h[(j, i)] = avg;
        }
    }

    h
}

// ======================================================
// Fock matrix
// ======================================================

/// Build scaled Fock matrix (RHF):
///   F = H + 2J − K   (+ Vxc outside if DFT)
pub fn build_fock_scaled(
    hcore: &DMatrix<f64>,
    j: &DMatrix<f64>,
    k: &DMatrix<f64>,
) -> DMatrix<f64> {
    hcore + 2.0 * j - k
}

// ======================================================
// Energies
// ======================================================

/// Electronic energy (RHF)
pub fn electronic_energy_scaled(
    density: &DMatrix<f64>,
    hcore: &DMatrix<f64>,
    fock: &DMatrix<f64>,
) -> f64 {

    let mut e = 0.0;
    let n = density.nrows();

    for i in 0..n {
        for j in 0..n {
            e += density[(i, j)] * (hcore[(i, j)] + fock[(i, j)]);
        }
    }

    0.5 * e
}

// ======================================================
// Roothaan equations
// ======================================================

/// Solve FC = S C ε
pub fn solve_roothaan(
    fock: &DMatrix<f64>,
    overlap: &DMatrix<f64>,
) -> (DMatrix<f64>, Vec<f64>) {

    // S^(-1/2)
    let s_eig = SymmetricEigen::new(overlap.clone());
    let mut s_inv_sqrt = DMatrix::zeros(overlap.nrows(), overlap.ncols());

    for i in 0..s_eig.eigenvalues.len() {
        s_inv_sqrt[(i, i)] = 1.0 / s_eig.eigenvalues[i].sqrt();
    }

    let x = &s_eig.eigenvectors * s_inv_sqrt * s_eig.eigenvectors.transpose();
    let f_prime = &x.transpose() * fock * &x;

    let eig = SymmetricEigen::new(f_prime);

    let c = x * eig.eigenvectors;
    let eps = eig.eigenvalues.iter().copied().collect();

    (c, eps)
}

// ======================================================
// DIIS helper
// ======================================================

/// DIIS error matrix:  e = FDS − SDF
pub fn diis_error(
    fock: &DMatrix<f64>,
    density: &DMatrix<f64>,
    overlap: &DMatrix<f64>,
) -> DMatrix<f64> {

    fock * density * overlap
        - overlap * density * fock
}

