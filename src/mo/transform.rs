//! AO → MO transformations
//!
//! Implements:
//! - AO matrix → MO matrix
//! - AO ERIs → MO ERIs (on-the-fly callable)
//!
//! This is performance-critical but conceptually simple.

use nalgebra::{DMatrix};
use crate::mo::space::MoSpace;

/// Transform AO matrix into MO matrix: Cᵀ A C
pub fn ao_to_mo_matrix(
    ao: &DMatrix<f64>,
    c: &DMatrix<f64>,
) -> DMatrix<f64> {
    let tmp = ao * c;
    c.transpose() * tmp
}

/// Builds a closure that computes MO ERIs on demand:
///
///   (pq|rs) = Σ Cμp Cνq Cλr Cσs (μν|λσ)
///
/// AO ERIs are accessed through a provided callback.
pub fn ao_to_mo_eri<'a>(
    c: &'a DMatrix<f64>,
    eri_ao: &'a dyn Fn(usize, usize, usize, usize) -> f64,
) -> impl Fn(usize, usize, usize, usize) -> f64 + 'a {

    let nao = c.nrows();

    move |p: usize, q: usize, r: usize, s: usize| -> f64 {
        let mut val = 0.0;

        for mu in 0..nao {
            let c_mp = c[(mu, p)];
            if c_mp.abs() < 1e-12 { continue; }

            for nu in 0..nao {
                let c_nq = c[(nu, q)];
                if c_nq.abs() < 1e-12 { continue; }

                for la in 0..nao {
                    let c_lr = c[(la, r)];
                    if c_lr.abs() < 1e-12 { continue; }

                    for si in 0..nao {
                        let c_ss = c[(si, s)];
                        if c_ss.abs() < 1e-12 { continue; }

                        val +=
                            c_mp * c_nq * c_lr * c_ss *
                            eri_ao(mu, nu, la, si);
                    }
                }
            }
        }
        val
    }
}

