//! Total nuclear gradients (HF / DFT / meta-GGA / UDFT)
//!
//! Includes:
//! - Nuclear repulsion
//! - One-electron gradients (T + V)
//! - Two-electron ERI gradients
//! - Pulay overlap term
//! - XC gradients:
//!     * LDA / GGA
//!     * meta-GGA (τ)
//!     * Spin-polarized (UDFT)

use crate::basis::shell::Shell;
use crate::system::atom::Atom;

// HF components
use crate::gradients::nuclear_repulsion::grad_nuclear_repulsion;
use crate::gradients::one_electron::grad_one_electron;
use crate::gradients::two_electron::grad_two_electron;
use crate::gradients::overlap_pulay::grad_overlap_pulay;

// DFT XC gradients
use crate::gradients::dft_xc::grad_xc_lda_gga;
use crate::gradients::dft_xc_spin::grad_xc_udft;
use crate::gradients::dft_xc_meta::grad_xc_meta;

// XC selector
use crate::dft::vxc::XcMethod;

/// Compute total nuclear gradient
///
/// Usage:
/// - RHF / DFT: provide `density`
/// - UDFT: provide `density_alpha` and `density_beta`
/// - meta-GGA: provide `coeff` and `n_occ`
pub fn total_gradient(
    shells: &[Shell],
    shell_centers: &[[f64; 3]],
    atoms: &[Atom],

    // Densities
    density: Option<&Vec<Vec<f64>>>,              // RHF / DFT
    density_alpha: Option<&Vec<Vec<f64>>>,        // UDFT
    density_beta: Option<&Vec<Vec<f64>>>,

    // Orbitals (needed for meta-GGA τ)
    coeff: Option<&Vec<Vec<f64>>>,
    n_occ: Option<usize>,

    // Common
    fock: &Vec<Vec<f64>>,
    eri_grad: &dyn Fn(usize, usize, usize, usize, usize) -> [f64; 3],
    xc: Option<XcMethod>,
) -> Vec<[f64; 3]> {

    let natoms = atoms.len();
    let mut grad = grad_nuclear_repulsion(atoms);

    // ==================================================
    // RHF / DFT / meta-GGA (spin-restricted)
    // ==================================================
    if let Some(p) = density {
        let g1 = grad_one_electron(shells, shell_centers, p, atoms);
        let g2 = grad_two_electron(shells, p, eri_grad, natoms);
        let gp = grad_overlap_pulay(shells, fock, natoms);

        for a in 0..natoms {
            for k in 0..3 {
                grad[a][k] += g1[a][k] + g2[a][k] + gp[a][k];
            }
        }

        if let Some(method) = xc {
            match method.as_ref() {
                XcMethod::LDA | XcMethod::GGA
                | XcMethod::Hybrid { base: XcMethod::LDA, .. }
                | XcMethod::Hybrid { base: XcMethod::GGA, .. } => {

                    let gxc = grad_xc_lda_gga(
                        shells,
                        shell_centers,
                        p,
                        atoms,
                        method,
                    );

                    for a in 0..natoms {
                        for k in 0..3 {
                            grad[a][k] += gxc[a][k];
                        }
                    }
                }

                XcMethod::MetaGGA
                | XcMethod::Hybrid { base: XcMethod::MetaGGA, .. } => {

                    let gmeta = grad_xc_meta(
                        shells,
                        shell_centers,
                        p,
                        coeff.expect("meta-GGA requires coeff"),
                        n_occ.expect("meta-GGA requires n_occ"),
                        atoms,
                        method,
                    );

                    for a in 0..natoms {
                        for k in 0..3 {
                            grad[a][k] += gmeta[a][k];
                        }
                    }
                }
            }
        }
    }

    // ==================================================
    // UDFT (spin-polarized)
    // ==================================================
    if let (Some(pa), Some(pb), Some(method)) =
        (density_alpha, density_beta, xc)
    {
        let gxc_spin = grad_xc_udft(
            shells,
            shell_centers,
            pa,
            pb,
            atoms,
            method,
        );

        for a in 0..natoms {
            for k in 0..3 {
                grad[a][k] += gxc_spin[a][k];
            }
        }
    }

    grad
}

