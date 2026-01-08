//! Second nuclear derivatives of the XC energy (DFT Hessian)
//!
//! Implements the *explicit* XC Hessian terms:
//!   ∂²E_xc / ∂R_Ai ∂R_Bj
//!
//! Covered:
//! - LDA
//! - GGA
//! - meta-GGA (τ)
//!
//! Not included here:
//! - Orbital response (CPHF / Z-vector)
//!   → handled in hessian/cphf.rs
//!
//! This matches the standard decomposition used in Psi4 / Q-Chem.

use crate::basis::shell::Shell;
use crate::system::atom::Atom;
use crate::dft::grid::{DftGrid, GridPoint};
use crate::dft::density::density_at_point;
use crate::dft::tau::{tau_at_point, dtau_dra};
use crate::dft::libxc::LibXC;
use crate::dft::vxc::XcMethod;

/// Explicit XC Hessian contribution
///
/// Returns Hessian matrix (3N x 3N)
pub fn hess_xc(
    shells: &[Shell],
    shell_centers: &[[f64;3]],
    density: &Vec<Vec<f64>>,
    coeff: Option<&Vec<Vec<f64>>>,   // needed for meta-GGA
    n_occ: Option<usize>,            // needed for meta-GGA
    atoms: &[Atom],
    method: XcMethod,
) -> Vec<Vec<f64>> {

    let natoms = atoms.len();
    let dim = 3 * natoms;
    let nao = density.len();

    let mut hess = vec![vec![0.0; dim]; dim];

    // -------------------------------------------
    // Resolve hybrid scaling
    // -------------------------------------------
    let (xc_base, hf_frac) = match method {
        XcMethod::Hybrid { base, hyb } => (*base, hyb.hf_fraction()),
        other => (other, 0.0),
    };

    // -------------------------------------------
    // Select libxc functionals
    // -------------------------------------------
    let (fx, fc, is_meta) = match xc_base {
        XcMethod::LDA => (
            LibXC::new(1, false),
            LibXC::new(7, false),
            false,
        ),
        XcMethod::GGA => (
            LibXC::new(101, false),
            LibXC::new(130, false),
            false,
        ),
        XcMethod::MetaGGA => (
            LibXC::new(263, false), // SCAN_X
            LibXC::new(267, false), // SCAN_C
            true,
        ),
        _ => unreachable!(),
    };

    // -------------------------------------------
    // Grid
    // -------------------------------------------
    let grid = DftGrid::new(atoms, 30, 14);

    for GridPoint { r, weight } in grid.points {

        let dp = density_at_point(shells, shell_centers, density, r);
        if dp.rho < 1e-12 {
            continue;
        }

        // AO values and gradients
        let mut phi = Vec::with_capacity(nao);
        let mut grad_phi = Vec::with_capacity(nao);

        for (sh, c) in shells.iter().zip(shell_centers.iter()) {
            for ao in &sh.orbitals {
                phi.push(ao.value(*c, r));
                grad_phi.push(ao.gradient(*c, r));
            }
        }

        // -------------------------------------------
        // Build invariants
        // -------------------------------------------
        let rho = vec![dp.rho];
        let sigma = vec![
            dp.grad[0]*dp.grad[0] +
            dp.grad[1]*dp.grad[1] +
            dp.grad[2]*dp.grad[2]
        ];

        let tau = if is_meta {
            tau_at_point(
                shells,
                shell_centers,
                coeff.expect("meta-GGA requires coeff"),
                n_occ.expect("meta-GGA requires n_occ"),
                r,
            )
        } else { 0.0 };

        // -------------------------------------------
        // XC second derivatives from libxc
        // -------------------------------------------
        // libxc returns:
        //   d²f/dρ², d²f/dρdσ, d²f/dσ², d²f/dτ², ...
        let xc2 = fx.eval_xc_hessian(&rho, &sigma, tau);
        let cc2 = fc.eval_xc_hessian(&rho, &sigma, tau);

        let vrr = (xc2.vrr[0] + cc2.vrr[0]) * (1.0 - hf_frac);
        let vrs = (xc2.vrs[0] + cc2.vrs[0]) * (1.0 - hf_frac);
        let vss = (xc2.vss[0] + cc2.vss[0]) * (1.0 - hf_frac);
        let vtt = if is_meta {
            (xc2.vtt[0] + cc2.vtt[0]) * (1.0 - hf_frac)
        } else { 0.0 };

        // -------------------------------------------
        // Nuclear derivatives
        // -------------------------------------------
        for a in 0..natoms {
            for b in 0..natoms {

                let mut d2rho = [0.0;3];
                let mut d2sig = [0.0;3];
                let mut d2tau = [0.0;3];

                for mu in 0..nao {
                    for nu in 0..nao {
                        let p = density[mu][nu];
                        if p.abs() < 1e-14 { continue; }

                        let dphi =
                            [ grad_phi[mu][0]*phi[nu] + phi[mu]*grad_phi[nu][0],
                              grad_phi[mu][1]*phi[nu] + phi[mu]*grad_phi[nu][1],
                              grad_phi[mu][2]*phi[nu] + phi[mu]*grad_phi[nu][2] ];

                        for k in 0..3 {
                            d2rho[k] += p * dphi[k];
                            d2sig[k] += 2.0 * dp.grad[k] * p * dphi[k];
                        }
                    }
                }

                if is_meta {
                    let dt = dtau_dra(
                        shells,
                        shell_centers,
                        coeff.unwrap(),
                        n_occ.unwrap(),
                        b,
                        r,
                    );
                    for k in 0..3 {
                        d2tau[k] = dt[k];
                    }
                }

                for ia in 0..3 {
                    let idx_a = 3*a + ia;
                    for ib in 0..3 {
                        let idx_b = 3*b + ib;

                        hess[idx_a][idx_b] += weight * (
                            vrr * d2rho[ia] * d2rho[ib] +
                            2.0 * vrs * d2rho[ia] * d2sig[ib] +
                            vss * d2sig[ia] * d2sig[ib] +
                            vtt * d2tau[ia] * d2tau[ib]
                        );
                    }
                }
            }
        }
    }

    // -------------------------------------------
    // Symmetrize
    // -------------------------------------------
    for i in 0..dim {
        for j in 0..i {
            let avg = 0.5 * (hess[i][j] + hess[j][i]);
            hess[i][j] = avg;
            hess[j][i] = avg;
        }
    }

    hess
}

