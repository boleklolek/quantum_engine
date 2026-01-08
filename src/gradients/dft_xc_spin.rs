//! Analytic XC gradients for spin-polarized DFT (UDFT)
//!
//! Supports:
//! - LDA (spin)
//! - GGA (spin)
//! - Hybrids (HF scaling handled outside)

use crate::basis::shell::Shell;
use crate::dft::grid::{DftGrid, GridPoint};
use crate::dft::density::spin_density_at_point;
use crate::dft::libxc::LibXC;
use crate::system::atom::Atom;
use crate::dft::vxc::{XcMethod};

/// ∂E_xc / ∂R_A (spin-polarized)
pub fn grad_xc_udft(
    shells: &[Shell],
    shell_centers: &[[f64; 3]],
    p_alpha: &Vec<Vec<f64>>,
    p_beta: &Vec<Vec<f64>>,
    atoms: &[Atom],
    method: XcMethod,
) -> Vec<[f64; 3]> {

    let (xc_base, hf_frac) = match method {
        XcMethod::Hybrid { base, hyb } => (base, hyb.hf_fraction()),
        m => (m, 0.0),
    };

    let (fx_id, fc_id) = match xc_base {
        XcMethod::LDA => (1, 7),
        _ => (101, 130),
    };

    let fx = LibXC::new(fx_id, true);
    let fc = LibXC::new(fc_id, true);

    let grid = DftGrid::new(atoms, 30, 14);

    let nao = p_alpha.len();
    let natoms = atoms.len();
    let mut grad = vec![[0.0; 3]; natoms];

    for GridPoint { r, weight } in grid.points {
        let dp = spin_density_at_point(
            shells,
            shell_centers,
            p_alpha,
            p_beta,
            r,
        );

        let rho_tot = dp.rho_a + dp.rho_b;
        if rho_tot < 1e-12 {
            continue;
        }

        // AO values and gradients
        let mut phi = Vec::with_capacity(nao);
        let mut grad_phi = Vec::with_capacity(nao);

        for (shell, center) in shells.iter().zip(shell_centers.iter()) {
            for ao in &shell.orbitals {
                phi.push(ao.value(*center, r));
                grad_phi.push(ao.gradient(*center, r));
            }
        }

        match xc_base {
            // =========================================
            // Spin-LDA
            // =========================================
            XcMethod::LDA => {
                let rho = vec![dp.rho_a, dp.rho_b];

                let (_, vx) = fx.eval_lda(&rho);
                let (_, vc) = fc.eval_lda(&rho);

                let v_a = (vx[0] + vc[0]) * (1.0 - hf_frac);
                let v_b = (vx[1] + vc[1]) * (1.0 - hf_frac);

                for (a, _) in atoms.iter().enumerate() {
                    let mut d_rho_a = [0.0; 3];
                    let mut d_rho_b = [0.0; 3];

                    for mu in 0..nao {
                        for nu in 0..nao {
                            let pa = p_alpha[mu][nu];
                            let pb = p_beta[mu][nu];
                            let dphi_mu = grad_phi[mu];
                            let dphi_nu = grad_phi[nu];

                            for k in 0..3 {
                                let dphi =
                                    dphi_mu[k]*phi[nu] +
                                    phi[mu]*dphi_nu[k];

                                d_rho_a[k] += pa * dphi;
                                d_rho_b[k] += pb * dphi;
                            }
                        }
                    }

                    for k in 0..3 {
                        grad[a][k] += weight * (
                            v_a * d_rho_a[k] +
                            v_b * d_rho_b[k]
                        );
                    }
                }
            }

            // =========================================
            // Spin-GGA
            // =========================================
            _ => {
                let sigma_aa =
                    dp.grad_a[0]*dp.grad_a[0] +
                    dp.grad_a[1]*dp.grad_a[1] +
                    dp.grad_a[2]*dp.grad_a[2];

                let sigma_bb =
                    dp.grad_b[0]*dp.grad_b[0] +
                    dp.grad_b[1]*dp.grad_b[1] +
                    dp.grad_b[2]*dp.grad_b[2];

                let sigma_ab =
                    dp.grad_a[0]*dp.grad_b[0] +
                    dp.grad_a[1]*dp.grad_b[1] +
                    dp.grad_a[2]*dp.grad_b[2];

                let rho = vec![dp.rho_a, dp.rho_b];
                let sigma = vec![sigma_aa, sigma_ab, sigma_bb];

                let (_, vrx, vsx) = fx.eval_gga(&rho, &sigma);
                let (_, vrc, vsc) = fc.eval_gga(&rho, &sigma);

                let v_ra = (vrx[0] + vrc[0]) * (1.0 - hf_frac);
                let v_rb = (vrx[1] + vrc[1]) * (1.0 - hf_frac);

                let v_saa = (vsx[0] + vsc[0]) * (1.0 - hf_frac);
                let v_sab = (vsx[1] + vsc[1]) * (1.0 - hf_frac);
                let v_sbb = (vsx[2] + vsc[2]) * (1.0 - hf_frac);

                for (a, _) in atoms.iter().enumerate() {
                    let mut d_ra = [0.0; 3];
                    let mut d_rb = [0.0; 3];
                    let mut d_saa = [0.0; 3];
                    let mut d_sab = [0.0; 3];
                    let mut d_sbb = [0.0; 3];

                    for mu in 0..nao {
                        for nu in 0..nao {
                            let pa = p_alpha[mu][nu];
                            let pb = p_beta[mu][nu];
                            let dphi_mu = grad_phi[mu];
                            let dphi_nu = grad_phi[nu];

                            for k in 0..3 {
                                let dphi =
                                    dphi_mu[k]*phi[nu] +
                                    phi[mu]*dphi_nu[k];

                                d_ra[k] += pa * dphi;
                                d_rb[k] += pb * dphi;

                                d_saa[k] += 2.0 * dp.grad_a[k] * pa * dphi;
                                d_sbb[k] += 2.0 * dp.grad_b[k] * pb * dphi;
                                d_sab[k] += dp.grad_b[k] * pa * dphi
                                          + dp.grad_a[k] * pb * dphi;
                            }
                        }
                    }

                    for k in 0..3 {
                        grad[a][k] += weight * (
                            v_ra * d_ra[k] +
                            v_rb * d_rb[k] +
                            v_saa * d_saa[k] +
                            v_sab * d_sab[k] +
                            v_sbb * d_sbb[k]
                        );
                    }
                }
            }
        }
    }

    grad
}
