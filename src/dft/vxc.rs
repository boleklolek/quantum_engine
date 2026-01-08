//! Exchange–correlation builder for DFT / UDFT
//!
//! Supports:
//! - LDA / spin-LDA
//! - GGA (PBE) / spin-GGA
//! - meta-GGA (SCAN) / spin-meta-GGA
//! - Hybrids (PBE0, B3LYP, hybrid meta-GGA)
//!
//! Returns AO-space Vxc and energy corrections:
//!   Exc − ∫ rho * vxc

use crate::basis::shell::Shell;
use crate::dft::grid::{DftGrid, GridPoint};
use crate::dft::density::{density_at_point, spin_density_at_point};
use crate::dft::tau::tau_at_point;
use crate::dft::libxc::LibXC;
use crate::system::atom::Atom;

//
// =========================
// XC selectors
// =========================
//

#[derive(Clone, Copy)]
pub enum Hybrid {
    None,
    PBE0,   // 25% HF
    B3LYP,  // 20% HF
}

impl Hybrid {
    pub fn hf_fraction(&self) -> f64 {
        match self {
            Hybrid::None => 0.0,
            Hybrid::PBE0 => 0.25,
            Hybrid::B3LYP => 0.20,
        }
    }
}

#[derive(Clone)]
pub enum XcMethod {
    LDA,
    GGA,
    MetaGGA,
    Hybrid { base: Box<XcMethod>, hyb: Hybrid },
}

//
// =========================
// Energy container
// =========================
//

pub struct DftEnergy {
    pub exc: f64,
    pub int_rho_vxc: f64,
}

//
// ======================================================================
// RHF / DFT (spin-restricted) Vxc
// ======================================================================
//

pub fn build_vxc(
    shells: &[Shell],
    shell_centers: &[[f64; 3]],
    density: &Vec<Vec<f64>>,
    coeff: Option<&Vec<Vec<f64>>>, // required only for meta-GGA
    n_occ: Option<usize>,
    atoms: &[Atom],
    method: XcMethod,
) -> (Vec<Vec<f64>>, DftEnergy) {

    let nao = density.len();
    let mut vxc = vec![vec![0.0; nao]; nao];

    let (xc_base, hf_frac) = match method {
        XcMethod::Hybrid { base, hyb } => (*base, hyb.hf_fraction()),
        other => (other, 0.0),
    };

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

    let grid = DftGrid::new(atoms, 30, 14);

    let mut exc = 0.0;
    let mut int_rho_vxc = 0.0;

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

        // basic invariants
        let rho = vec![dp.rho];
        let sigma = vec![dp.grad[0]*dp.grad[0]
                       + dp.grad[1]*dp.grad[1]
                       + dp.grad[2]*dp.grad[2]];

        // meta-GGA τ
        let tau = if is_meta {
            let c = coeff.expect("meta-GGA requires coeff");
            let n = n_occ.expect("meta-GGA requires n_occ");
            tau_at_point(shells, shell_centers, c, n, r)
        } else {
            0.0
        };

        // libxc evaluation
        let (eps_x, vrx, vsx, vtx_x) = fx.eval_all(&rho, &sigma, tau);
        let (eps_c, vrc, vsc, vtx_c) = fc.eval_all(&rho, &sigma, tau);

        let eps = (eps_x[0] + eps_c[0]) * (1.0 - hf_frac);
        let vrho = (vrx[0] + vrc[0]) * (1.0 - hf_frac);
        let vsig = (vsx[0] + vsc[0]) * (1.0 - hf_frac);
        let vtau = (vtx_x[0] + vtx_c[0]) * (1.0 - hf_frac);

        exc += weight * dp.rho * eps;
        int_rho_vxc += weight * dp.rho * vrho;

        for mu in 0..nao {
            for nu in 0..nao {
                let mut val = vrho * phi[mu] * phi[nu];

                // GGA
                let grad_dot =
                    dp.grad[0] * (grad_phi[mu][0]*phi[nu] + phi[mu]*grad_phi[nu][0]) +
                    dp.grad[1] * (grad_phi[mu][1]*phi[nu] + phi[mu]*grad_phi[nu][1]) +
                    dp.grad[2] * (grad_phi[mu][2]*phi[nu] + phi[mu]*grad_phi[nu][2]);

                val += 2.0 * vsig * grad_dot;

                // meta-GGA
                if is_meta {
                    val += vtau *
                        (grad_phi[mu][0]*grad_phi[nu][0] +
                         grad_phi[mu][1]*grad_phi[nu][1] +
                         grad_phi[mu][2]*grad_phi[nu][2]);
                }

                vxc[mu][nu] += weight * val;
            }
        }
    }

    (
        vxc,
        DftEnergy {
            exc,
            int_rho_vxc,
        },
    )
}

//
// ======================================================================
// UDFT (spin-polarized) Vxc
// ======================================================================
//

pub fn build_vxc_udft(
    shells: &[Shell],
    shell_centers: &[[f64; 3]],
    p_alpha: &Vec<Vec<f64>>,
    p_beta: &Vec<Vec<f64>>,
    coeff_a: Option<&Vec<Vec<f64>>>,
    coeff_b: Option<&Vec<Vec<f64>>>,
    n_occ_a: Option<usize>,
    n_occ_b: Option<usize>,
    atoms: &[Atom],
    method: XcMethod,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, DftEnergy) {

    let nao = p_alpha.len();
    let mut vxa = vec![vec![0.0; nao]; nao];
    let mut vxb = vec![vec![0.0; nao]; nao];

    let (xc_base, hf_frac) = match method {
        XcMethod::Hybrid { base, hyb } => (*base, hyb.hf_fraction()),
        other => (other, 0.0),
    };

    let (fx, fc, is_meta) = match xc_base {
        XcMethod::LDA => (
            LibXC::new(1, true),
            LibXC::new(7, true),
            false,
        ),
        XcMethod::GGA => (
            LibXC::new(101, true),
            LibXC::new(130, true),
            false,
        ),
        XcMethod::MetaGGA => (
            LibXC::new(263, true),
            LibXC::new(267, true),
            true,
        ),
        _ => unreachable!(),
    };

    let grid = DftGrid::new(atoms, 30, 14);

    let mut exc = 0.0;
    let mut int_rho_vxc = 0.0;

    for GridPoint { r, weight } in grid.points {
        let dp = spin_density_at_point(
            shells,
            shell_centers,
            p_alpha,
            p_beta,
            r,
        );

        if dp.rho_a + dp.rho_b < 1e-12 {
            continue;
        }

        // AO basis
        let mut phi = Vec::with_capacity(nao);
        let mut grad_phi = Vec::with_capacity(nao);

        for (sh, c) in shells.iter().zip(shell_centers.iter()) {
            for ao in &sh.orbitals {
                phi.push(ao.value(*c, r));
                grad_phi.push(ao.gradient(*c, r));
            }
        }

        let rho = vec![dp.rho_a, dp.rho_b];
        let sigma = vec![
            dp.grad_a[0]*dp.grad_a[0] + dp.grad_a[1]*dp.grad_a[1] + dp.grad_a[2]*dp.grad_a[2],
            dp.grad_a[0]*dp.grad_b[0] + dp.grad_a[1]*dp.grad_b[1] + dp.grad_a[2]*dp.grad_b[2],
            dp.grad_b[0]*dp.grad_b[0] + dp.grad_b[1]*dp.grad_b[1] + dp.grad_b[2]*dp.grad_b[2],
        ];

        let tau_a = if is_meta {
            tau_at_point(shells, shell_centers, coeff_a.unwrap(), n_occ_a.unwrap(), r)
        } else { 0.0 };
        let tau_b = if is_meta {
            tau_at_point(shells, shell_centers, coeff_b.unwrap(), n_occ_b.unwrap(), r)
        } else { 0.0 };

        let tau = vec![tau_a, tau_b];

        let (eps_x, vrx, vsx, vtx_x) = fx.eval_all_spin(&rho, &sigma, &tau);
        let (eps_c, vrc, vsc, vtx_c) = fc.eval_all_spin(&rho, &sigma, &tau);

        let v_ra = (vrx[0] + vrc[0]) * (1.0 - hf_frac);
        let v_rb = (vrx[1] + vrc[1]) * (1.0 - hf_frac);

        let v_saa = (vsx[0] + vsc[0]) * (1.0 - hf_frac);
        let v_sab = (vsx[1] + vsc[1]) * (1.0 - hf_frac);
        let v_sbb = (vsx[2] + vsc[2]) * (1.0 - hf_frac);

        let v_tau_a = (vtx_x[0] + vtx_c[0]) * (1.0 - hf_frac);
        let v_tau_b = (vtx_x[1] + vtx_c[1]) * (1.0 - hf_frac);

        exc += weight *
            (rho[0] + rho[1]) *
            (eps_x[0] + eps_c[0]) * (1.0 - hf_frac);

        int_rho_vxc += weight * (
            rho[0] * v_ra + rho[1] * v_rb
        );

        for mu in 0..nao {
            for nu in 0..nao {
                let mut val_a =
                    v_ra * phi[mu] * phi[nu]
                    + 2.0 * v_saa *
                        dp.grad_a.iter().zip(&grad_phi[mu]).map(|(a,b)| a*b).sum::<f64>()
                    + v_tau_a *
                        grad_phi[mu].iter().zip(&grad_phi[nu]).map(|(a,b)| a*b).sum::<f64>();

                let mut val_b =
                    v_rb * phi[mu] * phi[nu]
                    + 2.0 * v_sbb *
                        dp.grad_b.iter().zip(&grad_phi[mu]).map(|(a,b)| a*b).sum::<f64>()
                    + v_tau_b *
                        grad_phi[mu].iter().zip(&grad_phi[nu]).map(|(a,b)| a*b).sum::<f64>();

                vxa[mu][nu] += weight * val_a;
                vxb[mu][nu] += weight * val_b;
            }
        }
    }

    (
        vxa,
        vxb,
        DftEnergy {
            exc,
            int_rho_vxc,
        },
    )
}
