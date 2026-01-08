//! Analytic nuclear gradients for meta-GGA (τ-dependent)

use crate::basis::shell::Shell;
use crate::dft::grid::{DftGrid, GridPoint};
use crate::dft::density::density_at_point;
use crate::dft::tau::{tau_at_point, dtau_dra};
use crate::dft::libxc::LibXC;
use crate::system::atom::Atom;
use crate::dft::vxc::XcMethod;

pub fn grad_xc_meta(
    shells: &[Shell],
    centers: &[[f64;3]],
    density: &Vec<Vec<f64>>,
    coeff: &Vec<Vec<f64>>,
    n_occ: usize,
    atoms: &[Atom],
    method: XcMethod,
) -> Vec<[f64;3]> {

    let (xc_base, hf_frac) = match method {
        XcMethod::Hybrid { base, hyb } => (base, hyb.hf_fraction()),
        m => (m, 0.0),
    };
    assert!(matches!(xc_base, XcMethod::MetaGGA));

    let fx = LibXC::new(263, false); // SCAN_X
    let fc = LibXC::new(267, false); // SCAN_C

    let grid = DftGrid::new(atoms, 30, 14);
    let natoms = atoms.len();
    let nao = density.len();

    let mut grad = vec![[0.0;3]; natoms];

    for GridPoint{ r, weight } in grid.points {
        let dp = density_at_point(shells, centers, density, r);
        if dp.rho < 1e-12 { continue; }

        let tau = tau_at_point(shells, centers, coeff, n_occ, r);

        let rho = vec![dp.rho];
        let sigma = vec![dp.grad[0]*dp.grad[0] + dp.grad[1]*dp.grad[1] + dp.grad[2]*dp.grad[2]];

        let (_, vrx, vsx, vtx) = fx.eval_mgga(&rho, &sigma, tau);
        let (_, vrc, vsc, vtc) = fc.eval_mgga(&rho, &sigma, tau);

        let vrho = (vrx[0] + vrc[0]) * (1.0 - hf_frac);
        let vsig = (vsx[0] + vsc[0]) * (1.0 - hf_frac);
        let vtau = (vtx[0] + vtc[0]) * (1.0 - hf_frac);

        // AO values and gradients
        let mut phi = Vec::with_capacity(nao);
        let mut grad_phi = Vec::with_capacity(nao);
        for (sh,c) in shells.iter().zip(centers.iter()) {
            for ao in &sh.orbitals {
                phi.push(ao.value(*c, r));
                grad_phi.push(ao.gradient(*c, r));
            }
        }

        for a in 0..natoms {
            let mut d_rho = [0.0;3];
            let mut d_sig = [0.0;3];

            for mu in 0..nao {
                for nu in 0..nao {
                    let p = density[mu][nu];
                    let dphi =
                        [ grad_phi[mu][0]*phi[nu] + phi[mu]*grad_phi[nu][0],
                          grad_phi[mu][1]*phi[nu] + phi[mu]*grad_phi[nu][1],
                          grad_phi[mu][2]*phi[nu] + phi[mu]*grad_phi[nu][2] ];

                    for k in 0..3 {
                        d_rho[k] += p * dphi[k];
                        d_sig[k] += 2.0 * dp.grad[k] * p * dphi[k];
                    }
                }
            }

            let d_tau = dtau_dra(shells, centers, coeff, n_occ, a, r);

            for k in 0..3 {
                grad[a][k] += weight *
                    ( vrho * d_rho[k] + vsig * d_sig[k] + vtau * d_tau[k] );
            }
        }
    }

    grad
}
