//! Kinetic energy density τ and its nuclear derivative helpers

use crate::basis::shell::Shell;

/// τ(r) from occupied orbitals (RHF/DFT)
pub fn tau_at_point(
    shells: &[Shell],
    centers: &[[f64;3]],
    coeff: &Vec<Vec<f64>>,   // C_{μi}
    n_occ: usize,
    r: [f64;3],
) -> f64 {
    let nao = coeff.len();

    // AO gradients
    let mut grad_phi = Vec::with_capacity(nao);
    for (sh, c) in shells.iter().zip(centers.iter()) {
        for ao in &sh.orbitals {
            grad_phi.push(ao.gradient(*c, r));
        }
    }

    let mut tau = 0.0;
    for i in 0..n_occ {
        let mut gpsi = [0.0;3];
        for mu in 0..nao {
            for k in 0..3 {
                gpsi[k] += coeff[mu][i] * grad_phi[mu][k];
            }
        }
        tau += gpsi[0]*gpsi[0] + gpsi[1]*gpsi[1] + gpsi[2]*gpsi[2];
    }
    tau
}

/// ∂τ/∂R_A (AO-based; Pulay XC included implicitly)
pub fn dtau_dra(
    shells: &[Shell],
    centers: &[[f64;3]],
    coeff: &Vec<Vec<f64>>,
    n_occ: usize,
    atom_idx: usize,
    r: [f64;3],
) -> [f64;3] {
    let nao = coeff.len();

    let mut grad_phi = Vec::with_capacity(nao);
    let mut hess_phi = Vec::with_capacity(nao); // second derivatives wrt nuclear motion proxy

    for (sh, c) in shells.iter().zip(centers.iter()) {
        for ao in &sh.orbitals {
            grad_phi.push(ao.gradient(*c, r));
            hess_phi.push(ao.hessian(*c, r)); // assume available (or finite diff)
        }
    }

    let mut d = [0.0;3];
    for i in 0..n_occ {
        let mut gpsi = [0.0;3];
        let mut dgpsi = [0.0;3];

        for mu in 0..nao {
            for k in 0..3 {
                gpsi[k] += coeff[mu][i] * grad_phi[mu][k];
                dgpsi[k] += coeff[mu][i] * hess_phi[mu][k];
            }
        }

        for k in 0..3 {
            d[k] += 2.0 * gpsi[k] * dgpsi[k];
        }
    }
    d
}

