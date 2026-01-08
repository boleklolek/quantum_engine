//! Molecular dipole moment

use crate::basis::shell::Shell;
use crate::system::atom::Atom;
use crate::integrals::dipole::dipole_integrals;

/// μ = μ_nuclear + μ_electronic
pub fn dipole_moment(
    shells: &[Shell],
    centers: &[[f64;3]],
    density: &Vec<Vec<f64>>,
    atoms: &[Atom],
) -> [f64;3] {

    let nao = density.len();
    let mut mu_e = [0.0;3];

    let dip = dipole_integrals(shells, centers);

    for mu in 0..nao {
        for nu in 0..nao {
            for k in 0..3 {
                mu_e[k] -= density[mu][nu] * dip[k][mu][nu];
            }
        }
    }

    let mut mu_n = [0.0;3];
    for a in atoms {
        for k in 0..3 {
            mu_n[k] += a.charge as f64 * a.position[k];
        }
    }

    [
        mu_n[0] + mu_e[0],
        mu_n[1] + mu_e[1],
        mu_n[2] + mu_e[2],
    ]
}

