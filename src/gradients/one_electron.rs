use crate::basis::shell::Shell;
use crate::system::atom::Atom;

/// ⟨μ|∂(T+V)|ν⟩ contraction
pub fn grad_one_electron(
    shells: &[Shell],
    shell_centers: &[[f64; 3]],
    density: &Vec<Vec<f64>>,
    atoms: &[Atom],
) -> Vec<[f64; 3]> {

    let mut grad = vec![[0.0; 3]; atoms.len()];

    for (a, atom) in atoms.iter().enumerate() {
        let center = atom.position;

        for (i, shell_i) in shells.iter().enumerate() {
            for (j, shell_j) in shells.iter().enumerate() {

                let d_h = shell_i.grad_kinetic_nuclear(
                    &shell_j,
                    shell_centers[i],
                    shell_centers[j],
                    center,
                );

                for mu in 0..shell_i.orbitals.len() {
                    for nu in 0..shell_j.orbitals.len() {
                        let p = density[shell_i.offset + mu][shell_j.offset + nu];
                        for k in 0..3 {
                            grad[a][k] += p * d_h[k];
                        }
                    }
                }
            }
        }
    }
    grad
}
