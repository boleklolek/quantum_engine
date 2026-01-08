use crate::basis::shell::Shell;

/// −Σ F_μν ∂S_μν/∂R_A
pub fn grad_overlap_pulay(
    shells: &[Shell],
    fock: &Vec<Vec<f64>>,
    natoms: usize,
) -> Vec<[f64;3]> {

    let mut grad = vec![[0.0;3]; natoms];

    for shell_i in shells {
        for shell_j in shells {
            let dS = shell_i.grad_overlap(&shell_j);

            for mu in 0..shell_i.orbitals.len() {
                for nu in 0..shell_j.orbitals.len() {
                    let f = fock[shell_i.offset + mu][shell_j.offset + nu];
                    for a in 0..natoms {
                        for k in 0..3 {
                            grad[a][k] -= f * dS[a][k];
                        }
                    }
                }
            }
        }
    }
    grad
}
