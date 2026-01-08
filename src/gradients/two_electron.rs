use crate::basis::shell::Shell;

/// ERI-gradient contraction
pub fn grad_two_electron(
    shells: &[Shell],
    density: &Vec<Vec<f64>>,
    eri_grad: &dyn Fn(usize, usize, usize, usize, usize) -> [f64;3],
    natoms: usize,
) -> Vec<[f64;3]> {

    let nao = density.len();
    let mut grad = vec![[0.0;3]; natoms];

    for mu in 0..nao {
        for nu in 0..nao {
            for lam in 0..nao {
                for sig in 0..nao {
                    let p = density[mu][nu] * density[lam][sig];
                    if p.abs() < 1e-14 { continue; }

                    for a in 0..natoms {
                        let dg = eri_grad(mu,nu,lam,sig,a);
                        for k in 0..3 {
                            grad[a][k] += 0.5 * p * dg[k];
                        }
                    }
                }
            }
        }
    }
    grad
}
