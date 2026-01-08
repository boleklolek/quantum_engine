use crate::basis::shell::Shell;
use crate::integrals::eri::eri_contracted::eri_shell_shell_shell_shell;

/// Build Coulomb (J) and Exchange (K) matrices
///
/// shells        : basis shells
/// shell_centers : centers of shells (aligned with shells)
/// density       : full AO density matrix P
///
/// Returns (J, K)
pub fn build_jk(
    shells: &[Shell],
    shell_centers: &[[f64; 3]],
    density: &Vec<Vec<f64>>,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let nao = density.len();

    let mut j = vec![vec![0.0; nao]; nao];
    let mut k = vec![vec![0.0; nao]; nao];

    // AO index offset per shell
    let mut shell_offsets = Vec::new();
    let mut offset = 0;
    for sh in shells {
        shell_offsets.push(offset);
        offset += sh.n_orbitals();
    }

    let nshells = shells.len();

    // Loop over shell quartets
    for a in 0..nshells {
        for b in 0..nshells {
            for c in 0..nshells {
                for d in 0..nshells {

                    let eri_block = eri_shell_shell_shell_shell(
                        &shells[a],
                        &shells[b],
                        &shells[c],
                        &shells[d],
                    );

                    let na = shells[a].n_orbitals();
                    let nb = shells[b].n_orbitals();
                    let nc = shells[c].n_orbitals();
                    let nd = shells[d].n_orbitals();

                    let oa = shell_offsets[a];
                    let ob = shell_offsets[b];
                    let oc = shell_offsets[c];
                    let od = shell_offsets[d];

                    let idx = |i, j, k, l| ((i * nb + j) * nc + k) * nd + l;

                    for ia in 0..na {
                        for ib in 0..nb {
                            let mu = oa + ia;
                            let nu = ob + ib;

                            for ic in 0..nc {
                                for id in 0..nd {
                                    let lam = oc + ic;
                                    let sig = od + id;

                                    let eri = eri_block[idx(ia, ib, ic, id)];
                                    let p = density[lam][sig];

                                    // Coulomb
                                    j[mu][nu] += p * eri;

                                    // Exchange
                                    k[mu][lam] += density[nu][sig] * eri;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    (j, k)
}
