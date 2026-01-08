//! Shell–shell electron repulsion integrals (ERI)
//!
//! Devuelve el bloque (μν|λσ) para dos shells

use crate::basis::shell::Shell;
use crate::basis::contracted::Contracted;
use crate::integrals::eri::eri_contracted::eri_contracted;
use crate::integrals::schwarz::schwarz_shell_pair;

/// ERI block between two shells
///
/// (μν|λσ) where μ,ν ∈ shell A and λ,σ ∈ shell B
pub fn eri_shell_shell(
    shell_a: &Shell,
    shell_b: &Shell,
) -> Vec<Vec<f64>> {

    let na = shell_a.n_orbitals();
    let nb = shell_b.n_orbitals();

    let mut eri = vec![vec![0.0_f64; nb]; na];

    // Cada AO cartesiano comparte el mismo conjunto de primitivas
    let ca = Contracted::new(shell_a.primitives.clone());
    let cb = Contracted::new(shell_b.primitives.clone());

    for i in 0..na {
        for j in 0..nb {
            eri[i][j] = eri_contracted(&ca, &cb, &ca, &cb);
        }
    }

    eri
}

/// 4-shell ERI block (μν|λσ)
///
/// Devuelve el tensor aplanado:
/// ((μ * nb + ν) * nc + λ) * nd + σ
pub fn eri_shell_shell_shell_shell(
    shell_a: &Shell,
    shell_b: &Shell,
    shell_c: &Shell,
    shell_d: &Shell,
) -> Vec<f64> {

    let na = shell_a.n_orbitals();
    let nb = shell_b.n_orbitals();
    let nc = shell_c.n_orbitals();
    let nd = shell_d.n_orbitals();

    let mut eri = vec![0.0_f64; na * nb * nc * nd];

    // Schwarz screening (correcto)
    let bound_ab = schwarz_shell_pair(shell_a, shell_b);
    let bound_cd = schwarz_shell_pair(shell_c, shell_d);
    let cutoff = 1e-12;

    if bound_ab * bound_cd < cutoff {
        return eri;
    }

    let ao_a = Contracted::new(shell_a.primitives.clone());
    let ao_b = Contracted::new(shell_b.primitives.clone());
    let ao_c = Contracted::new(shell_c.primitives.clone());
    let ao_d = Contracted::new(shell_d.primitives.clone());

    let idx = |i, j, k, l| ((i * nb + j) * nc + k) * nd + l;

    for i in 0..na {
        for j in 0..nb {
            for k in 0..nc {
                for l in 0..nd {
                    eri[idx(i, j, k, l)] = eri_contracted(
                        &ao_a,
                        &ao_b,
                        &ao_c,
                        &ao_d,
                    );
                }
            }
        }
    }

    eri
}

