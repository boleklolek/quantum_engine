use crate::basis::contracted::Contracted;
use crate::basis::primitive::Primitive;
use crate::basis::shell::Shell;

use crate::integrals::eri::eri_ssss::eri_ssss;
use crate::integrals::eri::eri_vrr::{eri_psss, eri_ppss, eri_dsss, eri_fsss};
use crate::integrals::eri::eri_hrr::hrr_ab;
use crate::integrals::schwarz::schwarz_shell_pair;

/// AO–AO contracted ERI ⟨ab|cd⟩
pub fn eri_ao_ao(
    ao_a: &Contracted,
    ao_b: &Contracted,
    ao_c: &Contracted,
    ao_d: &Contracted,
) -> f64 {
    let mut value = 0.0_f64;

    for pa in &ao_a.primitives {
        for pb in &ao_b.primitives {
            for pc in &ao_c.primitives {
                for pd in &ao_d.primitives {
                    value += eri_primitive_dispatch(pa, pb, pc, pd);
                }
            }
        }
    }

    value
}

/// Primitive ERI dispatcher (VRR on A)
fn eri_primitive_dispatch(
    a: &Primitive,
    b: &Primitive,
    c: &Primitive,
    d: &Primitive,
) -> f64 {
    let [la, ma, na] = a.ang();
    let lsum = la + ma + na;

    match lsum {
        // (ss|ss)
        0 => eri_ssss(a, b, c, d),

        // (ps|ss)
        1 => {
            let dir = if la == 1 { 0 } else if ma == 1 { 1 } else { 2 };
            eri_psss(a, b, c, d, dir)
        }

        // (pp|ss)
        2 => {
            let (i, j) = cartesian_pair(la, ma, na);
            eri_ppss(a, b, c, d, i, j)
        }

        // (ds|ss)
        3 => {
            let (i, j) = cartesian_pair(la, ma, na);
            eri_dsss(a, b, c, d, i, j)
        }

        // (fs|ss)
        4 => {
            let (i, j, k) = cartesian_triplet(la, ma, na);
            eri_fsss(a, b, c, d, i, j, k)
        }

        _ => panic!("Angular momentum > f not supported"),
    }
}

/// Shell–shell ERI block (μν|μν) for Schwarz / J / K
///
/// Devuelve un bloque aplanado (μν|λσ) con screening
pub fn eri_shell_shell(
    shell_a: &Shell,
    shell_b: &Shell,
) -> Vec<f64> {

    let na = shell_a.n_orbitals();
    let nb = shell_b.n_orbitals();

    let mut eri = vec![0.0_f64; na * nb * na * nb];

    // Schwarz screening (correcto: shell–shell)
    let bound = schwarz_shell_pair(shell_a, shell_b);
    let cutoff = 1e-12;

    if bound < cutoff {
        return eri;
    }

    let ao_a = Contracted::new(shell_a.primitives.clone());
    let ao_b = Contracted::new(shell_b.primitives.clone());

    let idx = |i, j, k, l| ((i * nb + j) * na + k) * nb + l;

    for i in 0..na {
        for j in 0..nb {
            for k in 0..na {
                for l in 0..nb {
                    eri[idx(i, j, k, l)] = eri_ao_ao(
                        &ao_a,
                        &ao_b,
                        &ao_a,
                        &ao_b,
                    );
                }
            }
        }
    }

    eri
}

/// Map (l,m,n) → pair (p/d)
fn cartesian_pair(l: usize, m: usize, n: usize) -> (usize, usize) {
    let mut v = Vec::new();
    for (i, &c) in [l, m, n].iter().enumerate() {
        for _ in 0..c {
            v.push(i);
        }
    }
    (v[0], v[1])
}

/// Map (l,m,n) → triplet (f)
fn cartesian_triplet(l: usize, m: usize, n: usize) -> (usize, usize, usize) {
    let mut v = Vec::new();
    for (i, &c) in [l, m, n].iter().enumerate() {
        for _ in 0..c {
            v.push(i);
        }
    }
    (v[0], v[1], v[2])
}
/// Contracted ERI ⟨ab|cd⟩
///
/// Wrapper estable usado por shell–shell, Schwarz, J/K, MPI
#[inline]
pub fn eri_contracted(
    a: &Contracted,
    b: &Contracted,
    c: &Contracted,
    d: &Contracted,
) -> f64 {
    eri_ao_ao(a, b, c, d)
}
/// Public shell–shell–shell–shell ERI interface
///
/// This is the entry point used by J/K builders and SCF.
/// Internally it uses full AO contraction, VRR, HRR and Schwarz screening.
#[inline]
pub fn eri_shell_shell_shell_shell(
    shell_a: &Shell,
    shell_b: &Shell,
    shell_c: &Shell,
    shell_d: &Shell,
) -> Vec<f64> {
    // Centers are implicit in Primitive / Shell
    // This wrapper preserves the original full ERI capability
    super::eri_shell::eri_shell_shell_shell_shell(
        shell_a,
        shell_b,
        shell_c,
        shell_d,
    )
}

