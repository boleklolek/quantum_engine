use crate::basis::shell::Shell;
use crate::integrals::overlap_contracted::overlap_shell_shell;
use crate::integrals::kinetic::kinetic_shell_shell;
use crate::integrals::nuclear_attraction::nuclear_attraction_shell_shell;
use crate::scf::density::build_density;
use crate::scf::utils::solve_roothaan;

/// Build initial density using Core-Hamiltonian guess
pub fn core_h_guess(
    shells: &[Shell],
    shell_centers: &[[f64; 3]],
    atoms: &[crate::system::atom::Atom],
    n_electrons: usize,
) -> Vec<Vec<f64>> {
    let nao = shells.iter().map(|s| s.orbitals.len()).sum::<usize>();

    // Build S, T, V
    let s = build_matrix(shells, shell_centers, |a, ca, b, cb| {
        overlap_shell_shell(a, ca, b, cb)
    });
    let t = build_matrix(shells, shell_centers, |a, ca, b, cb| {
        kinetic_shell_shell(a, ca, b, cb)
    });
    let v = build_matrix(shells, shell_centers, |a, ca, b, cb| {
        nuclear_attraction_shell_shell(a, ca, b, cb, atoms)
    });

    let hcore = add(&t, &v);

    // Solve Hcore C = S C eps
    let (coeff, _) = solve_roothaan(&hcore, &s);

    // Build density
    build_density(&coeff, n_electrons)
}

pub fn build_matrix<F>(
    shells: &[Shell],
    centers: &[[f64; 3]],
    kernel: F,
) -> Vec<Vec<f64>>
where
    F: Fn(&Shell, [f64; 3], &Shell, [f64; 3]) -> Vec<Vec<f64>>,
{
    let nao = shells.iter().map(|s| s.orbitals.len()).sum::<usize>();
    let mut mat = vec![vec![0.0; nao]; nao];

    let mut offsets = Vec::new();
    let mut off = 0;
    for sh in shells {
        offsets.push(off);
        off += sh.orbitals.len();
    }

    for i in 0..shells.len() {
        for j in 0..shells.len() {
            let block = kernel(&shells[i], centers[i], &shells[j], centers[j]);
            for a in 0..block.len() {
                for b in 0..block[0].len() {
                    mat[offsets[i] + a][offsets[j] + b] = block[a][b];
                }
            }
        }
    }

    mat
}

pub fn add(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut c = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    c
}
