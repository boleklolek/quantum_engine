use crate::basis::shell::Shell;
use crate::scf::density::build_spin_density;
use crate::scf::jk::build_jk;
use crate::scf::diis::Diis;
use crate::scf::guess::{core_h_guess, build_matrix};
use crate::scf::utils::{add, solve_roothaan, diis_error};
use crate::integrals::overlap_contracted::overlap_shell_shell;
use crate::integrals::kinetic::kinetic_shell_shell;
use crate::integrals::nuclear_attraction::nuclear_attraction_shell_shell;

/// Run unrestricted Hartree–Fock (UHF)
pub fn run_uhf(
    shells: &[Shell],
    shell_centers: &[[f64; 3]],
    atoms: &[crate::system::atom::Atom],
    n_alpha: usize,
    n_beta: usize,
    max_iter: usize,
    conv_thresh: f64,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, f64) {
    let nao = shells.iter().map(|s| s.n_orbitals()).sum::<usize>();

    // --- Core Hamiltonian ---
    let s = build_matrix(
        shells, 
        shell_centers, 
        |sa, _ca, sb, _cb| {
            overlap_shell_shell(sa, sb)
        },
    );

    let t = build_matrix(shells, shell_centers, kinetic_shell_shell);
    let v = build_matrix(shells, shell_centers, |a, ca, b, cb| {
        nuclear_attraction_shell_shell(a, ca, b, cb, atoms)
    });
    let hcore = add(&t, &v);

    // --- Initial guess (same for alpha/beta) ---
    let p0 = core_h_guess(shells, shell_centers, atoms, n_alpha + n_beta);

    let mut p_alpha = p0.clone();
    let mut p_beta  = p0.clone();

    let mut diis_a = Diis::new(6);
    let mut diis_b = Diis::new(6);

    let mut e_old = 0.0;

    for iter in 0..max_iter {
        // Total density
        let p_tot = add(&p_alpha, &p_beta);

        // J/K from total density
        let (j, _) = build_jk(shells, shell_centers, &p_tot);
        let (_, k_alpha) = build_jk(shells, shell_centers, &p_alpha);
        let (_, k_beta)  = build_jk(shells, shell_centers, &p_beta);

        // Fock matrices
        let f_alpha = build_fock(&hcore, &j, &k_alpha);
        let f_beta  = build_fock(&hcore, &j, &k_beta);

        // DIIS errors
        let err_a = diis_error(&f_alpha, &p_alpha, &s);
        let err_b = diis_error(&f_beta, &p_beta, &s);

        diis_a.push(f_alpha.clone(), err_a);
        diis_b.push(f_beta.clone(), err_b);

        let f_alpha = diis_a.extrapolate().unwrap_or(f_alpha);
        let f_beta  = diis_b.extrapolate().unwrap_or(f_beta);

        // Solve Roothaan
        let (c_a, _) = solve_roothaan(&f_alpha, &s);
        let (c_b, _) = solve_roothaan(&f_beta, &s);

        // New densities
        let p_alpha_new = build_spin_density(&c_a, n_alpha);
        let p_beta_new  = build_spin_density(&c_b, n_beta);

        // Energy
        let e = uhf_energy(&p_alpha_new, &p_beta_new, &hcore, &j, &k_alpha, &k_beta);

        let dE = (e - e_old).abs();
        let dP = rms_diff(&p_alpha, &p_alpha_new)
               + rms_diff(&p_beta,  &p_beta_new);

        println!(
            "UHF iter {:3}  E = {:16.10}  dE = {:10.3e}  dP = {:10.3e}",
            iter, e, dE, dP
        );

        if dE < conv_thresh && dP < conv_thresh {
            return (p_alpha_new, p_beta_new, e);
        }

        p_alpha = p_alpha_new;
        p_beta  = p_beta_new;
        e_old = e;
    }

    panic!("UHF did not converge");
}
