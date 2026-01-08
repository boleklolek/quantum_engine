//! Spin-polarized DFT SCF driver
use crate::integrals::nuclear_attraction::nuclear_attraction_shell_shell;
use crate::basis::shell::Shell;
use crate::system::atom::Atom;
use crate::integrals::overlap_contracted::overlap_shell_shell;
use crate::scf::density::{build_spin_density,rms_density_diff};
use crate::scf::jk::build_jk;
use crate::scf::diis::Diis;
use crate::scf::guess::core_h_guess;
use crate::scf::utils::*;
use crate::integrals::kinetic::kinetic_shell_shell;
use crate::dft::vxc::{build_vxc_udft, XcMethod, DftEnergy};

/// Run UDFT SCF
pub fn run_udft(
    shells: &[Shell],
    shell_centers: &[[f64; 3]],
    atoms: &[Atom],
    n_alpha: usize,
    n_beta: usize,
    xc: XcMethod,
    max_iter: usize,
    conv: f64,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, f64) {

    let nao: usize = shells.iter().map(|s| s.orbitals.len()).sum();

    // 1e integrals
    let s = build_one_electron_matrix(shells, shell_centers, overlap_shell_shell);
    let t = build_one_electron_matrix(shells, shell_centers, kinetic_shell_shell);
    let v = build_one_electron_matrix(shells, shell_centers, |a, ca, b, cb| {
        nuclear_attraction_shell_shell(a, ca, b, cb, atoms)
    });

    let hcore = add(&t, &v);

    let p0 = core_h_guess(shells, shell_centers, atoms, n_alpha + n_beta);
    let mut p_alpha = p0.clone();
    let mut p_beta = p0.clone();

    let mut diis_a = Diis::new(6);
    let mut diis_b = Diis::new(6);

    let mut e_old = 0.0;
    let mut dft_energy: Option<DftEnergy> = None;

    let hf_frac = match xc {
        XcMethod::Hybrid { hyb, .. } => hyb.hf_fraction(),
        _ => 0.0,
    };

    for iter in 0..max_iter {
        let p_tot = add(&p_alpha, &p_beta);

        let (j, _) = build_jk(shells, shell_centers, &p_tot);
        let (_, k_a) = build_jk(shells, shell_centers, &p_alpha);
        let (_, k_b) = build_jk(shells, shell_centers, &p_beta);

        let mut f_a = build_fock_scaled(&hcore, &j, &k_a, hf_frac);
        let mut f_b = build_fock_scaled(&hcore, &j, &k_b, hf_frac);

        let (vxa, vxb, e_dft) =
            build_vxc_udft(shells, shell_centers, &p_alpha, &p_beta, xc);

        add_inplace(&mut f_a, &vxa);
        add_inplace(&mut f_b, &vxb);

        dft_energy = Some(e_dft);

        let err_a = diis_error(&f_a, &p_alpha, &s);
        let err_b = diis_error(&f_b, &p_beta, &s);

        diis_a.push(f_a.clone(), err_a);
        diis_b.push(f_b.clone(), err_b);

        let f_a = diis_a.extrapolate().unwrap_or(f_a);
        let f_b = diis_b.extrapolate().unwrap_or(f_b);

        let (c_a, _) = solve_roothaan(&f_a, &s);
        let (c_b, _) = solve_roothaan(&f_b, &s);

        let p_alpha_new = build_spin_density(&c_a, n_alpha);
        let p_beta_new = build_spin_density(&c_b, n_beta);

        let mut e =
            electronic_energy_scaled(&p_tot, &hcore, &j, &k_a, hf_frac)
          + electronic_energy_scaled(&p_tot, &hcore, &j, &k_b, hf_frac);

        if let Some(ref ed) = dft_energy {
            e += ed.exc - ed.int_rho_vxc;
        }

        let dE = (e - e_old).abs();
        let dP =
            rms_density_diff(&p_alpha, &p_alpha_new) +
            rms_density_diff(&p_beta, &p_beta_new);

        println!(
            "UDFT {:3}  E = {:16.10} dE = {:9.3e} dP = {:9.3e}",
            iter, e, dE, dP
        );

        if dE < conv && dP < conv {
            return (p_alpha_new, p_beta_new, e);
        }

        p_alpha = p_alpha_new;
        p_beta = p_beta_new;
        e_old = e;
    }

    panic!("UDFT did not converge");
}
