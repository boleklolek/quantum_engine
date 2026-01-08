//! Self-Consistent Field (SCF) cycle
//!
//! RHF / DFT / híbridos
//! Shells con AO implícitos (sin `orbitals`)

use nalgebra::DMatrix;
use crate::basis::shell::Shell;
use crate::system::atom::Atom;
use crate::scf::density::build_density;
use crate::scf::jk::build_jk;
use crate::scf::utils::solve_roothaan;
use crate::dft::vxc::{XcMethod, build_vxc};


/// Opciones SCF
pub struct ScfOptions {
    pub max_iter: usize,
    pub conv_tol: f64,
    /// None → HF
    /// Some(XcMethod) → DFT / híbrido
    pub xc_method: Option<XcMethod>,
}

/// Resultado SCF
pub struct ScfResult {
    pub energy: f64,
    pub density: Vec<Vec<f64>>,
}

/// Ciclo SCF principal
pub fn scf_cycle(
    shells: &[Shell],
    atoms: &[Atom],
    nelec: usize,
    h_core: &Vec<Vec<f64>>,   // H = T + V (AO)
    overlap: &Vec<Vec<f64>>,  // S (AO)
    options: &ScfOptions,
) -> ScfResult {

    // Número total de AO
    let nao: usize = shells.iter().map(|s| s.n_orbitals()).sum();

    // Centros de shells (una sola vez)
    let shell_centers: Vec<[f64; 3]> =
        shells.iter().map(|s| s.center).collect();

    // Convertir H y S a DMatrix (nalgebra)
    let h_core_mat = DMatrix::from_fn(nao, nao, |i, j| h_core[i][j]);
    let overlap_mat = DMatrix::from_fn(nao, nao, |i, j| overlap[i][j]);

    // Densidad inicial
    let mut p: Vec<Vec<f64>> = vec![vec![0.0; nao]; nao];

    let mut energy_old = 0.0;

    for iter in 0..options.max_iter {

        // -----------------------------
        // Construcción J y K
        // -----------------------------
        let (j_mat, k_mat) = build_jk(shells, &shell_centers, &p);

        // -----------------------------
        // Construcción Fock
        // -----------------------------
        let mut fock = h_core_mat.clone();

        for i in 0..nao {
            for j in 0..nao {
                fock[(i, j)] += 2.0 * j_mat[i][j] - k_mat[i][j];
            }
        }

        // -----------------------------
        // XC (DFT / híbrido)
        // -----------------------------
        let mut dft_energy_exc = 0.0;
        let mut dft_energy_rho_vxc = 0.0;

        if let Some(xc) = &options.xc_method {

            let (vxc_mat, dft_energy) = build_vxc(
                shells,
                &shell_centers,
                &p,
                None,      // coeff (solo meta-GGA)
                None,      // n_occ
                atoms,
                xc.clone(),
            );

            for i in 0..nao {
                for j in 0..nao {
                    fock[(i, j)] += vxc_mat[i][j];
                }
            }

            dft_energy_exc = dft_energy.exc;
            dft_energy_rho_vxc = dft_energy.int_rho_vxc;
        }

        // -----------------------------
        // Resolver Roothaan
        // -----------------------------
        let (coeff, _eps) = solve_roothaan(&fock, &overlap_mat);

        // -----------------------------
        // Nueva densidad
        // -----------------------------
        let p_new = build_density(coeff, nelec);

        // -----------------------------
        // Energía electrónica
        // -----------------------------
        let mut energy = 0.0;
        for i in 0..nao {
            for j in 0..nao {
                energy += p_new[i][j] * (h_core[i][j] + fock[(i, j)]);
            }
        }
        energy *= 0.5;

        // Corrección DFT: Exc − ∫ρ vxc
        if options.xc_method.is_some() {
            energy += dft_energy_exc - dft_energy_rho_vxc;
        }

        let delta_e = (energy - energy_old).abs();

        println!(
            "SCF iter {:3}  E = {:18.12}  ΔE = {:.3e}",
            iter + 1,
            energy,
            delta_e
        );

        // Convergencia
        if delta_e < options.conv_tol {
            return ScfResult {
                energy,
                density: p_new,
            };
        }

        // Actualizar
        p = p_new;
        energy_old = energy;
    }

    panic!(
        "SCF did not converge after {} iterations",
        options.max_iter
    );
}

