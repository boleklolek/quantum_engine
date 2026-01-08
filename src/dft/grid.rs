//! Numerical integration grid for DFT
//!
//! Provides atom-centered grids with radial + angular sampling.

use std::f64::consts::PI;
use crate::system::atom::Atom;

/// One grid point
#[derive(Clone)]
pub struct GridPoint {
    pub r: [f64; 3],
    pub weight: f64,
}

/// Complete molecular grid
pub struct DftGrid {
    pub points: Vec<GridPoint>,
}

impl DftGrid {
    /// Build molecular grid (sum of atomic grids)
    pub fn new(atoms: &[Atom], radial: usize, angular: usize) -> Self {
        let mut points = Vec::new();

        for atom in atoms {
            let atomic = atomic_grid(atom, radial, angular);
            points.extend(atomic);
        }

        Self { points }
    }
}

/// Build an atomic-centered grid
fn atomic_grid(atom: &Atom, n_radial: usize, n_ang: usize) -> Vec<GridPoint> {
    let mut pts = Vec::new();

    let r_max = 10.0; // bohr, enough for valence density

    for i in 0..n_radial {
        // Simple Gauss–Legendre–like radial grid
        let xi = (i as f64 + 0.5) / n_radial as f64;
        let r = r_max * xi * xi; // quadratic map
        let w_r = 2.0 * r_max * xi / n_radial as f64;

        for j in 0..n_ang {
            let theta = PI * (j as f64 + 0.5) / n_ang as f64;
            let sin_t = theta.sin();
            let cos_t = theta.cos();

            for k in 0..n_ang {
                let phi = 2.0 * PI * (k as f64 + 0.5) / n_ang as f64;

                let x = r * sin_t * phi.cos();
                let y = r * sin_t * phi.sin();
                let z = r * cos_t;

                let w_ang = 4.0 * PI / (n_ang * n_ang) as f64;

                pts.push(GridPoint {
                    r: [
                        atom.position[0] + x,
                        atom.position[1] + y,
                        atom.position[2] + z,
                    ],
                    weight: w_r * w_ang,
                });
            }
        }
    }

    pts
}
