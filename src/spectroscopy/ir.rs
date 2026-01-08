//! Infrared intensities

use nalgebra::{DMatrix, DVector};

/// IR intensities from dipole derivatives and normal modes
///
/// dip_deriv: (3 x 3N) matrix dμ_i / dR_j
/// modes: columns are normalized normal modes
pub fn ir_intensities(
    dip_deriv: &Vec<Vec<f64>>,   // 3 x (3N)
    modes: &Vec<Vec<f64>>,       // (3N x Nmodes)
) -> Vec<f64> {

    let nmode = modes[0].len();
    let mut intens = vec![0.0; nmode];

    let dmu = DMatrix::from_vec(3, dip_deriv[0].len(),
        dip_deriv.iter().flatten().cloned().collect());

    let q = DMatrix::from_vec(
        modes.len(),
        nmode,
        modes.iter().flatten().cloned().collect(),
    );

    let proj = dmu * q; // 3 x Nmodes

    for k in 0..nmode {
        let val =
            proj[(0,k)]*proj[(0,k)] +
            proj[(1,k)]*proj[(1,k)] +
            proj[(2,k)]*proj[(2,k)];
        intens[k] = val;
    }

    intens
}
