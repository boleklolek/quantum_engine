//! Vibrational frequencies from mass-weighted Hessian

use nalgebra::{DMatrix, SymmetricEigen};

const AU_TO_CM: f64 = 5140.48;

pub fn vibrational_frequencies(
    hessian_mw: &Vec<Vec<f64>>,
) -> Vec<f64> {

    let n = hessian_mw.len();
    let h = DMatrix::from_vec(n, n, hessian_mw.iter().flatten().cloned().collect());

    let eig = SymmetricEigen::new(h);
    eig.eigenvalues
        .iter()
        .map(|&x| {
            if x < 0.0 {
                -(-x).sqrt() * AU_TO_CM
            } else {
                x.sqrt() * AU_TO_CM
            }
        })
        .collect()
}
