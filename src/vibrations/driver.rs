//! Vibrational analysis driver

use crate::vibrations::{
    hessian_fd::hessian_fd,
    projector::project_tr_rotation,
    mass::mass_weight_hessian,
    frequencies::vibrational_frequencies,
};

pub fn compute_frequencies(
    coords: &Vec<f64>,
    masses: &Vec<f64>,
    gradient: &dyn Fn(&Vec<f64>) -> Vec<f64>,
) -> Vec<f64> {

    let h = hessian_fd(coords, gradient, 1e-3);
    let h_proj = project_tr_rotation(&h, coords, masses);
    let h_mw = mass_weight_hessian(&h_proj, masses);
    vibrational_frequencies(&h_mw)
}
