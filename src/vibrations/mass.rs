//! Mass-weighting utilities

pub fn mass_weight_hessian(
    hessian: &Vec<Vec<f64>>,
    masses: &Vec<f64>,
) -> Vec<Vec<f64>> {

    let n = hessian.len();
    let natoms = masses.len();
    let mut mw = vec![vec![0.0; n]; n];

    for i in 0..n {
        let mi = masses[i/3];
        for j in 0..n {
            let mj = masses[j/3];
            mw[i][j] = hessian[i][j] / (mi * mj).sqrt();
        }
    }
    mw
}
