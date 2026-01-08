//! Static polarizability via finite electric fields

use crate::spectroscopy::dipole::dipole_moment;

pub fn polarizability(
    field: f64,
    eval_mu: &dyn Fn(&[f64;3]) -> [f64;3],
) -> [[f64;3];3] {

    let mut alpha = [[0.0;3];3];

    for j in 0..3 {
        let mut e = [0.0;3];
        e[j] = field;

        let mu_p = eval_mu(&e);
        let mu_m = eval_mu(&[-e[0], -e[1], -e[2]]);

        for i in 0..3 {
            alpha[i][j] = (mu_p[i] - mu_m[i]) / (2.0 * field);
        }
    }
    alpha
}

