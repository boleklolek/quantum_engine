//! Hessian via finite differences of analytic gradients

pub fn hessian_fd(
    x: &Vec<f64>,
    grad: &dyn Fn(&Vec<f64>) -> Vec<f64>,
    step: f64,
) -> Vec<Vec<f64>> {

    let n = x.len();
    let mut h = vec![vec![0.0; n]; n];

    for i in 0..n {
        let mut x_p = x.clone();
        let mut x_m = x.clone();

        x_p[i] += step;
        x_m[i] -= step;

        let g_p = grad(&x_p);
        let g_m = grad(&x_m);

        for j in 0..n {
            h[j][i] = (g_p[j] - g_m[j]) / (2.0 * step);
        }
    }

    // Symmetrize
    for i in 0..n {
        for j in 0..i {
            let avg = 0.5 * (h[i][j] + h[j][i]);
            h[i][j] = avg;
            h[j][i] = avg;
        }
    }

    h
}
