//! Simple backtracking line search with Wolfe-like condition

pub fn line_search(
    x: &Vec<f64>,
    f: f64,
    g: &Vec<f64>,
    p: &Vec<f64>,
    eval: &dyn Fn(&Vec<f64>) -> (f64, Vec<f64>),
) -> f64 {

    let c1 = 1e-4;
    let mut alpha = 1.0;

    let dot_gp: f64 = g.iter().zip(p.iter()).map(|(a,b)| a*b).sum();

    loop {
        let x_new: Vec<f64> = x.iter()
            .zip(p.iter())
            .map(|(xi, pi)| xi + alpha * pi)
            .collect();

        let (f_new, _) = eval(&x_new);

        if f_new <= f + c1 * alpha * dot_gp {
            return alpha;
        }

        alpha *= 0.5;
        if alpha < 1e-8 {
            return alpha;
        }
    }
}
