//! Geometry optimization driver

use crate::optimization::{bfgs::Bfgs, lbfgs::Lbfgs};

pub enum Optimizer {
    BFGS,
    LBFGS { m: usize },
}

/// Generic geometry optimizer
///
/// eval(x) must return (energy, gradient)
pub fn optimize(
    mut x: Vec<f64>,
    eval: &dyn Fn(&Vec<f64>) -> (f64, Vec<f64>),
    optimizer: Optimizer,
    max_iter: usize,
    grad_tol: f64,
) -> (Vec<f64>, f64) {

    let (mut f, mut g) = eval(&x);

    match optimizer {
        Optimizer::BFGS => {
            let mut opt = Bfgs::new(x.len());
            for _ in 0..max_iter {
                if norm(&g) < grad_tol {
                    break;
                }
                let (x_new, f_new, g_new) = opt.step(&x, f, &g, eval);
                x = x_new; f = f_new; g = g_new;
            }
        }

        Optimizer::LBFGS { m } => {
            let mut opt = Lbfgs::new(m);
            for _ in 0..max_iter {
                if norm(&g) < grad_tol {
                    break;
                }
                let (x_new, f_new, g_new) = opt.step(&x, f, &g, eval);
                x = x_new; f = f_new; g = g_new;
            }
        }
    }

    (x, f)
}

fn norm(g: &Vec<f64>) -> f64 {
    g.iter().map(|x| x*x).sum::<f64>().sqrt()
}
