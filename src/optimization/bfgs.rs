//! Full BFGS geometry optimizer

use crate::optimization::line_search::line_search;

pub struct Bfgs {
    h_inv: Vec<Vec<f64>>,
}

impl Bfgs {
    pub fn new(dim: usize) -> Self {
        let mut h_inv = vec![vec![0.0; dim]; dim];
        for i in 0..dim {
            h_inv[i][i] = 1.0;
        }
        Self { h_inv }
    }

    pub fn step(
        &mut self,
        x: &Vec<f64>,
        f: f64,
        g: &Vec<f64>,
        eval: &dyn Fn(&Vec<f64>) -> (f64, Vec<f64>),
    ) -> (Vec<f64>, f64, Vec<f64>) {

        let dim = x.len();

        // p = -H^{-1} g
        let mut p = vec![0.0; dim];
        for i in 0..dim {
            for j in 0..dim {
                p[i] -= self.h_inv[i][j] * g[j];
            }
        }

        let alpha = line_search(x, f, g, &p, eval);

        let x_new: Vec<f64> = x.iter().zip(p.iter()).map(|(xi,pi)| xi + alpha*pi).collect();
        let (f_new, g_new) = eval(&x_new);

        // BFGS update
        let s: Vec<f64> = x_new.iter().zip(x.iter()).map(|(a,b)| a-b).collect();
        let y: Vec<f64> = g_new.iter().zip(g.iter()).map(|(a,b)| a-b).collect();

        let ys: f64 = y.iter().zip(s.iter()).map(|(a,b)| a*b).sum();

        if ys > 1e-10 {
            let rho = 1.0 / ys;
            let mut hy = vec![0.0; dim];
            for i in 0..dim {
                for j in 0..dim {
                    hy[i] += self.h_inv[i][j] * y[j];
                }
            }

            for i in 0..dim {
                for j in 0..dim {
                    self.h_inv[i][j] +=
                        (1.0 + rho * y.iter().zip(hy.iter()).map(|(a,b)| a*b).sum::<f64>())
                        * rho * s[i] * s[j]
                        - rho * (s[i] * hy[j] + hy[i] * s[j]);
                }
            }
        }

        (x_new, f_new, g_new)
    }
}
