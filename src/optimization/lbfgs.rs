//! Limited-memory BFGS (L-BFGS)

use crate::optimization::line_search::line_search;
use std::collections::VecDeque;

pub struct Lbfgs {
    m: usize,
    s_list: VecDeque<Vec<f64>>,
    y_list: VecDeque<Vec<f64>>,
}

impl Lbfgs {
    pub fn new(m: usize) -> Self {
        Self {
            m,
            s_list: VecDeque::new(),
            y_list: VecDeque::new(),
        }
    }

    fn two_loop(&self, g: &Vec<f64>) -> Vec<f64> {
        let mut q = g.clone();
        let mut alpha = Vec::new();

        for (s,y) in self.s_list.iter().zip(self.y_list.iter()).rev() {
            let rho = 1.0 / y.iter().zip(s.iter()).map(|(a,b)| a*b).sum::<f64>();
            let a = rho * s.iter().zip(q.iter()).map(|(a,b)| a*b).sum::<f64>();
            alpha.push(a);
            for i in 0..q.len() { q[i] -= a * y[i]; }
        }

        // Initial H0 ≈ I
        let mut r = q.clone();

        for ((s,y), a) in self.s_list.iter().zip(self.y_list.iter()).zip(alpha.into_iter().rev()) {
            let rho = 1.0 / y.iter().zip(s.iter()).map(|(a,b)| a*b).sum::<f64>();
            let b = rho * y.iter().zip(r.iter()).map(|(a,b)| a*b).sum::<f64>();
            for i in 0..r.len() { r[i] += s[i] * (a - b); }
        }

        r.iter().map(|x| -x).collect()
    }

    pub fn step(
        &mut self,
        x: &Vec<f64>,
        f: f64,
        g: &Vec<f64>,
        eval: &dyn Fn(&Vec<f64>) -> (f64, Vec<f64>),
    ) -> (Vec<f64>, f64, Vec<f64>) {

        let p = self.two_loop(g);
        let alpha = line_search(x, f, g, &p, eval);

        let x_new: Vec<f64> = x.iter().zip(p.iter()).map(|(xi,pi)| xi + alpha*pi).collect();
        let (f_new, g_new) = eval(&x_new);

        let s: Vec<f64> = x_new.iter().zip(x.iter()).map(|(a,b)| a-b).collect();
        let y: Vec<f64> = g_new.iter().zip(g.iter()).map(|(a,b)| a-b).collect();

        if self.s_list.len() == self.m {
            self.s_list.pop_front();
            self.y_list.pop_front();
        }

        self.s_list.push_back(s);
        self.y_list.push_back(y);

        (x_new, f_new, g_new)
    }
}
