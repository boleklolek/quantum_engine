//! Contracted Gaussian atomic orbitals
//!
//! Una función AO contraída:
//!   χ(r) = Σ_p φ_p(r)
//! donde φ_p son primitivas gaussianas

use crate::basis::primitive::Primitive;

/// One contracted Cartesian Gaussian atomic orbital
#[derive(Clone, Debug)]
pub struct Contracted {
    /// Primitives forming this contracted AO
    pub primitives: Vec<Primitive>,
}

impl Contracted {
    /// Create a new contracted AO from primitives
    pub fn new(primitives: Vec<Primitive>) -> Self {
        Self { primitives }
    }

    /// Value χ(r)
    #[inline]
    pub fn value(&self, r: [f64; 3]) -> f64 {
        self.primitives
            .iter()
            .map(|p| p.value(r))
            .sum()
    }

    /// Gradient ∇χ(r)
    #[inline]
    pub fn gradient(&self, r: [f64; 3]) -> [f64; 3] {
        let mut g = [0.0; 3];
        for p in &self.primitives {
            let gp = p.gradient(r);
            g[0] += gp[0];
            g[1] += gp[1];
            g[2] += gp[2];
        }
        g
    }
}

