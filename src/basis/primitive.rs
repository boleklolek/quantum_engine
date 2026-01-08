//! Primitive Gaussian functions

use std::f64::consts::PI;

/// One primitive Cartesian Gaussian
#[derive(Clone, Debug)]
pub struct Primitive {
    exponent: f64,
    coefficient: f64,
    center: [f64; 3],
    ang: [usize; 3],   // (lx, ly, lz)
    norm: f64,
}

impl Primitive {
    /// Create new primitive Gaussian
    pub fn new(
        exponent: f64,
        coefficient: f64,
        center: [f64; 3],
        ang: [usize; 3],
    ) -> Self {
        let norm = Self::normalization(exponent, ang);
        Self {
            exponent,
            coefficient,
            center,
            ang,
            norm,
        }
    }

    /// Value of primitive Gaussian at point r
    ///
    /// φ(r) = N * (x-Ax)^lx (y-Ay)^ly (z-Az)^lz * exp(-α |r-A|²)
    pub fn value(&self, r: [f64; 3]) -> f64 {
        let dx = r[0] - self.center[0];
        let dy = r[1] - self.center[1];
        let dz = r[2] - self.center[2];

        let poly =
            dx.powi(self.ang[0] as i32) *
            dy.powi(self.ang[1] as i32) *
            dz.powi(self.ang[2] as i32);

        let r2 = dx*dx + dy*dy + dz*dz;

        self.coefficient * self.norm * poly * (-self.exponent * r2).exp()
    }

    /// Gradient ∇φ(r)
    pub fn gradient(&self, r: [f64; 3]) -> [f64; 3] {
        let dx = r[0] - self.center[0];
        let dy = r[1] - self.center[1];
        let dz = r[2] - self.center[2];

        let r2 = dx*dx + dy*dy + dz*dz;
        let exp = (-self.exponent * r2).exp();

        let mut grad = [0.0; 3];

        let l = self.ang;

        // d/dx
        if l[0] > 0 {
            grad[0] += (l[0] as f64) * dx.powi(l[0] as i32 - 1)
                * dy.powi(l[1] as i32)
                * dz.powi(l[2] as i32);
        }
        grad[0] -= 2.0 * self.exponent * dx
            * dx.powi(l[0] as i32)
            * dy.powi(l[1] as i32)
            * dz.powi(l[2] as i32);

        // d/dy
        if l[1] > 0 {
            grad[1] += (l[1] as f64) * dx.powi(l[0] as i32)
                * dy.powi(l[1] as i32 - 1)
                * dz.powi(l[2] as i32);
        }
        grad[1] -= 2.0 * self.exponent * dy
            * dx.powi(l[0] as i32)
            * dy.powi(l[1] as i32)
            * dz.powi(l[2] as i32);

        // d/dz
        if l[2] > 0 {
            grad[2] += (l[2] as f64) * dx.powi(l[0] as i32)
                * dy.powi(l[1] as i32)
                * dz.powi(l[2] as i32 - 1)
        }
        grad[2] -= 2.0 * self.exponent * dz
            * dx.powi(l[0] as i32)
            * dy.powi(l[1] as i32)
            * dz.powi(l[2] as i32);

        grad[0] *= self.coefficient * self.norm * exp;
        grad[1] *= self.coefficient * self.norm * exp;
        grad[2] *= self.coefficient * self.norm * exp;

        grad
    }

    /// Normalization constant for Cartesian Gaussian
    fn normalization(alpha: f64, ang: [usize; 3]) -> f64 {
        let l = ang[0] + ang[1] + ang[2];
        let pref = (2.0 * alpha / PI).powf(0.75);
        let ang_fac = (4.0 * alpha).powi(l as i32);
        pref * ang_fac.sqrt()
    }
}

impl Primitive {
    // --- getters seguros ---

    #[inline]
    pub fn exponent(&self) -> f64 {
        self.exponent
    }

    #[inline]
    pub fn coefficient(&self) -> f64 {
        self.coefficient
    }

    #[inline]
    pub fn center(&self) -> [f64; 3] {
        self.center
    }

    #[inline]
    pub fn ang(&self) -> [usize; 3] {
        self.ang
    }

    #[inline]
    pub fn norm(&self) -> f64 {
        self.norm
    }
}

