//! Shell definition (contracted Gaussian shell)

use crate::basis::primitive::Primitive;

/// One contracted shell (s, p, d, ...)
#[derive(Clone, Debug)]
pub struct Shell {
    /// Primitives belonging to this shell
    pub primitives: Vec<Primitive>,

    /// Angular momentum (lx, ly, lz)
    pub ang: [usize; 3],

    /// Center of the shell
    pub center: [f64; 3],

    /// AO offset in global basis
    pub offset: usize,
}

impl Shell {
    /// Create a new shell
    pub fn new(
        primitives: Vec<Primitive>,
        ang: [usize; 3],
        center: [f64; 3],
        offset: usize,
    ) -> Self {
        Self {
            primitives,
            ang,
            center,
            offset,
        }
    }

    /// Number of Cartesian atomic orbitals in this shell
    pub fn n_orbitals(&self) -> usize {
        let l = self.ang[0] + self.ang[1] + self.ang[2];
        ((l + 1) * (l + 2)) / 2
    }

    /// Return list of Cartesian angular momentum combinations
    /// e.g. p-shell → [(1,0,0),(0,1,0),(0,0,1)]
    pub fn cartesian_components(&self) -> Vec<[usize; 3]> {
        let l = self.ang[0] + self.ang[1] + self.ang[2];
        let mut comps = Vec::new();

        for lx in 0..=l {
            for ly in 0..=(l - lx) {
                let lz = l - lx - ly;
                comps.push([lx, ly, lz]);
            }
        }

        comps
    }
}

