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

    pub fn num_orbitals(&self) -> usize {
        // Para un shell con momento angular `l`:
        // s (l=0): 1 orbital
        // p (l=1): 3 orbitales  
        // d (l=2): 6 orbitales (cartesianas) o 5 (esféricas)
        // f (l=3): 10 orbitales (cartesianas) o 7 (esféricas)
        
        match self.ang {
            0 => 1,   // s
            1 => 3,   // p
            2 => 6,   // d (cartesianas, cambia a 5 si usas esféricas)
            3 => 10,  // f (cartesianas, cambia a 7 si usas esféricas)
            l => ((l + 1) * (l + 2)) / 2, // fórmula general para cartesianas
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

