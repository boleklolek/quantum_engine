//! Molecular orbital space utilities

#[derive(Clone)]
pub struct MoSpace {
    pub n_occ: usize,
    pub n_vir: usize,
    pub n_mo: usize,
}

impl MoSpace {
    pub fn new(n_mo: usize, n_occ: usize) -> Self {
        assert!(n_occ < n_mo);
        Self {
            n_occ,
            n_vir: n_mo - n_occ,
            n_mo,
        }
    }

    #[inline]
    pub fn occ(&self) -> std::ops::Range<usize> {
        0..self.n_occ
    }

    #[inline]
    pub fn vir(&self) -> std::ops::Range<usize> {
        self.n_occ..self.n_mo
    }
}

