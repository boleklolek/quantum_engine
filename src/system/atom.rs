//! Atomic data container
//!
//! This module must NOT contain:
//! - parsing logic
//! - periodic table knowledge
//! - hardcoded elements
//!
//! It is a pure data structure.

#[derive(Clone, Debug)]
pub struct Atom {
    pub symbol: String,
    pub atomic_number: usize,
    pub position: [f64; 3],
}

impl Atom {
    pub fn new(
        symbol: String,
        atomic_number: usize,
        position: [f64; 3],
    ) -> Self {
        Self {
            symbol,
            atomic_number,
            position,
        }
    }
}

