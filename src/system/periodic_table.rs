//! Periodic table utilities

use std::collections::HashMap;

pub fn atomic_number(symbol: &str) -> Option<usize> {
    match symbol {
        "H" => Some(1),
        "He" => Some(2),
        "Li" => Some(3),
        "Be" => Some(4),
        "B" => Some(5),
        "C" => Some(6),
        "N" => Some(7),
        "O" => Some(8),
        "F" => Some(9),
        "Ne" => Some(10),
        // se puede extender sin tocar Atom
        _ => None,
    }
}

