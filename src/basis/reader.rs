//! Basis set reader (internal representation)

use std::collections::HashMap;

/// One shell in a basis set
#[derive(Clone, Debug)]
pub struct BasisShell {
    pub angular_momentum: [usize; 3],
    pub primitives: Vec<(f64, f64)>, // (exponent, coefficient)
}

/// Basis for one atom
#[derive(Clone, Debug)]
pub struct BasisSet {
    pub shells: Vec<BasisShell>,
}

/// Read basis set for a given element
///
/// In a real engine this would parse:
/// - .gbs
/// - EMSL JSON
/// - BSE YAML
///
/// For now, this is a stub you can expand later.
pub fn read_basis_set(
    basis_name: &str,
    element: &str,
) -> Option<BasisSet> {

    match (basis_name, element) {
        ("sto-3g", "H") => Some(sto3g_hydrogen()),
        _ => None,
    }
}

fn sto3g_hydrogen() -> BasisSet {
    BasisSet {
        shells: vec![
            BasisShell {
                angular_momentum: [0, 0, 0],
                primitives: vec![
                    (3.42525091, 0.15432897),
                    (0.62391373, 0.53532814),
                    (0.16885540, 0.44463454),
                ],
            },
        ],
    }
}

