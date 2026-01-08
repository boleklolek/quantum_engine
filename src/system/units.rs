/// Physical constants and unit conversions
///
/// All internal quantities in the engine are assumed to be in
/// atomic units (Bohr, Hartree).

/// Bohr radius in Angstrom
pub const BOHR_TO_ANGSTROM: f64 = 0.529177210903;
pub const ANGSTROM_TO_BOHR: f64 = 1.0 / BOHR_TO_ANGSTROM;

/// Hartree energy in eV (for reporting only)
pub const HARTREE_TO_EV: f64 = 27.211386245988;

/// Convert Angstrom to Bohr
#[inline]
pub fn angstrom_to_bohr(x: f64) -> f64 {
    x * ANGSTROM_TO_BOHR
}

/// Convert Bohr to Angstrom
#[inline]
pub fn bohr_to_angstrom(x: f64) -> f64 {
    x * BOHR_TO_ANGSTROM
}

/// Basic element data
#[derive(Debug, Clone, Copy)]
pub struct Element {
    pub symbol: &'static str,
    pub atomic_number: i32,
    pub atomic_mass: f64, // in amu
}

/// Periodic table (H → Ar)
pub const PERIODIC_TABLE: &[Element] = &[
    Element { symbol: "H",  atomic_number: 1,  atomic_mass: 1.008 },
    Element { symbol: "He", atomic_number: 2,  atomic_mass: 4.0026 },
    Element { symbol: "Li", atomic_number: 3,  atomic_mass: 6.94 },
    Element { symbol: "Be", atomic_number: 4,  atomic_mass: 9.0122 },
    Element { symbol: "B",  atomic_number: 5,  atomic_mass: 10.81 },
    Element { symbol: "C",  atomic_number: 6,  atomic_mass: 12.011 },
    Element { symbol: "N",  atomic_number: 7,  atomic_mass: 14.007 },
    Element { symbol: "O",  atomic_number: 8,  atomic_mass: 15.999 },
    Element { symbol: "F",  atomic_number: 9,  atomic_mass: 18.998 },
    Element { symbol: "Ne", atomic_number: 10, atomic_mass: 20.180 },
    Element { symbol: "Na", atomic_number: 11, atomic_mass: 22.990 },
    Element { symbol: "Mg", atomic_number: 12, atomic_mass: 24.305 },
    Element { symbol: "Al", atomic_number: 13, atomic_mass: 26.982 },
    Element { symbol: "Si", atomic_number: 14, atomic_mass: 28.085 },
    Element { symbol: "P",  atomic_number: 15, atomic_mass: 30.974 },
    Element { symbol: "S",  atomic_number: 16, atomic_mass: 32.06 },
    Element { symbol: "Cl", atomic_number: 17, atomic_mass: 35.45 },
    Element { symbol: "Ar", atomic_number: 18, atomic_mass: 39.948 },
];

/// Lookup atomic number from symbol
pub fn atomic_number(symbol: &str) -> i32 {
    PERIODIC_TABLE
        .iter()
        .find(|e| e.symbol == symbol)
        .map(|e| e.atomic_number)
        .unwrap_or_else(|| panic!("Unknown element symbol: {}", symbol))
}

/// Lookup atomic mass from symbol
pub fn atomic_mass(symbol: &str) -> f64 {
    PERIODIC_TABLE
        .iter()
        .find(|e| e.symbol == symbol)
        .map(|e| e.atomic_mass)
        .unwrap_or_else(|| panic!("Unknown element symbol: {}", symbol))
}
