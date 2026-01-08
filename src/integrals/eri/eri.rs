use crate::basis::primitive::Primitive;
use crate::basis::contracted::Contracted;
use crate::basis::shell::Shell;
use crate::system::atom::Atom;
use std::f64::consts::PI;

/// Boys function F_n(T) for small n (n=0 used here)
fn boys0(t: f64) -> f64 {
    if t < 1e-8 {
        1.0
    } else {
        0.5 * (PI / t).sqrt() * erf((t).sqrt())
    }
}

/// Error function (Abramowitz–Stegun approximation)
fn erf(x: f64) -> f64 {
    // constants
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0
        - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1)
            * t
            * (-x * x).exp());

    sign * y
}

/// Nuclear attraction integral between two primitive Gaussians and one nucleus
pub fn nuclear_attraction_primitive(
    a: &Primitive,
    center_a: [f64; 3],
    b: &Primitive,
    center_b: [f64; 3],
    nucleus_pos: [f64; 3],
    nuclear_charge: f64,
) -> f64 {
    let alpha = a.exponent();
    let beta = b.exponent();
    let p = alpha + beta;

    let px = (alpha * center_a[0] + beta * center_b[0]) / p;
    let py = (alpha * center_a[1] + beta * center_b[1]) / p;
    let pz = (alpha * center_a[2] + beta * center_b[2]) / p;

    let rpc2 = (px - nucleus_pos[0]).powi(2)
        + (py - nucleus_pos[1]).powi(2)
        + (pz - nucleus_pos[2]).powi(2);

    let t = p * rpc2;
    let boys = boys0(t);

    let rab2 = (center_a[0] - center_b[0]).powi(2)
        + (center_a[1] - center_b[1]).powi(2)
        + (center_a[2] - center_b[2]).powi(2);

    let prefactor = -2.0 * PI / p
        * nuclear_charge
        * (-alpha * beta / p * rab2).exp();

    a.coefficient() * b.coefficient() * a.norm() * b.norm() * prefactor * boys
}

/// Nuclear attraction between two contracted orbitals
pub fn nuclear_attraction_contracted(
    ao_a: &Contracted,
    center_a: [f64; 3],
    ao_b: &Contracted,
    center_b: [f64; 3],
    atoms: &[Atom],
) -> f64 {
    let mut v = 0.0;

    for atom in atoms {
        let z = atomic_number(&atom.symbol) as f64;
        ///let c = [atom.x, atom.y, atom.z];
        let c = atom.position;

        for pa in &ao_a.primitives {
            for pb in &ao_b.primitives {
                v += nuclear_attraction_primitive(
                    pa,
                    center_a,
                    pb,
                    center_b,
                    c,
                    z,
                );
            }
        }
    }

    v
}

/// Minimal atomic number lookup (reuse later from units.rs)
fn atomic_number(symbol: &str) -> i32 {
    match symbol {
        "H" => 1,
        "He" => 2,
        "C" => 6,
        "N" => 7,
        "O" => 8,
        "F" => 9,
        _ => panic!("Unknown element {}", symbol),
    }
}
