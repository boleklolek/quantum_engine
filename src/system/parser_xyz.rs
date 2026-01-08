use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::system::atom::Atom;
use crate::system::units::angstrom_to_bohr;
use crate::system::periodic_table::atomic_number;

pub fn read_xyz(path: &str) -> Result<Vec<Atom>, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let natoms: usize = lines
        .next()
        .ok_or("XYZ empty")?
        .map_err(|e| e.to_string())?
        .trim()
        .parse()
        .map_err(|_| "Invalid atom count")?;

    lines.next(); // comment

    let mut atoms = Vec::with_capacity(natoms);

    for _ in 0..natoms {
        let line = lines
            .next()
            .ok_or("Unexpected EOF")?
            .map_err(|e| e.to_string())?;

        let parts: Vec<_> = line.split_whitespace().collect();
        if parts.len() != 4 {
            return Err("Invalid XYZ line".into());
        }

        let symbol = parts[0];
        let z = atomic_number(symbol)
            .ok_or(format!("Unknown element {}", symbol))?;

        let x: f64 = parts[1].parse().map_err(|_| "Bad X")?;
        let y: f64 = parts[2].parse().map_err(|_| "Bad Y")?;
        let zc: f64 = parts[3].parse().map_err(|_| "Bad Z")?;

        atoms.push(Atom::new(
            symbol.to_string(),
            z,
            [
                angstrom_to_bohr(x),
                angstrom_to_bohr(y),
                angstrom_to_bohr(zc),
            ],
        ));
    }

    Ok(atoms)
}

