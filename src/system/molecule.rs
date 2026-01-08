use crate::system::atom::Atom;
use crate::system::parser_xyz::read_xyz;

pub struct Molecule {
    pub atoms: Vec<Atom>,
    pub charge: i32,
    pub multiplicity: usize,
}

impl Molecule {
    pub fn from_xyz(
        path: &str,
        charge: i32,
        multiplicity: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            atoms: read_xyz(path)?,
            charge,
            multiplicity,
        })
    }
}

