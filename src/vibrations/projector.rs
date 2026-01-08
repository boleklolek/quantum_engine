//! Projection of translational and rotational modes

use nalgebra::{DMatrix, DVector};

pub fn project_tr_rotation(
    hessian: &Vec<Vec<f64>>,
    coords: &Vec<f64>,
    masses: &Vec<f64>,
) -> Vec<Vec<f64>> {

    let n = coords.len();
    let natoms = n / 3;

    let mut proj = DMatrix::<f64>::identity(n, n);

    // Translation modes
    for axis in 0..3 {
        let mut v = DVector::<f64>::zeros(n);
        for a in 0..natoms {
            v[3*a + axis] = 1.0;
        }
        normalize_mass_weighted(&mut v, masses);
        proj -= &v * v.transpose();
    }

    // Rotation modes (around x,y,z)
    let com = center_of_mass(coords, masses);
    let rot_axes = [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]];

    for axis in rot_axes {
        let mut v = DVector::<f64>::zeros(n);
        for a in 0..natoms {
            let rx = coords[3*a]   - com[0];
            let ry = coords[3*a+1] - com[1];
            let rz = coords[3*a+2] - com[2];

            let cross = [
                axis[1]*rz - axis[2]*ry,
                axis[2]*rx - axis[0]*rz,
                axis[0]*ry - axis[1]*rx,
            ];

            for k in 0..3 {
                v[3*a + k] = cross[k];
            }
        }
        normalize_mass_weighted(&mut v, masses);
        proj -= &v * v.transpose();
    }

    let h = DMatrix::from_vec(n, n, hessian.iter().flatten().cloned().collect());
    let h_proj = &proj * h * &proj;

    h_proj.as_slice()
        .chunks(n)
        .map(|r| r.to_vec())
        .collect()
}

fn normalize_mass_weighted(v: &mut DVector<f64>, masses: &Vec<f64>) {
    let mut norm = 0.0;
    for i in 0..v.len()/3 {
        let m = masses[i];
        for k in 0..3 {
            norm += v[3*i + k] * v[3*i + k] * m;
        }
    }
    norm = norm.sqrt();
    *v /= norm;
}

fn center_of_mass(coords: &Vec<f64>, masses: &Vec<f64>) -> [f64;3] {
    let natoms = masses.len();
    let mut com = [0.0;3];
    let mut tot = 0.0;

    for a in 0..natoms {
        let m = masses[a];
        tot += m;
        for k in 0..3 {
            com[k] += m * coords[3*a + k];
        }
    }

    for k in 0..3 {
        com[k] /= tot;
    }
    com
}
