use crate::system::atom::Atom;

/// ∂E_nn / ∂R_A
pub fn grad_nuclear_repulsion(
    atoms: &[Atom],
) -> Vec<[f64; 3]> {
    let n = atoms.len();
    let mut grad = vec![[0.0; 3]; n];

    for a in 0..n {
        for b in 0..n {
            if a == b { continue; }

            let za = atoms[a].atomic_number as f64;
            let zb = atoms[b].atomic_number as f64;

            let ra = atoms[a].position;
            let rb = atoms[b].position;

            let dx = ra[0] - rb[0];
            let dy = ra[1] - rb[1];
            let dz = ra[2] - rb[2];

            let r2 = dx*dx + dy*dy + dz*dz;
            let r = r2.sqrt();
            let pref = za * zb / (r2 * r);

            grad[a][0] += pref * dx;
            grad[a][1] += pref * dy;
            grad[a][2] += pref * dz;
        }
    }
    grad
}
