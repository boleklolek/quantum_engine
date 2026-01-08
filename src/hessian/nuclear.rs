use crate::system::atom::Atom;

pub fn hess_nuclear_repulsion(atoms: &[Atom]) -> Vec<Vec<f64>> {
    let n = atoms.len();
    let dim = 3*n;
    let mut h = vec![vec![0.0; dim]; dim];

    for a in 0..n {
        for b in 0..n {
            if a == b { continue; }

            let za = atoms[a].atomic_number as f64;
            let zb = atoms[b].atomic_number as f64;

            let ra = atoms[a].position;
            let rb = atoms[b].position;

            let dx = ra[0]-rb[0];
            let dy = ra[1]-rb[1];
            let dz = ra[2]-rb[2];

            let r2 = dx*dx + dy*dy + dz*dz;
            let r = r2.sqrt();
            let pref = za*zb/(r2*r);

            for i in 0..3 {
                for j in 0..3 {
                    let val = pref * match (i,j) {
                        (0,0) => 3.0*dx*dx/r2 - 1.0,
                        (1,1) => 3.0*dy*dy/r2 - 1.0,
                        (2,2) => 3.0*dz*dz/r2 - 1.0,
                        (0,1)|(1,0) => 3.0*dx*dy/r2,
                        (0,2)|(2,0) => 3.0*dx*dz/r2,
                        (1,2)|(2,1) => 3.0*dy*dz/r2,
                        _ => 0.0,
                    };

                    let ia = 3*a+i;
                    let ib = 3*b+j;

                    h[ia][ib] += val;
                }
            }
        }
    }

    h
}

