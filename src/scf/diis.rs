/// Direct Inversion in the Iterative Subspace (DIIS)
///
/// Stores (Fock, error) pairs and extrapolates a new Fock matrix.

pub struct Diis {
    max_vecs: usize,
    focks: Vec<Vec<Vec<f64>>>,
    errors: Vec<Vec<f64>>, // flattened error matrices
}

impl Diis {
    pub fn new(max_vecs: usize) -> Self {
        Self {
            max_vecs,
            focks: Vec::new(),
            errors: Vec::new(),
        }
    }

    /// Push a new (Fock, error) pair
    pub fn push(&mut self, fock: Vec<Vec<f64>>, error: Vec<Vec<f64>>) {
        let e_flat = flatten(&error);

        self.focks.push(fock);
        self.errors.push(e_flat);

        if self.focks.len() > self.max_vecs {
            self.focks.remove(0);
            self.errors.remove(0);
        }
    }

    /// Extrapolate a new Fock matrix using DIIS
    ///
    /// Returns None if not enough vectors
    pub fn extrapolate(&self) -> Option<Vec<Vec<f64>>> {
        let m = self.errors.len();
        if m < 2 {
            return None;
        }

        // Build B matrix (size m+1)
        let mut b = vec![vec![0.0; m + 1]; m + 1];

        for i in 0..m {
            for j in 0..m {
                b[i][j] = dot(&self.errors[i], &self.errors[j]);
            }
            b[i][m] = -1.0;
            b[m][i] = -1.0;
        }
        b[m][m] = 0.0;

        // RHS
        let mut rhs = vec![0.0; m + 1];
        rhs[m] = -1.0;

        // Solve linear system
        let coeffs = solve_linear(&b, &rhs)?;

        // Combine Fock matrices
        let n = self.focks[0].len();
        let mut f_new = vec![vec![0.0; n]; n];

        for i in 0..m {
            let c = coeffs[i];
            for p in 0..n {
                for q in 0..n {
                    f_new[p][q] += c * self.focks[i][p][q];
                }
            }
        }

        Some(f_new)
    }
}

// ---------- helpers ----------

fn flatten(m: &Vec<Vec<f64>>) -> Vec<f64> {
    m.iter().flat_map(|row| row.iter()).cloned().collect()
}

fn dot(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Very small Gaussian elimination solver (for DIIS only)
fn solve_linear(a: &Vec<Vec<f64>>, b: &Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    let mut a = a.clone();
    let mut b = b.clone();

    for i in 0..n {
        // Pivot
        let mut max = i;
        for k in (i + 1)..n {
            if a[k][i].abs() > a[max][i].abs() {
                max = k;
            }
        }
        if a[max][i].abs() < 1e-12 {
            return None;
        }
        a.swap(i, max);
        b.swap(i, max);

        // Normalize
        let diag = a[i][i];
        for j in i..n {
            a[i][j] /= diag;
        }
        b[i] /= diag;

        // Eliminate
        for k in 0..n {
            if k != i {
                let factor = a[k][i];
                for j in i..n {
                    a[k][j] -= factor * a[i][j];
                }
                b[k] -= factor * b[i];
            }
        }
    }

    Some(b)
}
