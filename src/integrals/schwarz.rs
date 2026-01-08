//! Schwarz inequality screening for ERIs
//!
//! |(μν|λσ)| ≤ sqrt( max|(μν|μν)| * max|(λσ|λσ)| )

use crate::basis::shell::Shell;
use crate::integrals::eri::eri_shell::eri_shell_shell;

/// Compute Schwarz bound between two shells
///
/// sqrt( max|(aa|aa)| * max|(bb|bb)| )
pub fn schwarz_shell_pair(
    shell_a: &Shell,
    shell_b: &Shell,
) -> f64 {

    // (aa|aa) block
    let eri_aa = eri_shell_shell(shell_a, shell_a);

    let mut max_a: f64 = 0.0;
    for row in &eri_aa {
        for &val in row {
            max_a = max_a.max(val.abs());
        }
    }

    // (bb|bb) block
    let eri_bb = eri_shell_shell(shell_b, shell_b);

    let mut max_b: f64 = 0.0;
    for row in &eri_bb {
        for &val in row {
            max_b = max_b.max(val.abs());
        }
    }

    (max_a * max_b).sqrt()
}

/// Full Schwarz matrix
pub fn compute_schwarz_bounds(
    shells: &[Shell],
) -> Vec<Vec<f64>> {

    let ns = shells.len();
    let mut bounds = vec![vec![0.0_f64; ns]; ns];

    for i in 0..ns {
        for j in 0..ns {
            bounds[i][j] = schwarz_shell_pair(&shells[i], &shells[j]);
        }
    }

    bounds
}

