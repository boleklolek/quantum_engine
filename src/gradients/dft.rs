use crate::basis::shell::Shell;
use crate::dft::vxc::DftEnergy;

/// ∂E_xc / ∂R_A
/// For now: numerical derivative via grid density dependence
pub fn grad_dft_xc(
    _shells: &[Shell],
    _density: &Vec<Vec<f64>>,
    _dft_energy: &DftEnergy,
    natoms: usize,
) -> Vec<[f64;3]> {
    // Placeholder: analytical XC gradients come next
    vec![[0.0;3]; natoms]
}
