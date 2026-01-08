//! Raman intensities from polarizability derivatives

/// Raman activity (Placzek approximation)
pub fn raman_intensities(
    dalpha_dq: &Vec<[[f64;3];3]>, // per mode
) -> Vec<f64> {

    dalpha_dq.iter().map(|a| {
        let iso = (a[0][0] + a[1][1] + a[2][2]) / 3.0;

        let ani =
            ( (a[0][0]-a[1][1]).powi(2)
            + (a[1][1]-a[2][2]).powi(2)
            + (a[2][2]-a[0][0]).powi(2)
            + 6.0*(a[0][1]*a[0][1]
                 + a[1][2]*a[1][2]
                 + a[0][2]*a[0][2]) ) / 2.0;

        45.0*iso*iso + 7.0*ani
    }).collect()
}
