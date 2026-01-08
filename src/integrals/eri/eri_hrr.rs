//! Horizontal Recurrence Relations (HRR)
//!
//! Move angular momentum between centers A↔B and C↔D.
//! These are scalar combinators: they do NOT compute ERIs themselves,
//! only combine already-computed integrals (typically from VRR).

/// HRR A → B
/// (a+1,b|c,d) = (A_i - B_i)*(a,b|c,d) + (a,b+1|c,d)
#[inline]
pub fn hrr_ab(
    value_ab: f64,     // (a,b|c,d)
    value_abp1: f64,   // (a,b+1|c,d)
    ra_i: f64,
    rb_i: f64,
) -> f64 {
    (ra_i - rb_i) * value_ab + value_abp1
}

/// HRR B → A
/// (a,b+1|c,d) = (B_i - A_i)*(a,b|c,d) + (a+1,b|c,d)
#[inline]
pub fn hrr_ba(
    value_ab: f64,
    value_ap1b: f64,
    rb_i: f64,
    ra_i: f64,
) -> f64 {
    (rb_i - ra_i) * value_ab + value_ap1b
}

/// HRR C → D
/// (a,b|c+1,d) = (C_i - D_i)*(a,b|c,d) + (a,b|c,d+1)
#[inline]
pub fn hrr_cd(
    value_cd: f64,     // (a,b|c,d)
    value_cdp1: f64,   // (a,b|c,d+1)
    rc_i: f64,
    rd_i: f64,
) -> f64 {
    (rc_i - rd_i) * value_cd + value_cdp1
}

/// HRR D → C
/// (a,b|c,d+1) = (D_i - C_i)*(a,b|c,d) + (a,b|c+1,d)
#[inline]
pub fn hrr_dc(
    value_cd: f64,
    value_cp1d: f64,
    rd_i: f64,
    rc_i: f64,
) -> f64 {
    (rd_i - rc_i) * value_cd + value_cp1d
}
