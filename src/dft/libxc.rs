//! Rust wrapper for libxc
//!
//! This module provides:
//! - LDA / GGA / meta-GGA energies
//! - Potentials (first derivatives)
//! - Hessians (second derivatives)
//!
//! libxc is treated as a pure numerical backend.
//! No SCF / geometry logic belongs here.

use libc::{c_double, c_int};
use std::ptr;

// ==================================================
// FFI bindings (minimal, explicit)
// ==================================================

#[repr(C)]
struct xc_func_type {
    _private: [u8; 0],
}

extern "C" {
    fn xc_func_init(
        p: *mut *mut xc_func_type,
        func_id: c_int,
        spin: c_int,
    ) -> c_int;

    fn xc_func_end(p: *mut xc_func_type);

    fn xc_lda_exc(
        p: *const xc_func_type,
        n: c_int,
        rho: *const c_double,
        exc: *mut c_double,
    );

    fn xc_lda_vxc(
        p: *const xc_func_type,
        n: c_int,
        rho: *const c_double,
        vxc: *mut c_double,
    );

    fn xc_lda_fxc(
        p: *const xc_func_type,
        n: c_int,
        rho: *const c_double,
        fxc: *mut c_double,
    );

    fn xc_gga_exc_vxc(
        p: *const xc_func_type,
        n: c_int,
        rho: *const c_double,
        sigma: *const c_double,
        exc: *mut c_double,
        vrho: *mut c_double,
        vsigma: *mut c_double,
    );

    fn xc_gga_fxc(
        p: *const xc_func_type,
        n: c_int,
        rho: *const c_double,
        sigma: *const c_double,
        frr: *mut c_double,
        frs: *mut c_double,
        fss: *mut c_double,
    );

    fn xc_mgga_exc_vxc(
        p: *const xc_func_type,
        n: c_int,
        rho: *const c_double,
        sigma: *const c_double,
        tau: *const c_double,
        exc: *mut c_double,
        vrho: *mut c_double,
        vsigma: *mut c_double,
        vtau: *mut c_double,
    );

    fn xc_mgga_fxc(
        p: *const xc_func_type,
        n: c_int,
        rho: *const c_double,
        sigma: *const c_double,
        tau: *const c_double,
        frr: *mut c_double,
        frs: *mut c_double,
        fst: *mut c_double,
        fss: *mut c_double,
        ftt: *mut c_double,
    );
}

// ==================================================
// Rust-side containers
// ==================================================

pub struct LibXC {
    func: *mut xc_func_type,
    spin: bool,
}

pub struct XcHessian {
    pub vrr: Vec<f64>, // d²f/dρ²
    pub vrs: Vec<f64>, // d²f/dρdσ
    pub vss: Vec<f64>, // d²f/dσ²
    pub vtt: Vec<f64>, // d²f/dτ² (meta-GGA)
}

// ==================================================
// Constructor / destructor
// ==================================================

impl LibXC {
    pub fn new(func_id: i32, spin: bool) -> Self {
        let mut ptr: *mut xc_func_type = ptr::null_mut();
        let spin_flag = if spin { 1 } else { 0 };

        unsafe {
            let ret = xc_func_init(&mut ptr, func_id, spin_flag);
            if ret != 0 {
                panic!("libxc init failed for functional {}", func_id);
            }
        }

        LibXC { func: ptr, spin }
    }
}

impl Drop for LibXC {
    fn drop(&mut self) {
        unsafe {
            xc_func_end(self.func);
        }
    }
}

// ==================================================
// LDA
// ==================================================

impl LibXC {
    pub fn eval_lda(&self, rho: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = rho.len() as c_int;
        let mut exc = vec![0.0; rho.len()];
        let mut vxc = vec![0.0; rho.len()];

        unsafe {
            xc_lda_exc(self.func, n, rho.as_ptr(), exc.as_mut_ptr());
            xc_lda_vxc(self.func, n, rho.as_ptr(), vxc.as_mut_ptr());
        }
        (exc, vxc)
    }
}

// ==================================================
// GGA
// ==================================================

impl LibXC {
    pub fn eval_gga(
        &self,
        rho: &[f64],
        sigma: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {

        let n = rho.len() as c_int;
        let mut exc = vec![0.0; rho.len()];
        let mut vrho = vec![0.0; rho.len()];
        let mut vsigma = vec![0.0; sigma.len()];

        unsafe {
            xc_gga_exc_vxc(
                self.func,
                n,
                rho.as_ptr(),
                sigma.as_ptr(),
                exc.as_mut_ptr(),
                vrho.as_mut_ptr(),
                vsigma.as_mut_ptr(),
            );
        }
        (exc, vrho, vsigma)
    }

    pub fn eval_gga_hessian(
        &self,
        rho: &[f64],
        sigma: &[f64],
    ) -> XcHessian {

        let n = rho.len() as c_int;

        let mut frr = vec![0.0; rho.len()];
        let mut frs = vec![0.0; sigma.len()];
        let mut fss = vec![0.0; sigma.len()];

        unsafe {
            xc_gga_fxc(
                self.func,
                n,
                rho.as_ptr(),
                sigma.as_ptr(),
                frr.as_mut_ptr(),
                frs.as_mut_ptr(),
                fss.as_mut_ptr(),
            );
        }

        XcHessian {
            vrr: frr,
            vrs: frs,
            vss: fss,
            vtt: vec![],
        }
    }
}

// ==================================================
// meta-GGA
// ==================================================

impl LibXC {
    pub fn eval_mgga(
        &self,
        rho: &[f64],
        sigma: &[f64],
        tau: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {

        let n = rho.len() as c_int;

        let tau_vec = vec![tau; rho.len()];
        let mut exc = vec![0.0; rho.len()];
        let mut vrho = vec![0.0; rho.len()];
        let mut vsigma = vec![0.0; sigma.len()];
        let mut vtau = vec![0.0; rho.len()];

        unsafe {
            xc_mgga_exc_vxc(
                self.func,
                n,
                rho.as_ptr(),
                sigma.as_ptr(),
                tau_vec.as_ptr(),
                exc.as_mut_ptr(),
                vrho.as_mut_ptr(),
                vsigma.as_mut_ptr(),
                vtau.as_mut_ptr(),
            );
        }
        (exc, vrho, vsigma, vtau)
    }

    pub fn eval_mgga_hessian(
        &self,
        rho: &[f64],
        sigma: &[f64],
        tau: f64,
    ) -> XcHessian {

        let n = rho.len() as c_int;
        let tau_vec = vec![tau; rho.len()];

        let mut frr = vec![0.0; rho.len()];
        let mut frs = vec![0.0; sigma.len()];
        let mut fst = vec![0.0; sigma.len()];
        let mut fss = vec![0.0; sigma.len()];
        let mut ftt = vec![0.0; rho.len()];

        unsafe {
            xc_mgga_fxc(
                self.func,
                n,
                rho.as_ptr(),
                sigma.as_ptr(),
                tau_vec.as_ptr(),
                frr.as_mut_ptr(),
                frs.as_mut_ptr(),
                fst.as_mut_ptr(),
                fss.as_mut_ptr(),
                ftt.as_mut_ptr(),
            );
        }

        XcHessian {
            vrr: frr,
            vrs: frs,
            vss: fss,
            vtt: ftt,
        }
    }
}

