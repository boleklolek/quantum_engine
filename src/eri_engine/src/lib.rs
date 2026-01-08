
use pyo3::prelude::*;

#[pyfunction]
fn eri_ssss(a: f64) -> f64 {
    a
}

#[pymodule]
fn eri_engine(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(eri_ssss, m)?)?;
    Ok(())
}
