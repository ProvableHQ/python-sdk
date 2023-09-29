use pyo3::prelude::*;

mod aleo_account;
use aleo_account::PrivateKey;

#[pymodule]
#[pyo3(name = "aleo_python_sdk")]
fn account(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PrivateKey>()?;
    Ok(())
}
