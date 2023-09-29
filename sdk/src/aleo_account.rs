use snarkvm::console::account::private_key::PrivateKey as AleoPrivateKey;
use snarkvm::console::network::Testnet3 as CurrentNetwork;

use rand::{rngs::StdRng, SeedableRng};
use pyo3::prelude::*;

#[pyclass]
pub struct PrivateKey {
    private_key: AleoPrivateKey<CurrentNetwork>
}

#[pymethods]
impl PrivateKey {
    #[new]
    pub fn new() -> Self {
        PrivateKey { private_key: AleoPrivateKey::<CurrentNetwork>::new(&mut StdRng::from_entropy()).unwrap() }
    }

    /// Get the seed of the private key
    pub fn seed(&self) -> PyResult<String> {
        Ok(self.private_key.seed().to_string())
    }

    /// Get the sk_sig corresponding to the private key
    pub fn sk_sig(&self) -> PyResult<String> {
        Ok(self.private_key.sk_sig().to_string())
    }

    /// Get the r_sig corresponding to the private key
    pub fn r_sig(&self) -> PyResult<String> {
        Ok(self.private_key.r_sig().to_string())
    }

    /// Get the private key as a string
    pub fn to_string(&self) -> PyResult<String> {
        Ok(self.private_key.to_string())
    }
}
