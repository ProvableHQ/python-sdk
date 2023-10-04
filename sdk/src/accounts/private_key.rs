// Copyright (C) 2019-2023 Aleo Systems Inc.
// This file is part of the Aleo SDK library.

// The Aleo SDK library is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// The Aleo SDK library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with the Aleo SDK library. If not, see <https://www.gnu.org/licenses/>.

use super::*;

use rand::{rngs::StdRng, SeedableRng};

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

