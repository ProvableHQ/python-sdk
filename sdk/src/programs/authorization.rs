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

use crate::types::AuthorizationNative;

use pyo3::prelude::*;
use snarkvm::prelude::{FromBytes, ToBytes};

#[pyclass(frozen)]
#[derive(Clone)]
pub struct Authorization(AuthorizationNative);

#[pymethods]
impl Authorization {
    /// Reads in an Authorization from a JSON string.
    #[staticmethod]
    fn from_json(json: &str) -> anyhow::Result<Self> {
        Ok(Self(serde_json::from_str(json)?))
    }

    /// Serialize the given Authorization as a JSON string.
    fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string(&self.0)?)
    }

    /// Constructs an Authorization from a byte array.
    #[staticmethod]
    fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        AuthorizationNative::from_bytes_le(bytes).map(Self)
    }

    /// Returns the byte representation of an Authorization.
    fn bytes(&self) -> anyhow::Result<Vec<u8>> {
        self.0.to_bytes_le()
    }

    /// Returns the Authorization as a JSON string.
    fn __str__(&self) -> anyhow::Result<String> {
        self.to_json()
    }
}

impl From<AuthorizationNative> for Authorization {
    fn from(value: AuthorizationNative) -> Self {
        Self(value)
    }
}

impl From<Authorization> for AuthorizationNative {
    fn from(value: Authorization) -> Self {
        value.0
    }
}
