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

use crate::types::VerifyingKeyNative;

use pyo3::prelude::*;
use snarkvm::prelude::{FromBytes, ToBytes};

use std::str::FromStr;

#[pyclass(frozen)]
#[derive(Clone)]
pub struct VerifyingKey(VerifyingKeyNative);

#[pymethods]
impl VerifyingKey {
    /// Parses a veryifying key from string.
    #[staticmethod]
    fn from_string(s: &str) -> anyhow::Result<Self> {
        VerifyingKeyNative::from_str(s).map(Self)
    }

    /// Constructs a proving key from a byte array
    #[staticmethod]
    fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        VerifyingKeyNative::from_bytes_le(bytes).map(Self)
    }

    /// Returns the byte representation of a veryfying key
    fn bytes(&self) -> anyhow::Result<Vec<u8>> {
        self.0.to_bytes_le()
    }

    /// Returns the verifying key as a string.
    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __eq__(&self, other: &Self) -> bool {
        *self.0 == *other.0
    }

    #[classattr]
    const __hash__: Option<PyObject> = None;
}

impl From<VerifyingKeyNative> for VerifyingKey {
    fn from(value: VerifyingKeyNative) -> Self {
        Self(value)
    }
}

impl From<VerifyingKey> for VerifyingKeyNative {
    fn from(value: VerifyingKey) -> Self {
        value.0
    }
}
