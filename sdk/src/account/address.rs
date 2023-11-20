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

use crate::types::AddressNative;

use pyo3::prelude::*;

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    ops::Deref,
    str::FromStr,
};

#[pyclass(frozen)]
#[derive(Clone)]
pub struct Address(AddressNative);

#[pymethods]
impl Address {
    /// Reads in an account address string.
    #[staticmethod]
    fn from_string(s: &str) -> anyhow::Result<Self> {
        let address = FromStr::from_str(s)?;
        Ok(Self(address))
    }

    /// Returns the address as a base58 string.
    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.0.hash(&mut hasher);
        hasher.finish()
    }
}

impl Deref for Address {
    type Target = AddressNative;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<AddressNative> for Address {
    fn from(address: AddressNative) -> Self {
        Self(address)
    }
}
