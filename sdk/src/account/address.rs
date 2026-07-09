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

use crate::{types::AddressNative, Field, Group};

use pyo3::prelude::*;
use snarkvm::prelude::ToField;

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    ops::Deref,
    str::FromStr,
};

/// The Aleo address type.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct Address(AddressNative);

impl Address {
    /// Initializes an address from a group element (for internal use).
    pub(crate) fn new(group: crate::types::GroupNative) -> Self {
        Self(AddressNative::new(group))
    }
}

#[pymethods]
impl Address {
    /// Reads in an account address string.
    #[staticmethod]
    fn from_string(s: &str) -> anyhow::Result<Self> {
        AddressNative::from_str(s).map(Self)
    }

    /// Returns the address as a group element.
    fn to_group(&self) -> Group {
        (*self.0.to_group()).into()
    }

    /// Returns the address as a field element (x-coordinate).
    fn to_field(&self) -> anyhow::Result<Field> {
        self.0.to_field().map(Into::into)
    }

    /// Returns the little-endian byte representation of the address.
    fn to_bytes_le(&self) -> anyhow::Result<Vec<u8>> {
        use snarkvm::prelude::ToBytes;
        self.0.to_bytes_le()
    }

    /// Parses an address from little-endian bytes.
    #[staticmethod]
    fn from_bytes_le(bytes: Vec<u8>) -> anyhow::Result<Self> {
        use snarkvm::prelude::FromBytes;
        AddressNative::from_bytes_le(&bytes).map(Self)
    }

    /// Recovers an address from a group element.
    #[staticmethod]
    fn from_group(group: &Group) -> anyhow::Result<Self> {
        // Address is built on top of Group, so just wrap the inner group element directly.
        Ok(Self(AddressNative::new(**group)))
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
    fn from(value: AddressNative) -> Self {
        Self(value)
    }
}
