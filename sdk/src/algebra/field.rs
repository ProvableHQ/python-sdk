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

use crate::types::FieldNative;

use pyo3::prelude::*;
use snarkvm::prelude::Zero;

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    ops::Deref,
    str::FromStr,
};

/// The Aleo field type.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct Field(FieldNative);

#[pymethods]
impl Field {
    /// Parses a field from a string.
    #[staticmethod]
    fn from_string(s: &str) -> anyhow::Result<Self> {
        FieldNative::from_str(s).map(Self)
    }

    /// Initializes a new field from a `u128`.
    #[staticmethod]
    fn from_u128(value: u128) -> Self {
        FieldNative::from_u128(value).into()
    }

    /// Returns the `0` element of the field.
    #[staticmethod]
    fn zero() -> Self {
        FieldNative::zero().into()
    }

    /// Returns the Field as a string.
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

impl Deref for Field {
    type Target = FieldNative;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<FieldNative> for Field {
    fn from(value: FieldNative) -> Self {
        Self(value)
    }
}

impl From<Field> for FieldNative {
    fn from(value: Field) -> Self {
        value.0
    }
}
