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

use crate::{types::GroupNative, Address, Field, Scalar};

use pyo3::prelude::*;
use snarkvm::prelude::{Double, FromField, Zero};

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    ops::Deref,
    str::FromStr,
};

/// The Aleo group type.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct Group(GroupNative);

#[pymethods]
impl Group {
    /// Parses a group from a string.
    #[staticmethod]
    fn from_string(s: &str) -> anyhow::Result<Self> {
        GroupNative::from_str(s).map(Self)
    }

    /// Returns the `0` element of the group.
    #[staticmethod]
    fn zero() -> Self {
        GroupNative::zero().into()
    }

    /// Returns the prime subgroup generator.
    #[staticmethod]
    fn generator() -> Self {
        GroupNative::generator().into()
    }

    /// Returns the group from a field element (x-coordinate recovery).
    #[staticmethod]
    fn from_field(field: &Field) -> anyhow::Result<Self> {
        GroupNative::from_field(&**field).map(Self)
    }

    /// Returns the x-coordinate of the group element as a field.
    fn to_x_coordinate(&self) -> Field {
        self.0.to_x_coordinate().into()
    }

    /// Returns the address corresponding to this group element.
    fn to_address(&self) -> Address {
        Address::new(self.0)
    }

    /// Returns the double of this group element.
    fn double(&self) -> Self {
        Self(Double::double(&self.0))
    }

    /// Returns the sum of self and other.
    fn add(&self, other: &Self) -> Self {
        Self(self.0 + other.0)
    }

    /// Returns the difference of self and other.
    fn subtract(&self, other: &Self) -> Self {
        Self(self.0 - other.0)
    }

    /// Returns the negation of self.
    fn negate(&self) -> Self {
        Self(-self.0)
    }

    /// Returns self scaled by scalar (scalar multiplication).
    fn scalar_multiply(&self, scalar: &Scalar) -> Self {
        Self(self.0 * **scalar)
    }

    /// Returns the little-endian byte representation of the group element (x-coordinate).
    fn to_bytes_le(&self) -> anyhow::Result<Vec<u8>> {
        use snarkvm::prelude::ToBytes;
        self.0.to_bytes_le()
    }

    /// Parses a group from little-endian bytes (x-coordinate).
    #[staticmethod]
    fn from_bytes_le(bytes: Vec<u8>) -> anyhow::Result<Self> {
        use snarkvm::prelude::FromBytes;
        GroupNative::from_bytes_le(&bytes).map(Self)
    }

    // ---------- dunders ----------

    /// Returns the Group as a string.
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

    fn __add__(&self, other: &Self) -> Self {
        self.add(other)
    }

    fn __sub__(&self, other: &Self) -> Self {
        self.subtract(other)
    }

    fn __neg__(&self) -> Self {
        self.negate()
    }

    fn __mul__(&self, scalar: &Scalar) -> Self {
        self.scalar_multiply(scalar)
    }
}

impl Deref for Group {
    type Target = GroupNative;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<GroupNative> for Group {
    fn from(value: GroupNative) -> Self {
        Self(value)
    }
}

impl From<Group> for GroupNative {
    fn from(value: Group) -> Self {
        value.0
    }
}

impl From<&GroupNative> for Group {
    fn from(value: &GroupNative) -> Self {
        Self(*value)
    }
}
