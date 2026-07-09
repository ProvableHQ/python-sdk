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

use crate::{types::ScalarNative, Field};

use pyo3::{exceptions::PyZeroDivisionError, prelude::*};
use snarkvm::prelude::{One, ToField, Zero};

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    ops::Deref,
    str::FromStr,
};

/// The Aleo scalar type.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct Scalar(ScalarNative);

#[pymethods]
impl Scalar {
    /// Parses a scalar from a string.
    #[staticmethod]
    fn from_string(s: &str) -> anyhow::Result<Self> {
        ScalarNative::from_str(s).map(Self)
    }

    /// Returns the `0` element of the scalar.
    #[staticmethod]
    fn zero() -> Self {
        ScalarNative::zero().into()
    }

    /// Returns the `1` element of the scalar.
    #[staticmethod]
    fn one() -> Self {
        ScalarNative::one().into()
    }

    /// Returns the little-endian byte representation of the scalar element.
    fn to_bytes_le(&self) -> anyhow::Result<Vec<u8>> {
        use snarkvm::prelude::ToBytes;
        self.0.to_bytes_le()
    }

    /// Parses a scalar from little-endian bytes.
    #[staticmethod]
    fn from_bytes_le(bytes: Vec<u8>) -> anyhow::Result<Self> {
        use snarkvm::prelude::FromBytes;
        ScalarNative::from_bytes_le(&bytes).map(Self)
    }

    /// Returns this scalar as a field element.
    fn to_field(&self) -> anyhow::Result<Field> {
        self.0.to_field().map(Into::into)
    }

    /// Returns the sum of self and other.
    fn add(&self, other: Self) -> Self {
        Self(self.0 + other.0)
    }

    /// Returns the difference of self and other.
    fn subtract(&self, other: Self) -> Self {
        Self(self.0 - other.0)
    }

    /// Returns the product of self and other.
    fn multiply(&self, other: Self) -> Self {
        Self(self.0 * other.0)
    }

    /// Returns self divided by other (panics on zero divisor).
    fn divide(&self, other: Self) -> PyResult<Self> {
        if other.is_zero() {
            Err(PyZeroDivisionError::new_err("division by zero"))
        } else {
            Ok(Self(self.0 / other.0))
        }
    }

    /// Returns the Scalar as a string.
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

    fn __add__(&self, other: Self) -> Self {
        self.add(other)
    }

    fn __sub__(&self, other: Self) -> Self {
        self.subtract(other)
    }

    fn __mul__(&self, other: Self) -> Self {
        self.multiply(other)
    }

    fn __truediv__(&self, other: Self) -> PyResult<Self> {
        self.divide(other)
    }
}

impl Deref for Scalar {
    type Target = ScalarNative;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<ScalarNative> for Scalar {
    fn from(value: ScalarNative) -> Self {
        Self(value)
    }
}

impl From<Scalar> for ScalarNative {
    fn from(value: Scalar) -> Self {
        value.0
    }
}
