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

use crate::{
    types::{
        GroupNative, I128Native, I16Native, I32Native, I64Native, I8Native, LiteralNative,
        PlaintextNative, ScalarNative, U128Native, U16Native, U32Native, U64Native, U8Native,
    },
    Address, Boolean, Field, Scalar,
};

use pyo3::prelude::*;
use rand::rngs::StdRng;
use snarkvm::prelude::{
    Double, FromBits, FromBytes, FromField, ToBits, ToBytes, ToFields, Uniform, Zero,
};

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    ops::Deref,
    str::FromStr,
    sync::OnceLock,
};

/// The Aleo group type.
#[pyclass(frozen, from_py_object)]
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

    /// Returns a random group element.
    #[staticmethod]
    fn random() -> Self {
        <GroupNative as Uniform>::rand(&mut rand::make_rng::<StdRng>()).into()
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
        self.0.to_bytes_le()
    }

    /// Parses a group from little-endian bytes (x-coordinate).
    #[staticmethod]
    fn from_bytes_le(bytes: Vec<u8>) -> anyhow::Result<Self> {
        GroupNative::from_bytes_le(&bytes).map(Self)
    }

    /// Returns the little-endian bit representation of the group element.
    fn to_bits_le(&self) -> Vec<bool> {
        self.0.to_bits_le()
    }

    /// Parses a group from little-endian bits.
    #[staticmethod]
    fn from_bits_le(bits: Vec<bool>) -> anyhow::Result<Self> {
        GroupNative::from_bits_le(&bits).map(Self)
    }

    /// Returns the field elements representation.
    fn to_fields(&self) -> anyhow::Result<Vec<Field>> {
        self.0
            .to_fields()
            .map(|fs| fs.into_iter().map(Into::into).collect())
    }

    /// Converts to a Plaintext wrapper.
    fn to_plaintext(&self) -> crate::Plaintext {
        crate::Plaintext::from(PlaintextNative::Literal(
            LiteralNative::Group(self.0),
            OnceLock::new(),
        ))
    }

    // ── cast conversions ───────────────────────────────────────────────────

    /// Returns the x-coordinate field element (alias for to_x_coordinate).
    fn to_field(&self) -> Field {
        self.to_x_coordinate()
    }

    /// Cast to Scalar with lossy truncation (via x-coordinate).
    fn to_scalar_lossy(&self) -> Scalar {
        ScalarNative::from_field_lossy(&self.0.to_x_coordinate()).into()
    }

    /// Cast to Boolean with lossy truncation (LSB of x-coordinate).
    fn to_boolean_lossy(&self) -> Boolean {
        Boolean::from(crate::types::BooleanNative::new(
            self.0.to_x_coordinate().to_bits_le()[0],
        ))
    }

    // Group → Integer lossy conversions (via x-coordinate)

    /// Cast to U8 with lossy truncation.
    fn to_u8_lossy(&self) -> crate::U8 {
        crate::U8::from(U8Native::from_field_lossy(&self.0.to_x_coordinate()))
    }

    /// Cast to U16 with lossy truncation.
    fn to_u16_lossy(&self) -> crate::U16 {
        crate::U16::from(U16Native::from_field_lossy(&self.0.to_x_coordinate()))
    }

    /// Cast to U32 with lossy truncation.
    fn to_u32_lossy(&self) -> crate::U32 {
        crate::U32::from(U32Native::from_field_lossy(&self.0.to_x_coordinate()))
    }

    /// Cast to U64 with lossy truncation.
    fn to_u64_lossy(&self) -> crate::U64 {
        crate::U64::from(U64Native::from_field_lossy(&self.0.to_x_coordinate()))
    }

    /// Cast to U128 with lossy truncation.
    fn to_u128_lossy(&self) -> crate::U128 {
        crate::U128::from(U128Native::from_field_lossy(&self.0.to_x_coordinate()))
    }

    /// Cast to I8 with lossy truncation.
    fn to_i8_lossy(&self) -> crate::I8 {
        crate::I8::from(I8Native::from_field_lossy(&self.0.to_x_coordinate()))
    }

    /// Cast to I16 with lossy truncation.
    fn to_i16_lossy(&self) -> crate::I16 {
        crate::I16::from(I16Native::from_field_lossy(&self.0.to_x_coordinate()))
    }

    /// Cast to I32 with lossy truncation.
    fn to_i32_lossy(&self) -> crate::I32 {
        crate::I32::from(I32Native::from_field_lossy(&self.0.to_x_coordinate()))
    }

    /// Cast to I64 with lossy truncation.
    fn to_i64_lossy(&self) -> crate::I64 {
        crate::I64::from(I64Native::from_field_lossy(&self.0.to_x_coordinate()))
    }

    /// Cast to I128 with lossy truncation.
    fn to_i128_lossy(&self) -> crate::I128 {
        crate::I128::from(I128Native::from_field_lossy(&self.0.to_x_coordinate()))
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
