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
        FieldNative, I128Native, I16Native, I32Native, I64Native, I8Native, LiteralNative,
        PlaintextNative, ScalarNative, U128Native, U16Native, U32Native, U64Native, U8Native,
    },
    Address, Boolean, Group, Scalar,
};

use pyo3::{exceptions::PyZeroDivisionError, prelude::*};
use rand::rngs::StdRng;
use snarkvm::prelude::{CastLossy, Double, FromBits, One, Pow, ToBits, Uniform, Zero};

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    ops::Deref,
    str::FromStr,
    sync::OnceLock,
};

/// The Aleo field type.
#[pyclass(frozen, from_py_object)]
#[derive(Clone)]
pub struct Field(FieldNative);

#[pymethods]
impl Field {
    /// Parses a field from a string.
    #[staticmethod]
    fn from_string(s: &str) -> anyhow::Result<Self> {
        FieldNative::from_str(s).map(Self)
    }

    /// Generates a new field using a cryptographically secure random number generator
    #[staticmethod]
    fn random() -> Self {
        <FieldNative as Uniform>::rand(&mut rand::make_rng::<StdRng>()).into()
    }

    /// Initializes a new field as a domain separator.
    #[staticmethod]
    fn domain_separator(domain: &str) -> Self {
        Self(FieldNative::new_domain_separator(domain))
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

    /// Returns the `1` element of the field.
    #[staticmethod]
    fn one() -> Self {
        FieldNative::one().into()
    }

    /// Constructs a field from little-endian bits.
    #[staticmethod]
    fn from_bits_le(bits: Vec<bool>) -> anyhow::Result<Self> {
        FieldNative::from_bits_le(&bits).map(Self)
    }

    /// Returns the little-endian byte representation of the field element.
    fn to_bytes_le(&self) -> anyhow::Result<Vec<u8>> {
        use snarkvm::prelude::ToBytes;
        self.0.to_bytes_le()
    }

    /// Parses a field from little-endian bytes.
    #[staticmethod]
    fn from_bytes_le(bytes: Vec<u8>) -> anyhow::Result<Self> {
        use snarkvm::prelude::FromBytes;
        FieldNative::from_bytes_le(&bytes).map(Self)
    }

    /// Returns the double of this field element.
    fn double(&self) -> Self {
        Self(Double::double(&self.0))
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

    /// Returns self raised to the power of other.
    fn pow(&self, other: Self) -> Self {
        Self(Pow::pow(self.0, other.0))
    }

    /// Returns the negation of self.
    fn negate(&self) -> Self {
        Self(-self.0)
    }

    /// Converts the field to a scalar using lossy truncation.
    fn to_scalar_lossy(&self) -> Scalar {
        ScalarNative::from_field_lossy(&self.0).into()
    }

    /// Converts the field to a group element (x-coordinate recovery).
    fn to_group(&self) -> anyhow::Result<Group> {
        use snarkvm::prelude::FromField;
        crate::types::GroupNative::from_field(&self.0).map(Group::from)
    }

    /// Converts the field to a group element with lossy conversion (Elligator-2 fallback).
    fn to_group_lossy(&self) -> Group {
        let g: crate::types::GroupNative = self.0.cast_lossy();
        Group::from(g)
    }

    /// Converts the field to an Address (strict, via group x-coordinate recovery).
    fn to_address(&self) -> anyhow::Result<Address> {
        use snarkvm::prelude::FromField;
        let g = crate::types::GroupNative::from_field(&self.0)?;
        Ok(Address::new(g))
    }

    /// Converts the field to an Address with lossy conversion.
    fn to_address_lossy(&self) -> Address {
        let g: crate::types::GroupNative = self.0.cast_lossy();
        Address::new(g)
    }

    /// Converts the field to a Boolean (strict: must be 0 or 1).
    fn to_boolean(&self) -> anyhow::Result<Boolean> {
        if self.0.is_zero() {
            Ok(Boolean::from(crate::types::BooleanNative::new(false)))
        } else if self.0.is_one() {
            Ok(Boolean::from(crate::types::BooleanNative::new(true)))
        } else {
            Err(anyhow::anyhow!(
                "Failed to convert field to boolean: field is not zero or one"
            ))
        }
    }

    /// Converts the field to a Boolean with lossy truncation (LSB).
    fn to_boolean_lossy(&self) -> Boolean {
        Boolean::from(crate::types::BooleanNative::new(self.0.to_bits_le()[0]))
    }

    /// Converts the field to a Plaintext wrapper.
    fn to_plaintext(&self) -> crate::Plaintext {
        crate::Plaintext::from(PlaintextNative::Literal(
            LiteralNative::Field(self.0),
            OnceLock::new(),
        ))
    }

    // Field → Integer lossy conversions

    /// Cast to U8 with lossy truncation.
    fn to_u8_lossy(&self) -> crate::U8 {
        crate::U8::from(U8Native::from_field_lossy(&self.0))
    }

    /// Cast to U16 with lossy truncation.
    fn to_u16_lossy(&self) -> crate::U16 {
        crate::U16::from(U16Native::from_field_lossy(&self.0))
    }

    /// Cast to U32 with lossy truncation.
    fn to_u32_lossy(&self) -> crate::U32 {
        crate::U32::from(U32Native::from_field_lossy(&self.0))
    }

    /// Cast to U64 with lossy truncation.
    fn to_u64_lossy(&self) -> crate::U64 {
        crate::U64::from(U64Native::from_field_lossy(&self.0))
    }

    /// Cast to U128 with lossy truncation.
    fn to_u128_lossy(&self) -> crate::U128 {
        crate::U128::from(U128Native::from_field_lossy(&self.0))
    }

    /// Cast to I8 with lossy truncation.
    fn to_i8_lossy(&self) -> crate::I8 {
        crate::I8::from(I8Native::from_field_lossy(&self.0))
    }

    /// Cast to I16 with lossy truncation.
    fn to_i16_lossy(&self) -> crate::I16 {
        crate::I16::from(I16Native::from_field_lossy(&self.0))
    }

    /// Cast to I32 with lossy truncation.
    fn to_i32_lossy(&self) -> crate::I32 {
        crate::I32::from(I32Native::from_field_lossy(&self.0))
    }

    /// Cast to I64 with lossy truncation.
    fn to_i64_lossy(&self) -> crate::I64 {
        crate::I64::from(I64Native::from_field_lossy(&self.0))
    }

    /// Cast to I128 with lossy truncation.
    fn to_i128_lossy(&self) -> crate::I128 {
        crate::I128::from(I128Native::from_field_lossy(&self.0))
    }

    // ── dunders ────────────────────────────────────────────────────────────

    /// Returns the Field as a string.
    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __mul__(&self, other: Self) -> Self {
        self.multiply(other)
    }

    fn __truediv__(&self, other: Self) -> PyResult<Self> {
        self.divide(other)
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

    fn __neg__(&self) -> Self {
        self.negate()
    }

    fn __pow__(&self, other: Self, _modulo: Option<Self>) -> Self {
        self.pow(other)
    }

    /// Returns the little-endian bit representation of the field element.
    pub fn to_bits_le(&self) -> Vec<bool> {
        self.0.to_bits_le()
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
