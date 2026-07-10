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
        FieldNative, GroupNative, I128Native, I16Native, I32Native, I64Native, I8Native,
        LiteralNative, PlaintextNative, ScalarNative, U128Native, U16Native, U32Native, U64Native,
        U8Native,
    },
    Address, Boolean, Field, Group,
};

use pyo3::{exceptions::PyZeroDivisionError, prelude::*};
use rand::rngs::StdRng;
use snarkvm::prelude::{
    CastLossy, Double, FromBits, FromBytes, FromField, One, Pow, ToBits, ToBytes, ToField, Uniform,
    Zero,
};

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    ops::Deref,
    str::FromStr,
    sync::OnceLock,
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

    /// Returns a random scalar element.
    #[staticmethod]
    fn random() -> Self {
        <ScalarNative as Uniform>::rand(&mut rand::make_rng::<StdRng>()).into()
    }

    /// Returns the little-endian byte representation of the scalar element.
    fn to_bytes_le(&self) -> anyhow::Result<Vec<u8>> {
        self.0.to_bytes_le()
    }

    /// Parses a scalar from little-endian bytes.
    #[staticmethod]
    fn from_bytes_le(bytes: Vec<u8>) -> anyhow::Result<Self> {
        ScalarNative::from_bytes_le(&bytes).map(Self)
    }

    /// Returns the little-endian bit representation of the scalar element.
    fn to_bits_le(&self) -> Vec<bool> {
        self.0.to_bits_le()
    }

    /// Parses a scalar from little-endian bits.
    #[staticmethod]
    fn from_bits_le(bits: Vec<bool>) -> anyhow::Result<Self> {
        ScalarNative::from_bits_le(&bits).map(Self)
    }

    /// Returns this scalar as a field element.
    fn to_field(&self) -> anyhow::Result<Field> {
        self.0.to_field().map(Into::into)
    }

    /// Doubles this scalar element.
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

    /// Returns self divided by other (errors on zero divisor).
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

    /// Converts to a Plaintext wrapper.
    fn to_plaintext(&self) -> crate::Plaintext {
        crate::Plaintext::from(PlaintextNative::Literal(
            LiteralNative::Scalar(self.0),
            OnceLock::new(),
        ))
    }

    // ── cast conversions ───────────────────────────────────────────────────

    /// Converts to a Boolean (strict: must be zero or one).
    fn to_boolean(&self) -> anyhow::Result<Boolean> {
        if self.0.is_zero() {
            Ok(Boolean::from(crate::types::BooleanNative::new(false)))
        } else if self.0.is_one() {
            Ok(Boolean::from(crate::types::BooleanNative::new(true)))
        } else {
            Err(anyhow::anyhow!(
                "Failed to convert scalar to boolean: scalar is not zero or one"
            ))
        }
    }

    /// Converts to a Boolean with lossy truncation (LSB).
    fn to_boolean_lossy(&self) -> Boolean {
        Boolean::from(crate::types::BooleanNative::new(self.0.to_bits_le()[0]))
    }

    /// Converts to a Group element (strict, via Field x-coordinate recovery).
    fn to_group(&self) -> anyhow::Result<Group> {
        let f: FieldNative = self.0.to_field()?;
        GroupNative::from_field(&f).map(Into::into)
    }

    /// Converts to a Group element with lossy conversion (Elligator-2 fallback).
    fn to_group_lossy(&self) -> anyhow::Result<Group> {
        let f: FieldNative = self.0.to_field()?;
        let g: GroupNative = f.cast_lossy();
        Ok(Group::from(g))
    }

    /// Converts to an Address (strict, via Group x-coordinate recovery).
    fn to_address(&self) -> anyhow::Result<Address> {
        let f: FieldNative = self.0.to_field()?;
        let g = GroupNative::from_field(&f)?;
        Ok(Address::new(g))
    }

    /// Converts to an Address with lossy conversion.
    fn to_address_lossy(&self) -> anyhow::Result<Address> {
        let f: FieldNative = self.0.to_field()?;
        let g: GroupNative = f.cast_lossy();
        Ok(Address::new(g))
    }

    // Scalar → Integer lossy conversions (via field)

    /// Cast to U8 with lossy truncation.
    fn to_u8_lossy(&self) -> anyhow::Result<crate::U8> {
        let f = self.0.to_field()?;
        Ok(crate::U8::from(U8Native::from_field_lossy(&f)))
    }

    /// Cast to U16 with lossy truncation.
    fn to_u16_lossy(&self) -> anyhow::Result<crate::U16> {
        let f = self.0.to_field()?;
        Ok(crate::U16::from(U16Native::from_field_lossy(&f)))
    }

    /// Cast to U32 with lossy truncation.
    fn to_u32_lossy(&self) -> anyhow::Result<crate::U32> {
        let f = self.0.to_field()?;
        Ok(crate::U32::from(U32Native::from_field_lossy(&f)))
    }

    /// Cast to U64 with lossy truncation.
    fn to_u64_lossy(&self) -> anyhow::Result<crate::U64> {
        let f = self.0.to_field()?;
        Ok(crate::U64::from(U64Native::from_field_lossy(&f)))
    }

    /// Cast to U128 with lossy truncation.
    fn to_u128_lossy(&self) -> anyhow::Result<crate::U128> {
        let f = self.0.to_field()?;
        Ok(crate::U128::from(U128Native::from_field_lossy(&f)))
    }

    /// Cast to I8 with lossy truncation.
    fn to_i8_lossy(&self) -> anyhow::Result<crate::I8> {
        let f = self.0.to_field()?;
        Ok(crate::I8::from(I8Native::from_field_lossy(&f)))
    }

    /// Cast to I16 with lossy truncation.
    fn to_i16_lossy(&self) -> anyhow::Result<crate::I16> {
        let f = self.0.to_field()?;
        Ok(crate::I16::from(I16Native::from_field_lossy(&f)))
    }

    /// Cast to I32 with lossy truncation.
    fn to_i32_lossy(&self) -> anyhow::Result<crate::I32> {
        let f = self.0.to_field()?;
        Ok(crate::I32::from(I32Native::from_field_lossy(&f)))
    }

    /// Cast to I64 with lossy truncation.
    fn to_i64_lossy(&self) -> anyhow::Result<crate::I64> {
        let f = self.0.to_field()?;
        Ok(crate::I64::from(I64Native::from_field_lossy(&f)))
    }

    /// Cast to I128 with lossy truncation.
    fn to_i128_lossy(&self) -> anyhow::Result<crate::I128> {
        let f = self.0.to_field()?;
        Ok(crate::I128::from(I128Native::from_field_lossy(&f)))
    }

    // ── dunders ────────────────────────────────────────────────────────────

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

    fn __pow__(&self, other: Self, _modulo: Option<u32>) -> Self {
        self.pow(other)
    }

    fn __neg__(&self) -> Self {
        self.negate()
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
