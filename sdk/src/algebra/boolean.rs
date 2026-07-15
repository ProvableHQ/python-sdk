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
        BooleanNative, FieldNative, GroupNative, I128Native, I16Native, I32Native, I64Native,
        I8Native, LiteralNative, PlaintextNative, ScalarNative, U128Native, U16Native, U32Native,
        U64Native, U8Native,
    },
    Address, Field, Group, Scalar,
};

use pyo3::prelude::*;
use rand::rngs::StdRng;
use snarkvm::prelude::{FromBits, FromBytes, FromField, ToBits, ToBytes, Uniform};

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    str::FromStr,
    sync::OnceLock,
};

/// The Aleo boolean type.
#[pyclass(frozen, from_py_object)]
#[derive(Copy, Clone)]
pub struct Boolean(BooleanNative);

#[allow(clippy::wrong_self_convention)]
#[pymethods]
impl Boolean {
    #[new]
    fn new(value: bool) -> Self {
        Self(BooleanNative::new(value))
    }

    /// Parses a boolean from a string ("true"/"false").
    #[staticmethod]
    fn from_string(s: &str) -> anyhow::Result<Self> {
        BooleanNative::from_str(s).map(Self)
    }

    /// Returns a random boolean.
    #[staticmethod]
    fn random() -> Self {
        Self(<BooleanNative as Uniform>::rand(&mut rand::make_rng::<
            StdRng,
        >()))
    }

    // ── serialization ──────────────────────────────────────────────────────

    /// Returns the little-endian byte representation.
    fn to_bytes_le(&self) -> anyhow::Result<Vec<u8>> {
        self.0.to_bytes_le()
    }

    /// Constructs from little-endian bytes.
    #[staticmethod]
    fn from_bytes_le(bytes: Vec<u8>) -> anyhow::Result<Self> {
        BooleanNative::from_bytes_le(&bytes).map(Self)
    }

    /// Returns the little-endian bit representation (a single-element Vec).
    fn to_bits_le(&self) -> Vec<bool> {
        self.0.to_bits_le()
    }

    /// Constructs from little-endian bits (must be exactly 1 element).
    #[staticmethod]
    fn from_bits_le(bits: Vec<bool>) -> anyhow::Result<Self> {
        BooleanNative::from_bits_le(&bits).map(Self)
    }

    // ── dunders ────────────────────────────────────────────────────────────

    /// Returns the boolean as a string.
    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __bool__(&self) -> bool {
        *self.0
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.0.hash(&mut hasher);
        hasher.finish()
    }

    /// Logical NOT dunder.
    fn __invert__(&self) -> Self {
        self.not_()
    }

    /// Logical AND dunder.
    fn __and__(&self, other: &Self) -> Self {
        self.and_(other)
    }

    /// Logical OR dunder.
    fn __or__(&self, other: &Self) -> Self {
        self.or_(other)
    }

    /// Logical XOR dunder.
    fn __xor__(&self, other: &Self) -> Self {
        self.xor(other)
    }

    // ── named logical ops ──────────────────────────────────────────────────

    /// Logical NOT.
    fn not_(&self) -> Self {
        Self(!self.0)
    }

    /// Logical AND.
    fn and_(&self, other: &Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Logical OR.
    fn or_(&self, other: &Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Logical XOR.
    fn xor(&self, other: &Self) -> Self {
        Self(self.0 ^ other.0)
    }

    /// Logical NAND (NOT AND).
    fn nand(&self, other: &Self) -> Self {
        Self(!(self.0 & other.0))
    }

    /// Logical NOR (NOT OR).
    fn nor(&self, other: &Self) -> Self {
        Self(!(self.0 | other.0))
    }

    // ── conversions ────────────────────────────────────────────────────────

    /// Converts to a Plaintext wrapper.
    fn to_plaintext(&self) -> crate::Plaintext {
        crate::Plaintext::from(PlaintextNative::Literal(
            LiteralNative::Boolean(self.0),
            OnceLock::new(),
        ))
    }

    /// Converts to a Field element (false=0field, true=1field).
    fn to_field(&self) -> anyhow::Result<Field> {
        FieldNative::from_bits_le(&[*self.0]).map(Into::into)
    }

    /// Converts to a Scalar element (false=0scalar, true=1scalar).
    fn to_scalar(&self) -> anyhow::Result<Scalar> {
        ScalarNative::from_bits_le(&[*self.0]).map(Into::into)
    }

    /// Converts to a Group element via x-coordinate recovery (strict).
    fn to_group(&self) -> anyhow::Result<Group> {
        let f = FieldNative::from_bits_le(&[*self.0])?;
        GroupNative::from_field(&f).map(Into::into)
    }

    /// Converts to an Address via Group (strict).
    fn to_address(&self) -> anyhow::Result<Address> {
        let f = FieldNative::from_bits_le(&[*self.0])?;
        let g = GroupNative::from_field(&f)?;
        Ok(Address::new(g))
    }

    // Boolean → Integer conversions (false=0, true=1, lossless).

    /// Converts to U8 (false=0, true=1).
    fn to_u8(&self) -> crate::U8 {
        crate::U8::from(U8Native::new(if *self.0 { 1 } else { 0 }))
    }

    /// Converts to U16 (false=0, true=1).
    fn to_u16(&self) -> crate::U16 {
        crate::U16::from(U16Native::new(if *self.0 { 1 } else { 0 }))
    }

    /// Converts to U32 (false=0, true=1).
    fn to_u32(&self) -> crate::U32 {
        crate::U32::from(U32Native::new(if *self.0 { 1 } else { 0 }))
    }

    /// Converts to U64 (false=0, true=1).
    fn to_u64(&self) -> crate::U64 {
        crate::U64::from(U64Native::new(if *self.0 { 1 } else { 0 }))
    }

    /// Converts to U128 (false=0, true=1).
    fn to_u128(&self) -> crate::U128 {
        crate::U128::from(U128Native::new(if *self.0 { 1 } else { 0 }))
    }

    /// Converts to I8 (false=0, true=1).
    fn to_i8(&self) -> crate::I8 {
        crate::I8::from(I8Native::new(if *self.0 { 1 } else { 0 }))
    }

    /// Converts to I16 (false=0, true=1).
    fn to_i16(&self) -> crate::I16 {
        crate::I16::from(I16Native::new(if *self.0 { 1 } else { 0 }))
    }

    /// Converts to I32 (false=0, true=1).
    fn to_i32(&self) -> crate::I32 {
        crate::I32::from(I32Native::new(if *self.0 { 1 } else { 0 }))
    }

    /// Converts to I64 (false=0, true=1).
    fn to_i64(&self) -> crate::I64 {
        crate::I64::from(I64Native::new(if *self.0 { 1 } else { 0 }))
    }

    /// Converts to I128 (false=0, true=1).
    fn to_i128(&self) -> crate::I128 {
        crate::I128::from(I128Native::new(if *self.0 { 1 } else { 0 }))
    }
}

impl From<Boolean> for BooleanNative {
    fn from(value: Boolean) -> Self {
        value.0
    }
}

impl From<BooleanNative> for Boolean {
    fn from(value: BooleanNative) -> Self {
        Self(value)
    }
}
