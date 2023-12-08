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
    types::LiteralNative, Address, Boolean, Field, Group, Scalar, Signature, I128, I16, I32, I64,
    I8, U128, U16, U32, U64, U8,
};

use pyo3::prelude::*;

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    str::FromStr,
};

/// The literal type represents all supported types in snarkVM.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct Literal(LiteralNative);

#[pymethods]
impl Literal {
    /// Parses a literal from a string.
    #[staticmethod]
    fn parse(s: &str) -> anyhow::Result<Self> {
        LiteralNative::from_str(s).map(Self)
    }

    #[staticmethod]
    fn from_address(address: Address) -> Self {
        Self(LiteralNative::Address(*address))
    }

    #[staticmethod]
    fn from_field(field: Field) -> Self {
        Self(LiteralNative::Field(*field))
    }

    #[staticmethod]
    fn from_group(group: Group) -> Self {
        Self(LiteralNative::Group(*group))
    }

    #[staticmethod]
    fn from_scalar(scalar: Scalar) -> Self {
        Self(LiteralNative::Scalar(*scalar))
    }

    #[staticmethod]
    fn from_signature(signature: Signature) -> Self {
        Self(LiteralNative::Signature(Box::new(*signature)))
    }

    #[staticmethod]
    fn from_boolean(b: Boolean) -> Self {
        Self(LiteralNative::Boolean(b.into()))
    }

    #[staticmethod]
    fn from_i8(value: I8) -> Self {
        Self(LiteralNative::I8(value.into()))
    }

    #[staticmethod]
    fn from_i16(value: I16) -> Self {
        Self(LiteralNative::I16(value.into()))
    }

    #[staticmethod]
    fn from_i32(value: I32) -> Self {
        Self(LiteralNative::I32(value.into()))
    }

    #[staticmethod]
    fn from_i64(value: I64) -> Self {
        Self(LiteralNative::I64(value.into()))
    }

    #[staticmethod]
    fn from_i128(value: I128) -> Self {
        Self(LiteralNative::I128(value.into()))
    }

    #[staticmethod]
    fn from_u8(value: U8) -> Self {
        Self(LiteralNative::U8(value.into()))
    }

    #[staticmethod]
    fn from_u16(value: U16) -> Self {
        Self(LiteralNative::U16(value.into()))
    }

    #[staticmethod]
    fn from_u32(value: U32) -> Self {
        Self(LiteralNative::U32(value.into()))
    }

    #[staticmethod]
    fn from_u64(value: U64) -> Self {
        Self(LiteralNative::U64(value.into()))
    }

    #[staticmethod]
    fn from_u128(value: U128) -> Self {
        Self(LiteralNative::U128(value.into()))
    }

    /// Returns the type of the literal.
    fn type_name(&self) -> String {
        self.0.to_type().type_name().to_string()
    }

    /// Returns the literal as a string.
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

impl From<Literal> for LiteralNative {
    fn from(value: Literal) -> Self {
        value.0
    }
}
