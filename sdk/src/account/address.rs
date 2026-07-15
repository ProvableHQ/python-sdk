// Copyright (C) 2019-2026 Provable Inc.
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
        AddressNative, BooleanNative, FieldNative, I128Native, I16Native, I32Native, I64Native,
        I8Native, ProgramIDNative, ScalarNative, U128Native, U16Native, U32Native, U64Native,
        U8Native,
    },
    Boolean, Field, Group, Plaintext, Scalar, I128, I16, I32, I64, I8, U128, U16, U32, U64, U8,
};

use pyo3::prelude::*;
use snarkvm::prelude::{FromBits, FromFields, Network, ToBits, ToField, ToFields};

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

    /// Returns the address of a program based on the program ID string.
    #[staticmethod]
    pub fn from_program_id(program_id: &str) -> anyhow::Result<Self> {
        let program_id_native = ProgramIDNative::from_str(program_id)?;
        let name_field = program_id_native.name().to_field()?;
        let network_field = program_id_native.network().to_field()?;
        let group = <crate::types::CurrentNetwork as Network>::hash_to_group_psd4(&[
            name_field,
            network_field,
        ])?;
        Ok(Self(AddressNative::new(group)))
    }

    /// Returns true if the string is a valid Aleo address (auto-lowercased before validation).
    #[staticmethod]
    pub fn is_valid(address: &str) -> bool {
        AddressNative::from_str(&address.to_lowercase()).is_ok()
    }

    /// Returns the little-endian bit representation of the address.
    pub fn to_bits_le(&self) -> Vec<bool> {
        self.0.to_bits_le()
    }

    /// Recovers an address from little-endian bits.
    #[staticmethod]
    pub fn from_bits_le(bits: Vec<bool>) -> anyhow::Result<Self> {
        AddressNative::from_bits_le(&bits).map(Self)
    }

    /// Returns the field element encoding of the address.
    pub fn to_fields(&self) -> anyhow::Result<Vec<Field>> {
        Ok(self.0.to_fields()?.into_iter().map(Into::into).collect())
    }

    /// Recovers an address from field elements.
    #[staticmethod]
    pub fn from_fields(fields: Vec<Field>) -> anyhow::Result<Self> {
        let native: Vec<FieldNative> = fields.into_iter().map(Into::into).collect();
        AddressNative::from_fields(&native).map(Self)
    }

    /// Returns the address wrapped as a Plaintext::Literal(Address).
    pub fn to_plaintext(&self) -> Plaintext {
        use crate::types::{LiteralNative, PlaintextNative};
        Plaintext::from(PlaintextNative::from(LiteralNative::Address(self.0)))
    }

    /// Cast the address to a Scalar with lossy truncation (via x-coordinate).
    pub fn to_scalar_lossy(&self) -> Scalar {
        ScalarNative::from_field_lossy(&self.0.to_group().to_x_coordinate()).into()
    }

    /// Cast the address to a Boolean (LSB of x-coordinate).
    pub fn to_boolean_lossy(&self) -> Boolean {
        let bit = self.0.to_group().to_x_coordinate().to_bits_le()[0];
        BooleanNative::new(bit).into()
    }

    /// Cast the address to a U8 with lossy truncation.
    pub fn to_u8_lossy(&self) -> U8 {
        U8Native::from_field_lossy(&self.0.to_group().to_x_coordinate()).into()
    }

    /// Cast the address to a U16 with lossy truncation.
    pub fn to_u16_lossy(&self) -> U16 {
        U16Native::from_field_lossy(&self.0.to_group().to_x_coordinate()).into()
    }

    /// Cast the address to a U32 with lossy truncation.
    pub fn to_u32_lossy(&self) -> U32 {
        U32Native::from_field_lossy(&self.0.to_group().to_x_coordinate()).into()
    }

    /// Cast the address to a U64 with lossy truncation.
    pub fn to_u64_lossy(&self) -> U64 {
        U64Native::from_field_lossy(&self.0.to_group().to_x_coordinate()).into()
    }

    /// Cast the address to a U128 with lossy truncation.
    pub fn to_u128_lossy(&self) -> U128 {
        U128Native::from_field_lossy(&self.0.to_group().to_x_coordinate()).into()
    }

    /// Cast the address to an I8 with lossy truncation.
    pub fn to_i8_lossy(&self) -> I8 {
        I8Native::from_field_lossy(&self.0.to_group().to_x_coordinate()).into()
    }

    /// Cast the address to an I16 with lossy truncation.
    pub fn to_i16_lossy(&self) -> I16 {
        I16Native::from_field_lossy(&self.0.to_group().to_x_coordinate()).into()
    }

    /// Cast the address to an I32 with lossy truncation.
    pub fn to_i32_lossy(&self) -> I32 {
        I32Native::from_field_lossy(&self.0.to_group().to_x_coordinate()).into()
    }

    /// Cast the address to an I64 with lossy truncation.
    pub fn to_i64_lossy(&self) -> I64 {
        I64Native::from_field_lossy(&self.0.to_group().to_x_coordinate()).into()
    }

    /// Cast the address to an I128 with lossy truncation.
    pub fn to_i128_lossy(&self) -> I128 {
        I128Native::from_field_lossy(&self.0.to_group().to_x_coordinate()).into()
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
