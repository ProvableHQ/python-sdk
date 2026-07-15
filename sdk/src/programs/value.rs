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
    types::{LiteralNative, PlaintextNative, RecordPlaintextNative, ValueNative},
    Field, Literal, Plaintext, RecordPlaintext,
};
use snarkvm::prelude::{FromBytes, ToBits, ToBytes, ToFields};

use pyo3::prelude::*;

use std::str::FromStr;

/// The Aleo value type to interact with a call to an Aleo program.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct Value(ValueNative);

#[pymethods]
impl Value {
    /// Parses a string (Future, Plaintext, Record) into a value.
    #[staticmethod]
    fn parse(s: &str) -> anyhow::Result<Self> {
        ValueNative::from_str(s).map(Self)
    }

    /// Initializes the value from a literal.
    #[staticmethod]
    fn from_literal(literal: Literal) -> Self {
        Self(ValueNative::from(LiteralNative::from(literal)))
    }

    /// Initializes the value from a record.
    #[staticmethod]
    fn from_record_plaintext(record_plaintext: RecordPlaintext) -> Self {
        Self(ValueNative::from(RecordPlaintextNative::from(
            record_plaintext,
        )))
    }

    /// Creates a Value wrapping the given Plaintext.
    #[staticmethod]
    pub fn from_plaintext(plaintext: &Plaintext) -> Self {
        use std::ops::Deref;
        let native: PlaintextNative = plaintext.deref().clone();
        Self(ValueNative::Plaintext(native))
    }

    /// Extracts the inner Plaintext. Errors if this is not a Plaintext variant.
    pub fn to_plaintext(&self) -> anyhow::Result<Plaintext> {
        match &self.0 {
            ValueNative::Plaintext(p) => Ok(Plaintext::from(p.clone())),
            _ => anyhow::bail!("Value is not a Plaintext variant"),
        }
    }

    /// Extracts the inner Record as a RecordPlaintext. Errors if not a Record variant.
    pub fn to_record_plaintext(&self) -> anyhow::Result<RecordPlaintext> {
        match &self.0 {
            ValueNative::Record(r) => Ok(RecordPlaintext::from(r.clone())),
            _ => anyhow::bail!("Value is not a Record variant"),
        }
    }

    /// Returns true if this value is a Plaintext variant.
    pub fn is_plaintext(&self) -> bool {
        matches!(&self.0, ValueNative::Plaintext(..))
    }

    /// Returns true if this value is a Record variant.
    pub fn is_record(&self) -> bool {
        matches!(&self.0, ValueNative::Record(..))
    }

    /// Returns true if this value is a Future variant.
    pub fn is_future(&self) -> bool {
        matches!(&self.0, ValueNative::Future(..))
    }

    /// Returns the variant type as a string: "plaintext", "record", "future", "dynamic_record", or "dynamic_future".
    #[getter]
    pub fn value_type(&self) -> String {
        match &self.0 {
            ValueNative::Plaintext(..) => "plaintext".to_string(),
            ValueNative::Record(..) => "record".to_string(),
            ValueNative::Future(..) => "future".to_string(),
            ValueNative::DynamicRecord(..) => "dynamic_record".to_string(),
            ValueNative::DynamicFuture(..) => "dynamic_future".to_string(),
        }
    }

    /// Returns the byte representation of the value (little-endian).
    pub fn bytes(&self) -> anyhow::Result<Vec<u8>> {
        self.0.to_bytes_le()
    }

    /// Recovers a Value from its little-endian byte representation.
    #[staticmethod]
    pub fn from_bytes(bytes: Vec<u8>) -> anyhow::Result<Self> {
        Ok(Self(ValueNative::read_le(&bytes[..])?))
    }

    /// Returns the little-endian bit representation.
    pub fn to_bits_le(&self) -> Vec<bool> {
        self.0.to_bits_le()
    }

    /// Returns the field element encoding of the value.
    pub fn to_fields(&self) -> anyhow::Result<Vec<Field>> {
        Ok(self.0.to_fields()?.into_iter().map(Into::into).collect())
    }

    /// Returns the value as a string.
    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    #[classattr]
    const __hash__: Option<PyObject> = None;
}

impl From<ValueNative> for Value {
    fn from(value: ValueNative) -> Self {
        Self(value)
    }
}

impl From<Value> for ValueNative {
    fn from(value: Value) -> Self {
        value.0
    }
}
