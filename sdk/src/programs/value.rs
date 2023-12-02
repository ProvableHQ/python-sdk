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
    types::{LiteralNative, RecordPlaintextNative, ValueNative},
    Literal, RecordPlaintext,
};

use pyo3::prelude::*;

use std::str::FromStr;

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
