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
    types::{CiphertextNative, LiteralNative, PlaintextNative},
    Address, Field, Group, Identifier, Literal, Scalar, ViewKey,
};
use std::ops::Deref;

use once_cell::sync::OnceCell;
use pyo3::{exceptions::PyTypeError, prelude::*};

use std::{collections::HashMap, str::FromStr};

/// The Aleo ciphertext type.
#[pyclass(frozen)]
pub struct Ciphertext(CiphertextNative);

#[pymethods]
impl Ciphertext {
    /// Reads in the ciphertext string.
    #[staticmethod]
    fn from_string(s: &str) -> anyhow::Result<Self> {
        CiphertextNative::from_str(s).map(Self)
    }

    /// Decrypts self into plaintext using the given account view key & nonce.
    pub fn decrypt(&self, view_key: ViewKey, nonce: Group) -> anyhow::Result<Plaintext> {
        self.0
            .decrypt(view_key.into(), nonce.into())
            .map(Into::into)
    }

    /// Decrypts self into plaintext using the given plaintext view key.
    pub fn decrypt_symmetric(&self, plaintext_view_key: Field) -> anyhow::Result<Plaintext> {
        self.0
            .decrypt_symmetric(plaintext_view_key.into())
            .map(Into::into)
    }

    /// Returns the ciphertext as a string.
    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Deref for Ciphertext {
    type Target = CiphertextNative;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<CiphertextNative> for Ciphertext {
    fn from(value: CiphertextNative) -> Self {
        Self(value)
    }
}

/// The Aleo plaintext type.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct Plaintext(PlaintextNative);

#[pymethods]
impl Plaintext {
    /// Returns a plaintext from a string literal.
    #[staticmethod]
    fn from_string(s: &str) -> anyhow::Result<Self> {
        PlaintextNative::from_str(s).map(Self)
    }

    /// Returns a new Plaintext from a Literal.
    #[staticmethod]
    fn new_literal(literal: Literal) -> Self {
        PlaintextNative::from(LiteralNative::from(literal)).into()
    }

    /// Returns a new Plaintext::Struct from a list of (key, value).
    #[staticmethod]
    fn new_struct(kv: Vec<(Identifier, Plaintext)>) -> Self {
        let kv: Vec<_> = kv.into_iter().map(|(k, v)| (k.into(), v.into())).collect();
        PlaintextNative::Struct(indexmap::IndexMap::from_iter(kv), OnceCell::new()).into()
    }

    /// Returns a new Plaintext::Array from a list of values.
    #[staticmethod]
    fn new_array(values: Vec<Plaintext>) -> Self {
        let values: Vec<_> = values.into_iter().map(Into::into).collect();
        PlaintextNative::Array(values, OnceCell::new()).into()
    }

    /// Encrypts self to the given address under the given randomizer.
    fn encrypt(&self, address: Address, randomizer: Scalar) -> anyhow::Result<Ciphertext> {
        self.0.encrypt(&address, randomizer.into()).map(Into::into)
    }

    /// Encrypts self under the given plaintext view key.
    fn encrypt_symmetric(&self, plaintext_view_key: Field) -> anyhow::Result<Ciphertext> {
        self.0
            .encrypt_symmetric(plaintext_view_key.into())
            .map(Into::into)
    }

    /// Returns true if self if Plaintext::Literal.
    fn is_literal(&self) -> bool {
        matches!(self.0, PlaintextNative::Literal(..))
    }

    /// Returns true if self if Plaintext::Struct.
    fn is_struct(&self) -> bool {
        matches!(self.0, PlaintextNative::Struct(..))
    }

    /// Returns true if self if Plaintext::Array
    fn is_array(&self) -> bool {
        matches!(self.0, PlaintextNative::Array(..))
    }

    /// Unboxes the underlying Plaintext::Literal.
    fn as_literal(&self) -> PyResult<Literal> {
        match &self.0 {
            PlaintextNative::Literal(literal, _) => Ok(literal.clone().into()),
            _ => Err(PyTypeError::new_err("Plaintext is not a literal")),
        }
    }

    /// Unboxes the underlying Plaintext::Struct.
    fn as_struct(&self) -> PyResult<HashMap<Identifier, Plaintext>> {
        match &self.0 {
            PlaintextNative::Struct(s, _) => {
                let res: HashMap<Identifier, Plaintext> = s
                    .clone()
                    .into_iter()
                    .map(|(k, v)| (k.into(), v.into()))
                    .collect();
                Ok(res)
            }
            _ => Err(PyTypeError::new_err("Plaintext is not a struct")),
        }
    }

    /// Unboxes the underlying Plaintext::Array.
    fn as_array(&self) -> PyResult<Vec<Plaintext>> {
        match &self.0 {
            PlaintextNative::Array(s, _) => {
                let res: Vec<Plaintext> = s.clone().into_iter().map(|v| v.into()).collect();
                Ok(res)
            }
            _ => Err(PyTypeError::new_err("Plaintext is not an array")),
        }
    }

    /// Returns the plaintext as a string.
    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Deref for Plaintext {
    type Target = PlaintextNative;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<PlaintextNative> for Plaintext {
    fn from(value: PlaintextNative) -> Self {
        Self(value)
    }
}

impl From<Plaintext> for PlaintextNative {
    fn from(value: Plaintext) -> Self {
        value.0
    }
}
