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
    types::{CiphertextNative, IdentifierNative, LiteralNative, PlaintextNative, U32Native},
    Address, Field, Group, Identifier, Literal, Scalar, ViewKey, U32,
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
    /// Creates a ciphertext from string
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
    /// Reads in the plaintext string.
    #[staticmethod]
    fn from_string(s: &str) -> anyhow::Result<Self> {
        PlaintextNative::from_str(s).map(Self)
    }

    #[staticmethod]
    fn from_literal(literal: Literal) -> Self {
        PlaintextNative::from(LiteralNative::from(literal)).into()
    }

    #[staticmethod]
    fn new_struct(kv: Vec<(Identifier, Plaintext)>) -> Self {
        let kv: Vec<_> = kv.into_iter().map(|(k, v)| (k.into(), v.into())).collect();
        PlaintextNative::Struct(indexmap::IndexMap::from_iter(kv), OnceCell::new()).into()
    }

    fn encrypt(&self, address: Address, randomizer: Scalar) -> anyhow::Result<Ciphertext> {
        self.0.encrypt(&address, randomizer.into()).map(Into::into)
    }

    fn encrypt_symmetric(&self, plaintext_view_key: Field) -> anyhow::Result<Ciphertext> {
        self.0
            .encrypt_symmetric(plaintext_view_key.into())
            .map(Into::into)
    }

    fn find_by_identifier(&self, identifier: Identifier) -> anyhow::Result<Plaintext> {
        // FIXME use access instead of identifier
        self.0
            .find(&[IdentifierNative::from(identifier)])
            .map(Into::into)
    }

    fn find_by_index(&self, index: U32) -> anyhow::Result<Plaintext> {
        // FIXME use access instead of index
        self.0.find(&[U32Native::from(index)]).map(Into::into)
    }

    fn is_literal(&self) -> bool {
        matches!(self.0, PlaintextNative::Literal(..))
    }

    fn is_struct(&self) -> bool {
        matches!(self.0, PlaintextNative::Struct(..))
    }

    fn is_array(&self) -> bool {
        matches!(self.0, PlaintextNative::Array(..))
    }

    fn as_literal(&self) -> PyResult<Literal> {
        match &self.0 {
            PlaintextNative::Literal(literal, _) => Ok(literal.clone().into()),
            _ => Err(PyTypeError::new_err("Plaintext is not a literal")),
        }
    }

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
            _ => Err(PyTypeError::new_err("Plaintext is not a literal")),
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
