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
        IdentifierNative, LiteralNative, PlaintextNative, RecordCiphertextNative,
        RecordPlaintextNative,
    },
    Field, GraphKey, Group, Identifier, Plaintext, PrivateKey, ProgramID, ViewKey,
};
use snarkvm::prelude::Entry;
use std::ops::Deref;
use std::str::FromStr;

use pyo3::prelude::*;

/// A value(ciphertext) stored in program record.
#[pyclass(frozen)]
pub struct RecordCiphertext(RecordCiphertextNative);

#[pymethods]
impl RecordCiphertext {
    /// Creates a record ciphertext from string
    #[staticmethod]
    fn from_string(s: &str) -> anyhow::Result<Self> {
        RecordCiphertextNative::from_str(s).map(Self)
    }

    /// Decrypts self into plaintext using the given view key and checks that the owner matches the view key.
    pub fn decrypt(&self, view_key: &ViewKey) -> anyhow::Result<RecordPlaintext> {
        self.0.decrypt(view_key).map(Into::into)
    }

    /// Determines whether the record belongs to the view key associated with an account.
    pub fn is_owner(&self, view_key: &ViewKey) -> bool {
        self.0.is_owner(view_key)
    }

    /// Returns the nonce of the record ciphertext.
    #[getter]
    fn nonce(&self) -> Group {
        (*self.0.nonce()).into()
    }

    /// Computes the record view key as `(nonce * view_key_scalar).to_x_coordinate()`.
    pub fn record_view_key(&self, view_key: &ViewKey) -> Field {
        (*self.0.nonce() * ***view_key).to_x_coordinate().into()
    }

    /// Decrypts self into plaintext using the given record view key (unchecked — no owner verification).
    pub fn decrypt_with_record_view_key(
        &self,
        record_view_key: &Field,
    ) -> anyhow::Result<RecordPlaintext> {
        self.0
            .decrypt_symmetric_unchecked(&**record_view_key)
            .map(Into::into)
    }

    /// Computes the record tag from a graph key and commitment.
    #[staticmethod]
    fn tag(graph_key: &GraphKey, commitment: &Field) -> anyhow::Result<Field> {
        #[allow(clippy::useless_conversion)]
        RecordPlaintextNative::tag(graph_key.sk_tag().into(), commitment.clone().into())
            .map(Into::into)
    }

    /// Returns the record ciphertext as a string.
    fn __str__(&self) -> String {
        self.0.to_string()
    }
}

impl Deref for RecordCiphertext {
    type Target = RecordCiphertextNative;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<RecordCiphertextNative> for RecordCiphertext {
    fn from(value: RecordCiphertextNative) -> Self {
        Self(value)
    }
}

/// A value(plaintext) stored in program record.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct RecordPlaintext(RecordPlaintextNative);

#[pymethods]
impl RecordPlaintext {
    /// Reads in the plaintext string.
    #[staticmethod]
    fn from_string(s: &str) -> anyhow::Result<Self> {
        RecordPlaintextNative::from_str(s).map(Self)
    }

    /// Returns the version of the program record.
    #[getter]
    fn version(&self) -> u8 {
        **self.0.version()
    }

    /// Returns the owner of the record as a string.
    #[getter]
    fn owner(&self) -> String {
        self.0.owner().to_string()
    }

    /// Returns the nonce of the program record.
    #[getter]
    fn nonce(&self) -> Group {
        (*self.0.nonce()).into()
    }

    /// Returns the amount of microcredits in the record (0 if no microcredits field).
    #[getter]
    fn microcredits(&self) -> u64 {
        self.0
            .find(&[IdentifierNative::from_str("microcredits").unwrap()])
            .ok()
            .and_then(|entry| match entry {
                Entry::Private(PlaintextNative::Literal(LiteralNative::U64(amount), _)) => {
                    Some(*amount)
                }
                Entry::Public(PlaintextNative::Literal(LiteralNative::U64(amount), _)) => {
                    Some(*amount)
                }
                _ => None,
            })
            .unwrap_or(0)
    }

    /// Returns the commitment of the record for the given program_id, record_name, and record_view_key.
    pub fn commitment(
        &self,
        program_id: &ProgramID,
        record_name: &Identifier,
        record_view_key: &Field,
    ) -> anyhow::Result<Field> {
        self.0
            .to_commitment(&**program_id, &**record_name, &**record_view_key)
            .map(Into::into)
    }

    /// Computes the record view key as `(nonce * view_key_scalar).to_x_coordinate()`.
    pub fn record_view_key(&self, view_key: &ViewKey) -> Field {
        (*self.0.nonce() * ***view_key).to_x_coordinate().into()
    }

    /// Computes the record tag from a graph key and commitment.
    pub fn tag(&self, graph_key: &GraphKey, commitment: &Field) -> anyhow::Result<Field> {
        #[allow(clippy::useless_conversion)]
        RecordPlaintextNative::tag(graph_key.sk_tag().into(), commitment.clone().into())
            .map(Into::into)
    }

    /// Returns the record entry with the given name as a Plaintext.
    pub fn get_member(&self, name: &str) -> anyhow::Result<Plaintext> {
        let id = IdentifierNative::from_str(name)?;
        let entry = self
            .0
            .data()
            .get(&id)
            .ok_or_else(|| anyhow::anyhow!("Record member '{}' not found", name))?;
        let plaintext = match entry {
            Entry::Constant(p) => p.clone(),
            Entry::Public(p) => p.clone(),
            Entry::Private(p) => p.clone(),
        };
        Ok(plaintext.into())
    }

    /// Attempt to get the serial number of a record to determine whether or not is has been spent
    pub fn serial_number(
        &self,
        private_key: &PrivateKey,
        program_id: &ProgramID,
        record_identifier: &Identifier,
        record_view_key: &Field,
    ) -> anyhow::Result<Field> {
        let commitment =
            self.0
                .to_commitment(&**program_id, &**record_identifier, &**record_view_key)?;
        RecordPlaintextNative::serial_number(**private_key, commitment).map(Into::into)
    }

    /// Returns the plaintext as a string.
    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Deref for RecordPlaintext {
    type Target = RecordPlaintextNative;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<RecordPlaintextNative> for RecordPlaintext {
    fn from(value: RecordPlaintextNative) -> Self {
        Self(value)
    }
}

impl From<RecordPlaintext> for RecordPlaintextNative {
    fn from(value: RecordPlaintext) -> Self {
        value.0
    }
}
