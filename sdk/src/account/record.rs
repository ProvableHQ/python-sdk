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
    types::{RecordCiphertextNative, RecordPlaintextNative},
    Field, Group, Identifier, PrivateKey, ProgramID, ViewKey,
};
use std::ops::Deref;

use pyo3::prelude::*;

use std::str::FromStr;

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

    /// Returns the owner of the record as a string
    fn owner(&self) -> String {
        self.0.owner().to_string()
    }

    /// Returns the nonce of the program record.
    fn nonce(&self) -> Group {
        (*self.0.nonce()).into()
    }

    /// Attempt to get the serial number of a record to determine whether or not is has been spent
    pub fn serial_number(
        &self,
        private_key: &PrivateKey,
        program_id: &ProgramID,
        record_identifier: &Identifier,
    ) -> anyhow::Result<Field> {
        let commitment = self.to_commitment(program_id, record_identifier)?;
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
