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
    account::ViewKey,
    types::{RecordCiphertextNative, RecordPlaintextNative},
};

use pyo3::prelude::*;

use std::str::FromStr;

#[pyclass(frozen)]
pub struct RecordCiphertext(RecordCiphertextNative);

#[pymethods]
impl RecordCiphertext {
    /// Reads in the ciphertext string.
    #[staticmethod]
    fn from_string(s: &str) -> anyhow::Result<Self> {
        let view_key = FromStr::from_str(s)?;
        Ok(Self(view_key))
    }

    /// Decrypts self into plaintext using the given view key and checks that the owner matches the view key.
    pub fn decrypt(&self, view_key: &ViewKey) -> anyhow::Result<RecordPlaintext> {
        let plaintext = self.0.decrypt(view_key)?;
        Ok(RecordPlaintext(plaintext))
    }

    /// Determines whether the record belongs to the account.
    pub fn is_owner(&self, view_key: &ViewKey) -> bool {
        self.0.is_owner(view_key)
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }
}

#[pyclass(frozen)]
pub struct RecordPlaintext(RecordPlaintextNative);

#[pymethods]
impl RecordPlaintext {
    fn __str__(&self) -> String {
        self.0.to_string()
    }
}
