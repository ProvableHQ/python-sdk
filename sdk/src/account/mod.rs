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

mod address;
pub use address::Address;

mod compute_key;
pub use compute_key::ComputeKey;

mod private_key;
pub use private_key::PrivateKey;

mod record;
pub use record::{RecordCiphertext, RecordPlaintext};

mod signature;
pub use signature::Signature;

mod view_key;
pub use view_key::ViewKey;

use pyo3::prelude::*;

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

#[pyclass(frozen)]
pub struct Account {
    private_key: PrivateKey,
    view_key: ViewKey,
    address: Address,
}

#[pymethods]
impl Account {
    /// Generates a new account using a cryptographically secure random number generator
    #[new]
    fn new() -> Self {
        Self::from(PrivateKey::new())
    }

    /// Creates a new account from the given private key.
    #[staticmethod]
    fn from_private_key(private_key: PrivateKey) -> Self {
        let view_key = private_key.view_key();
        let address = private_key.address().unwrap();
        Self {
            private_key,
            view_key,
            address,
        }
    }

    /// Returns an account private key.
    fn private_key(&self) -> PrivateKey {
        self.private_key
    }

    /// Returns an account view key.
    fn view_key(&self) -> ViewKey {
        self.view_key.clone()
    }

    /// Returns an account address.
    fn address(&self) -> Address {
        self.address.clone()
    }

    /// Returns a signature for the given message (as bytes)
    fn sign(&self, message: &[u8]) -> anyhow::Result<Signature> {
        self.private_key.sign(message)
    }

    /// Verifies the signature of the given message.
    fn verify(&self, signature: &Signature, message: &[u8]) -> bool {
        signature.verify(&self.address, message)
    }

    /// Decrypts a record ciphertext with a view key
    fn decrypt(&self, record_ciphertext: &RecordCiphertext) -> anyhow::Result<RecordPlaintext> {
        record_ciphertext.decrypt(&self.view_key)
    }

    /// Determines whether the record belongs to the account.
    pub fn is_owner(&self, record_ciphertext: &RecordCiphertext) -> bool {
        record_ciphertext.is_owner(&self.view_key)
    }

    fn __str__(&self) -> String {
        self.address.to_string()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.private_key == other.private_key
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        // because it's enouth to hash the private key only we add a dummy string so that:
        // hash(Account) != hash(PrivateKey)
        "account".hash(&mut hasher);
        self.private_key.hash(&mut hasher);
        hasher.finish()
    }
}

impl From<PrivateKey> for Account {
    fn from(private_key: PrivateKey) -> Self {
        Self::from_private_key(private_key)
    }
}