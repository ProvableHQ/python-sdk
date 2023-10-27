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
    account::{Address, ComputeKey, Signature, ViewKey},
    types::{AddressNative, ComputeKeyNative, PrivateKeyNative, ViewKeyNative},
};

use pyo3::prelude::*;
use rand::{rngs::StdRng, SeedableRng};

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    ops::Deref,
    str::FromStr,
};

#[pyclass(frozen)]
#[derive(Clone, PartialEq, Eq)]
pub struct PrivateKey(PrivateKeyNative);

impl Deref for PrivateKey {
    type Target = PrivateKeyNative;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[pymethods]
impl PrivateKey {
    /// Generates a new private key using a cryptographically secure random number generator
    #[new]
    pub fn random() -> Self {
        let key = PrivateKeyNative::new(&mut StdRng::from_entropy()).unwrap();
        Self(key)
    }

    /// Reads in an account private key from a base58 string.
    #[staticmethod]
    fn from_string(s: &str) -> anyhow::Result<Self> {
        let private_key = FromStr::from_str(s)?;
        Ok(Self(private_key))
    }

    /// Returns a signature for the given message (as bytes) using the private key.
    pub fn sign(&self, message: &[u8]) -> anyhow::Result<Signature> {
        Signature::sign(self, message)
    }

    /// Returns the account seed.
    fn seed(&self) -> String {
        self.0.seed().to_string()
    }

    /// Returns the signature secret key.
    fn sk_sig(&self) -> String {
        self.0.sk_sig().to_string()
    }

    /// Returns the signature randomizer.
    fn r_sig(&self) -> String {
        self.0.r_sig().to_string()
    }

    /// Derives the account address from an account private key.
    pub fn address(&self) -> Address {
        let address = AddressNative::try_from(&self.0).unwrap();
        Address::from_native(address)
    }

    /// Derives the account compute key from an account private key.
    fn compute_key(&self) -> ComputeKey {
        let compute_key = ComputeKeyNative::try_from(&self.0).unwrap();
        ComputeKey::from_native(compute_key)
    }

    /// Initializes a new account view key from an account private key.
    pub fn view_key(&self) -> ViewKey {
        let view_key = ViewKeyNative::try_from(&self.0).unwrap();
        ViewKey::from_native(view_key)
    }

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
