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
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct PrivateKey(PrivateKeyNative);

#[pymethods]
impl PrivateKey {
    /// Generates a new private key using a cryptographically secure random number generator
    #[allow(clippy::new_without_default)]
    #[new]
    pub fn new() -> Self {
        Self(PrivateKeyNative::new(&mut StdRng::from_entropy()).unwrap())
    }

    /// Derives the account address from an account private key.
    pub fn address(&self) -> anyhow::Result<Address> {
        Ok(Address::from(AddressNative::try_from(&self.0)?))
    }

    /// Derives the account compute key from an account private key.
    fn compute_key(&self) -> ComputeKey {
        let compute_key = ComputeKeyNative::try_from(&self.0).unwrap();
        ComputeKey::from(compute_key)
    }

    /// Reads in an account private key from a base58 string.
    #[staticmethod]
    fn from_string(private_key: &str) -> anyhow::Result<Self> {
        Ok(Self(PrivateKeyNative::from_str(private_key)?))
    }

    /// Returns the account seed.
    fn seed(&self) -> String {
        self.0.seed().to_string()
    }

    /// Returns a signature for the given message (as bytes) using the private key.
    pub fn sign(&self, message: &[u8]) -> anyhow::Result<Signature> {
        Signature::sign(self, message)
    }

    /// Returns the signature secret key.
    fn sk_sig(&self) -> String {
        self.0.sk_sig().to_string()
    }

    /// Returns the signature randomizer.
    fn r_sig(&self) -> String {
        self.0.r_sig().to_string()
    }

    /// Initializes a new account view key from an account private key.
    pub fn view_key(&self) -> ViewKey {
        let view_key = ViewKeyNative::try_from(&self.0).unwrap();
        ViewKey::from(view_key)
    }

    /// Returns the private key as a base58 string.
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

impl Deref for PrivateKey {
    type Target = PrivateKeyNative;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<PrivateKey> for PrivateKeyNative {
    fn from(private_key: PrivateKey) -> Self {
        private_key.0
    }
}

impl From<PrivateKeyNative> for PrivateKey {
    fn from(private_key: PrivateKeyNative) -> Self {
        Self(private_key)
    }
}
