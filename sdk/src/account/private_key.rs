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
    types::{
        AddressNative, ComputeKeyNative, CurrentNetwork, PrivateKeyNative, ValueNative,
        ViewKeyNative,
    },
    Address, ComputeKey, Field, PrivateKeyCiphertext, Scalar, Signature, ViewKey,
};

use pyo3::prelude::*;
use rand::rngs::StdRng;
use snarkvm::prelude::ToFields;

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    ops::Deref,
    str::FromStr,
};

/// The Aleo private key type.
#[pyclass(frozen)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct PrivateKey(PrivateKeyNative);

#[pymethods]
impl PrivateKey {
    /// Generates a new private key using a cryptographically secure random number generator
    #[staticmethod]
    pub fn random() -> Self {
        PrivateKeyNative::new(&mut rand::make_rng::<StdRng>())
            .unwrap()
            .into()
    }

    /// Derives the account address from an account private key.
    #[getter]
    pub fn address(&self) -> anyhow::Result<Address> {
        AddressNative::try_from(&self.0).map(Into::into)
    }

    /// Derives the account compute key from an account private key.
    #[getter]
    fn compute_key(&self) -> ComputeKey {
        ComputeKeyNative::try_from(&self.0).unwrap().into()
    }

    /// Returns the account private key from an account seed.
    #[staticmethod]
    fn from_seed(seed: Field) -> anyhow::Result<Self> {
        PrivateKeyNative::try_from(seed.into()).map(Self)
    }

    /// Returns a private key from a 32-byte seed (unchecked; uses LE-mod-order reduction).
    #[staticmethod]
    pub fn from_seed_unchecked(seed: Vec<u8>) -> anyhow::Result<Self> {
        #[allow(unused_imports)]
        use snarkvm::prelude::*;
        let seed: [u8; 32] = seed
            .try_into()
            .map_err(|_| anyhow::anyhow!("seed must be exactly 32 bytes"))?;
        let inner = <CurrentNetwork as Environment>::Field::from_bytes_le_mod_order(&seed);
        let field = crate::types::FieldNative::new(inner);
        Ok(Self(PrivateKeyNative::try_from(field)?))
    }

    /// Reads in an account private key from a base58 string.
    #[staticmethod]
    fn from_string(private_key: &str) -> anyhow::Result<Self> {
        PrivateKeyNative::from_str(private_key).map(Self)
    }

    /// Returns the account seed.
    #[getter]
    pub fn seed(&self) -> Field {
        self.0.seed().into()
    }

    /// Returns a signature for the given message (as bytes) using the private key.
    pub fn sign(&self, message: &[u8]) -> anyhow::Result<Signature> {
        Signature::sign(self, message)
    }

    /// Signs a Value-domain message (any valid Aleo literal, struct, array, or record).
    pub fn sign_value(&self, message: &str) -> anyhow::Result<Signature> {
        let value = ValueNative::from_str(message)?;
        let fields = value.to_fields()?;
        Ok(self
            .0
            .sign(&fields, &mut rand::make_rng::<StdRng>())?
            .into())
    }

    /// Returns the little-endian byte representation of the private key.
    pub fn bytes(&self) -> anyhow::Result<Vec<u8>> {
        use snarkvm::prelude::ToBytes;
        self.0.to_bytes_le()
    }

    /// Recovers a private key from its little-endian byte representation.
    #[staticmethod]
    pub fn from_bytes(bytes: Vec<u8>) -> anyhow::Result<Self> {
        use snarkvm::prelude::FromBytes;
        Ok(Self(PrivateKeyNative::read_le(&bytes[..])?))
    }

    /// Generates a new random private key and returns it encrypted with the given secret.
    #[staticmethod]
    pub fn new_encrypted(secret: &str) -> anyhow::Result<PrivateKeyCiphertext> {
        let key = Self::random();
        PrivateKeyCiphertext::encrypt_private_key(&key, secret)
    }

    /// Encrypts this private key with the given secret and returns the ciphertext.
    #[allow(clippy::wrong_self_convention)]
    pub fn to_ciphertext(&self, secret: &str) -> anyhow::Result<PrivateKeyCiphertext> {
        PrivateKeyCiphertext::encrypt_private_key(self, secret)
    }

    /// Decrypts a PrivateKeyCiphertext with the given secret and returns the private key.
    #[staticmethod]
    pub fn from_private_key_ciphertext(
        ciphertext: &PrivateKeyCiphertext,
        secret: &str,
    ) -> anyhow::Result<Self> {
        ciphertext.decrypt_to_private_key(secret)
    }

    /// Returns the signature secret key.
    fn sk_sig(&self) -> Scalar {
        self.0.sk_sig().into()
    }

    /// Returns the signature randomizer.
    fn r_sig(&self) -> Scalar {
        self.0.r_sig().into()
    }

    /// Initializes a new account view key from an account private key.
    #[getter]
    pub fn view_key(&self) -> ViewKey {
        let view_key = ViewKeyNative::try_from(&self.0).unwrap();
        view_key.into()
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
    fn from(value: PrivateKey) -> Self {
        value.0
    }
}

impl From<PrivateKeyNative> for PrivateKey {
    fn from(value: PrivateKeyNative) -> Self {
        Self(value)
    }
}
