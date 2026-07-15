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
        CiphertextNative, CurrentNetwork, FieldNative, IdentifierNative, LiteralNative,
        PlaintextNative, PrivateKeyNative,
    },
    PrivateKey,
};

use pyo3::prelude::*;
use rand::rngs::StdRng;
use snarkvm::prelude::{Network, Uniform};

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    ops::Deref,
    str::FromStr,
    sync::OnceLock,
};

/// Private key encrypted into ciphertext using a secret.
#[pyclass(frozen, from_py_object)]
#[derive(Clone)]
pub struct PrivateKeyCiphertext(CiphertextNative);

impl PrivateKeyCiphertext {
    /// Internal: encrypts a field element with domain separation.
    fn encrypt_field(
        field: &FieldNative,
        secret: &str,
        domain: &str,
    ) -> anyhow::Result<CiphertextNative> {
        let domain_f = FieldNative::new_domain_separator(domain);
        let secret_f = FieldNative::new_domain_separator(secret);
        let nonce = <FieldNative as Uniform>::rand(&mut rand::make_rng::<StdRng>());
        let blinding = CurrentNetwork::hash_psd2(&[domain_f, nonce, secret_f])?;
        let key = blinding * field;
        let plaintext = PlaintextNative::Struct(
            indexmap::IndexMap::from_iter(vec![
                (
                    IdentifierNative::from_str("key")?,
                    PlaintextNative::Literal(LiteralNative::Field(key), OnceLock::new()),
                ),
                (
                    IdentifierNative::from_str("nonce")?,
                    PlaintextNative::Literal(LiteralNative::Field(nonce), OnceLock::new()),
                ),
            ]),
            OnceLock::new(),
        );
        plaintext.encrypt_symmetric(secret_f)
    }

    /// Internal: decrypts a field element from ciphertext.
    fn decrypt_field(
        ciphertext: &CiphertextNative,
        secret: &str,
        domain: &str,
    ) -> anyhow::Result<FieldNative> {
        let domain_f = FieldNative::new_domain_separator(domain);
        let secret_f = FieldNative::new_domain_separator(secret);
        let decrypted = ciphertext.decrypt_symmetric(secret_f)?;
        let recovered_key = Self::extract_field(&decrypted, "key")?;
        let recovered_nonce = Self::extract_field(&decrypted, "nonce")?;
        let recovered_blinding = CurrentNetwork::hash_psd2(&[domain_f, recovered_nonce, secret_f])?;
        Ok(recovered_key / recovered_blinding)
    }

    /// Internal: extracts a Field literal from a struct plaintext member.
    fn extract_field(plaintext: &PlaintextNative, identifier: &str) -> anyhow::Result<FieldNative> {
        let id = IdentifierNative::from_str(identifier)?;
        let value = plaintext.find(&[id])?;
        match value {
            PlaintextNative::Literal(LiteralNative::Field(f), _) => Ok(f),
            _ => anyhow::bail!("expected field literal for '{identifier}'"),
        }
    }
}

#[pymethods]
impl PrivateKeyCiphertext {
    /// Encrypts a private key with a secret string.
    #[staticmethod]
    pub fn encrypt_private_key(private_key: &PrivateKey, secret: &str) -> anyhow::Result<Self> {
        let seed = (*private_key).seed();
        Ok(Self(Self::encrypt_field(&seed, secret, "private_key")?))
    }

    /// Decrypts self into a private key using the given secret.
    pub fn decrypt_to_private_key(&self, secret: &str) -> anyhow::Result<PrivateKey> {
        let seed = Self::decrypt_field(&self.0, secret, "private_key")?;
        Ok(PrivateKeyNative::try_from(seed)?.into())
    }

    /// Parses a ciphertext from its string representation.
    #[staticmethod]
    pub fn from_string(s: &str) -> anyhow::Result<Self> {
        CiphertextNative::from_str(s).map(Self)
    }

    /// Returns the string representation of the ciphertext.
    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.0.to_string().hash(&mut hasher);
        hasher.finish()
    }
}

impl Deref for PrivateKeyCiphertext {
    type Target = CiphertextNative;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<CiphertextNative> for PrivateKeyCiphertext {
    fn from(value: CiphertextNative) -> Self {
        Self(value)
    }
}
