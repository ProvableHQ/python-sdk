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
    account::{Address, ComputeKey, PrivateKey},
    types::SignatureNative,
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
pub struct Signature(SignatureNative);

#[pymethods]
impl Signature {
    /// Returns the verifier challenge.
    #[getter]
    fn challenge(&self) -> String {
        self.0.challenge().to_string()
    }

    /// Returns the signer compute key.
    #[getter]
    fn compute_key(&self) -> ComputeKey {
        ComputeKey::from(self.0.compute_key())
    }

    /// Creates a signature from a string representation.
    #[staticmethod]
    fn from_string(s: &str) -> anyhow::Result<Self> {
        let signature = FromStr::from_str(s)?;
        Ok(Self(signature))
    }

    /// Returns the prover response.
    #[getter]
    fn response(&self) -> String {
        self.0.response().to_string()
    }

    /// Returns a signature for the given message (as bytes) using the private key.
    #[staticmethod]
    pub fn sign(private_key: &PrivateKey, message: &[u8]) -> anyhow::Result<Self> {
        let signature = private_key.sign_bytes(message, &mut StdRng::from_entropy())?;
        Ok(Self(signature))
    }

    /// Returns a string representation of the signature.
    #[allow(clippy::inherent_to_string)]
    fn to_string(&self) -> String {
        self.0.to_string()
    }

    /// Verifies (challenge == challenge') && (address == address') where:
    ///     challenge' := HashToScalar(G^response pk_sig^challenge, pk_sig, pr_sig, address, message)
    pub fn verify(&self, address: &Address, message: &[u8]) -> bool {
        self.0.verify_bytes(address, message)
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

impl Deref for Signature {
    type Target = SignatureNative;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<SignatureNative> for Signature {
    fn from(signature: SignatureNative) -> Self {
        Self(signature)
    }
}
