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

use crate::{types::SignatureNative, Address, ComputeKey, PrivateKey, Scalar};

use pyo3::prelude::*;
use rand::{rngs::StdRng, SeedableRng};

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    ops::Deref,
    str::FromStr,
};

#[pyclass(frozen)]
#[derive(Clone)]
pub struct Signature(SignatureNative);

#[pymethods]
impl Signature {
    /// Returns the verifier challenge.
    fn challenge(&self) -> Scalar {
        self.0.challenge().into()
    }

    /// Returns the signer compute key.
    fn compute_key(&self) -> ComputeKey {
        self.0.compute_key().into()
    }

    /// Creates a signature from a string representation.
    #[staticmethod]
    fn from_string(s: &str) -> anyhow::Result<Self> {
        FromStr::from_str(s).map(Self)
    }

    /// Returns the prover response.
    fn response(&self) -> Scalar {
        self.0.response().into()
    }

    /// Returns a signature for the given message (as bytes) using the private key.
    #[staticmethod]
    pub fn sign(private_key: &PrivateKey, message: &[u8]) -> anyhow::Result<Self> {
        private_key
            .sign_bytes(message, &mut StdRng::from_entropy())
            .map(Self)
    }

    /// Verifies (challenge == challenge') && (address == address') where:
    ///     challenge' := HashToScalar(G^response pk_sig^challenge, pk_sig, pr_sig, address, message)
    pub fn verify(&self, address: &Address, message: &[u8]) -> bool {
        self.0.verify_bytes(address, message)
    }

    /// Returns a string representation of the signature.
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
    fn from(value: SignatureNative) -> Self {
        Self(value)
    }
}
