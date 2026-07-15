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
    types::{LiteralNative, SignatureNative, ValueNative},
    Address, ComputeKey, Field, Plaintext, PrivateKey, Scalar,
};

use pyo3::prelude::*;
use rand::rngs::StdRng;
use snarkvm::prelude::{FromBits, FromBytes, ToBits, ToBytes, ToFields};

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    ops::Deref,
    str::FromStr,
};

/// The Aleo signature type.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct Signature(SignatureNative);

#[pymethods]
impl Signature {
    /// Returns the verifier challenge.
    #[getter]
    fn challenge(&self) -> Scalar {
        self.0.challenge().into()
    }

    /// Returns the signer compute key.
    #[getter]
    fn compute_key(&self) -> ComputeKey {
        self.0.compute_key().into()
    }

    /// Creates a signature from a string representation.
    #[staticmethod]
    fn from_string(s: &str) -> anyhow::Result<Self> {
        SignatureNative::from_str(s).map(Self)
    }

    /// Returns the prover response.
    #[getter]
    fn response(&self) -> Scalar {
        self.0.response().into()
    }

    /// Returns a signature for the given message (as bytes) using the private key.
    #[staticmethod]
    pub fn sign(private_key: &PrivateKey, message: &[u8]) -> anyhow::Result<Self> {
        private_key
            .sign_bytes(message, &mut rand::make_rng::<StdRng>())
            .map(Self)
    }

    /// Verifies (challenge == challenge') && (address == address') where:
    ///     challenge' := HashToScalar(G^response pk_sig^challenge, pk_sig, pr_sig, address, message)
    pub fn verify(&self, address: &Address, message: &[u8]) -> bool {
        self.0.verify_bytes(address, message)
    }

    /// Returns the signer address derived from the compute key embedded in this signature.
    pub fn to_address(&self) -> Address {
        self.0.to_address().into()
    }

    /// Returns the field elements encoding this signature.
    pub fn to_fields(&self) -> anyhow::Result<Vec<Field>> {
        Ok(self.0.to_fields()?.into_iter().map(Into::into).collect())
    }

    /// Returns the little-endian bit representation of the signature.
    pub fn to_bits_le(&self) -> Vec<bool> {
        self.0.to_bits_le()
    }

    /// Returns the little-endian byte representation of the signature.
    pub fn bytes(&self) -> anyhow::Result<Vec<u8>> {
        self.0.to_bytes_le()
    }

    /// Recovers a signature from its little-endian byte representation.
    #[staticmethod]
    pub fn from_bytes(bytes: Vec<u8>) -> anyhow::Result<Self> {
        Ok(Self(SignatureNative::read_le(&bytes[..])?))
    }

    /// Recovers a signature from little-endian bits.
    #[staticmethod]
    pub fn from_bits_le(bits: Vec<bool>) -> anyhow::Result<Self> {
        SignatureNative::from_bits_le(&bits).map(Self)
    }

    /// Returns the signature wrapped as a Plaintext::Literal(Signature).
    pub fn to_plaintext(&self) -> Plaintext {
        use crate::types::PlaintextNative;
        Plaintext::from(PlaintextNative::from(LiteralNative::Signature(Box::new(
            self.0,
        ))))
    }

    /// Signs a Value-domain message (any valid Aleo literal, struct, array, or record).
    #[staticmethod]
    pub fn sign_value(private_key: &PrivateKey, message: &str) -> anyhow::Result<Self> {
        let value = ValueNative::from_str(message)?;
        let fields = value.to_fields()?;
        Ok(Self(
            (**private_key).sign(&fields, &mut rand::make_rng::<StdRng>())?,
        ))
    }

    /// Verifies a Value-domain signature against an address.
    pub fn verify_value(&self, address: &Address, message: &str) -> anyhow::Result<bool> {
        let value = ValueNative::from_str(message)?;
        let fields = value.to_fields()?;
        Ok(self.0.verify(address, &fields))
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
