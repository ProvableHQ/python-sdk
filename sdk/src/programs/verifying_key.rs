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

use crate::types::VerifyingKeyNative;

use pyo3::prelude::*;
use sha2::Digest;
use snarkvm::prelude::{FromBytes, ToBytes};

use std::str::FromStr;

/// The Aleo verifying key type.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct VerifyingKey(VerifyingKeyNative);

#[pymethods]
impl VerifyingKey {
    /// Parses a veryifying key from string.
    #[staticmethod]
    fn from_string(s: &str) -> anyhow::Result<Self> {
        VerifyingKeyNative::from_str(s).map(Self)
    }

    /// Constructs a proving key from a byte array
    #[staticmethod]
    fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        VerifyingKeyNative::from_bytes_le(bytes).map(Self)
    }

    /// Returns the byte representation of a veryfying key
    fn bytes(&self) -> anyhow::Result<Vec<u8>> {
        self.0.to_bytes_le()
    }

    /// Returns the verifying key as a string.
    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __eq__(&self, other: &Self) -> bool {
        *self.0 == *other.0
    }

    #[classattr]
    const __hash__: Option<PyObject> = None;

    // ---- Task 7 additions ----

    /// Returns the SHA256 hex checksum of the verifying key bytes.
    fn checksum(&self) -> anyhow::Result<String> {
        let bytes = self.0.to_bytes_le()?;
        Ok(hex::encode(sha2::Sha256::digest(&bytes)))
    }

    /// Returns the number of constraints in the circuit.
    fn num_constraints(&self) -> u32 {
        self.0.circuit_info.num_constraints as u32
    }

    // ---- credits static getters ----

    #[staticmethod]
    fn bond_public_verifier() -> anyhow::Result<VerifyingKey> {
        #[cfg(not(feature = "testnet"))]
        let bytes = snarkvm::parameters::mainnet::BondPublicVerifier::load_bytes()?;
        #[cfg(feature = "testnet")]
        let bytes = snarkvm::parameters::testnet::BondPublicVerifier::load_bytes()?;
        VerifyingKeyNative::from_bytes_le(&bytes).map(Self)
    }

    #[staticmethod]
    fn bond_validator_verifier() -> anyhow::Result<VerifyingKey> {
        #[cfg(not(feature = "testnet"))]
        let bytes = snarkvm::parameters::mainnet::BondValidatorVerifier::load_bytes()?;
        #[cfg(feature = "testnet")]
        let bytes = snarkvm::parameters::testnet::BondValidatorVerifier::load_bytes()?;
        VerifyingKeyNative::from_bytes_le(&bytes).map(Self)
    }

    #[staticmethod]
    fn claim_unbond_public_verifier() -> anyhow::Result<VerifyingKey> {
        #[cfg(not(feature = "testnet"))]
        let bytes = snarkvm::parameters::mainnet::ClaimUnbondPublicVerifier::load_bytes()?;
        #[cfg(feature = "testnet")]
        let bytes = snarkvm::parameters::testnet::ClaimUnbondPublicVerifier::load_bytes()?;
        VerifyingKeyNative::from_bytes_le(&bytes).map(Self)
    }

    #[staticmethod]
    fn fee_private_verifier() -> anyhow::Result<VerifyingKey> {
        #[cfg(not(feature = "testnet"))]
        let bytes = snarkvm::parameters::mainnet::FeePrivateVerifier::load_bytes()?;
        #[cfg(feature = "testnet")]
        let bytes = snarkvm::parameters::testnet::FeePrivateVerifier::load_bytes()?;
        VerifyingKeyNative::from_bytes_le(&bytes).map(Self)
    }

    #[staticmethod]
    fn fee_public_verifier() -> anyhow::Result<VerifyingKey> {
        #[cfg(not(feature = "testnet"))]
        let bytes = snarkvm::parameters::mainnet::FeePublicVerifier::load_bytes()?;
        #[cfg(feature = "testnet")]
        let bytes = snarkvm::parameters::testnet::FeePublicVerifier::load_bytes()?;
        VerifyingKeyNative::from_bytes_le(&bytes).map(Self)
    }

    #[staticmethod]
    fn inclusion_verifier() -> anyhow::Result<VerifyingKey> {
        #[cfg(not(feature = "testnet"))]
        let bytes = snarkvm::parameters::mainnet::InclusionVerifier::load_bytes()?;
        #[cfg(feature = "testnet")]
        let bytes = snarkvm::parameters::testnet::InclusionVerifier::load_bytes()?;
        VerifyingKeyNative::from_bytes_le(&bytes).map(Self)
    }

    #[staticmethod]
    fn join_verifier() -> anyhow::Result<VerifyingKey> {
        #[cfg(not(feature = "testnet"))]
        let bytes = snarkvm::parameters::mainnet::JoinVerifier::load_bytes()?;
        #[cfg(feature = "testnet")]
        let bytes = snarkvm::parameters::testnet::JoinVerifier::load_bytes()?;
        VerifyingKeyNative::from_bytes_le(&bytes).map(Self)
    }

    #[staticmethod]
    fn set_validator_state_verifier() -> anyhow::Result<VerifyingKey> {
        #[cfg(not(feature = "testnet"))]
        let bytes = snarkvm::parameters::mainnet::SetValidatorStateVerifier::load_bytes()?;
        #[cfg(feature = "testnet")]
        let bytes = snarkvm::parameters::testnet::SetValidatorStateVerifier::load_bytes()?;
        VerifyingKeyNative::from_bytes_le(&bytes).map(Self)
    }

    #[staticmethod]
    fn split_verifier() -> anyhow::Result<VerifyingKey> {
        #[cfg(not(feature = "testnet"))]
        let bytes = snarkvm::parameters::mainnet::SplitVerifier::load_bytes()?;
        #[cfg(feature = "testnet")]
        let bytes = snarkvm::parameters::testnet::SplitVerifier::load_bytes()?;
        VerifyingKeyNative::from_bytes_le(&bytes).map(Self)
    }

    #[staticmethod]
    fn transfer_private_verifier() -> anyhow::Result<VerifyingKey> {
        #[cfg(not(feature = "testnet"))]
        let bytes = snarkvm::parameters::mainnet::TransferPrivateVerifier::load_bytes()?;
        #[cfg(feature = "testnet")]
        let bytes = snarkvm::parameters::testnet::TransferPrivateVerifier::load_bytes()?;
        VerifyingKeyNative::from_bytes_le(&bytes).map(Self)
    }

    #[staticmethod]
    fn transfer_private_to_public_verifier() -> anyhow::Result<VerifyingKey> {
        #[cfg(not(feature = "testnet"))]
        let bytes = snarkvm::parameters::mainnet::TransferPrivateToPublicVerifier::load_bytes()?;
        #[cfg(feature = "testnet")]
        let bytes = snarkvm::parameters::testnet::TransferPrivateToPublicVerifier::load_bytes()?;
        VerifyingKeyNative::from_bytes_le(&bytes).map(Self)
    }

    #[staticmethod]
    fn transfer_public_verifier() -> anyhow::Result<VerifyingKey> {
        #[cfg(not(feature = "testnet"))]
        let bytes = snarkvm::parameters::mainnet::TransferPublicVerifier::load_bytes()?;
        #[cfg(feature = "testnet")]
        let bytes = snarkvm::parameters::testnet::TransferPublicVerifier::load_bytes()?;
        VerifyingKeyNative::from_bytes_le(&bytes).map(Self)
    }

    #[staticmethod]
    fn transfer_public_as_signer_verifier() -> anyhow::Result<VerifyingKey> {
        #[cfg(not(feature = "testnet"))]
        let bytes = snarkvm::parameters::mainnet::TransferPublicAsSignerVerifier::load_bytes()?;
        #[cfg(feature = "testnet")]
        let bytes = snarkvm::parameters::testnet::TransferPublicAsSignerVerifier::load_bytes()?;
        VerifyingKeyNative::from_bytes_le(&bytes).map(Self)
    }

    #[staticmethod]
    fn transfer_public_to_private_verifier() -> anyhow::Result<VerifyingKey> {
        #[cfg(not(feature = "testnet"))]
        let bytes = snarkvm::parameters::mainnet::TransferPublicToPrivateVerifier::load_bytes()?;
        #[cfg(feature = "testnet")]
        let bytes = snarkvm::parameters::testnet::TransferPublicToPrivateVerifier::load_bytes()?;
        VerifyingKeyNative::from_bytes_le(&bytes).map(Self)
    }

    #[staticmethod]
    fn unbond_public_verifier() -> anyhow::Result<VerifyingKey> {
        #[cfg(not(feature = "testnet"))]
        let bytes = snarkvm::parameters::mainnet::UnbondPublicVerifier::load_bytes()?;
        #[cfg(feature = "testnet")]
        let bytes = snarkvm::parameters::testnet::UnbondPublicVerifier::load_bytes()?;
        VerifyingKeyNative::from_bytes_le(&bytes).map(Self)
    }

    // ---- is_*_verifier checkers ----

    fn is_bond_public_verifier(&self) -> bool {
        Self::bond_public_verifier()
            .map(|vk| *self.0 == *vk.0)
            .unwrap_or(false)
    }

    fn is_bond_validator_verifier(&self) -> bool {
        Self::bond_validator_verifier()
            .map(|vk| *self.0 == *vk.0)
            .unwrap_or(false)
    }

    fn is_claim_unbond_public_verifier(&self) -> bool {
        Self::claim_unbond_public_verifier()
            .map(|vk| *self.0 == *vk.0)
            .unwrap_or(false)
    }

    fn is_fee_private_verifier(&self) -> bool {
        Self::fee_private_verifier()
            .map(|vk| *self.0 == *vk.0)
            .unwrap_or(false)
    }

    fn is_fee_public_verifier(&self) -> bool {
        Self::fee_public_verifier()
            .map(|vk| *self.0 == *vk.0)
            .unwrap_or(false)
    }

    fn is_inclusion_verifier(&self) -> bool {
        Self::inclusion_verifier()
            .map(|vk| *self.0 == *vk.0)
            .unwrap_or(false)
    }

    fn is_join_verifier(&self) -> bool {
        Self::join_verifier()
            .map(|vk| *self.0 == *vk.0)
            .unwrap_or(false)
    }

    fn is_set_validator_state_verifier(&self) -> bool {
        Self::set_validator_state_verifier()
            .map(|vk| *self.0 == *vk.0)
            .unwrap_or(false)
    }

    fn is_split_verifier(&self) -> bool {
        Self::split_verifier()
            .map(|vk| *self.0 == *vk.0)
            .unwrap_or(false)
    }

    fn is_transfer_private_verifier(&self) -> bool {
        Self::transfer_private_verifier()
            .map(|vk| *self.0 == *vk.0)
            .unwrap_or(false)
    }

    fn is_transfer_private_to_public_verifier(&self) -> bool {
        Self::transfer_private_to_public_verifier()
            .map(|vk| *self.0 == *vk.0)
            .unwrap_or(false)
    }

    fn is_transfer_public_verifier(&self) -> bool {
        Self::transfer_public_verifier()
            .map(|vk| *self.0 == *vk.0)
            .unwrap_or(false)
    }

    fn is_transfer_public_as_signer_verifier(&self) -> bool {
        Self::transfer_public_as_signer_verifier()
            .map(|vk| *self.0 == *vk.0)
            .unwrap_or(false)
    }

    fn is_transfer_public_to_private_verifier(&self) -> bool {
        Self::transfer_public_to_private_verifier()
            .map(|vk| *self.0 == *vk.0)
            .unwrap_or(false)
    }

    fn is_unbond_public_verifier(&self) -> bool {
        Self::unbond_public_verifier()
            .map(|vk| *self.0 == *vk.0)
            .unwrap_or(false)
    }
}

impl From<VerifyingKeyNative> for VerifyingKey {
    fn from(value: VerifyingKeyNative) -> Self {
        Self(value)
    }
}

impl From<VerifyingKey> for VerifyingKeyNative {
    fn from(value: VerifyingKey) -> Self {
        value.0
    }
}
