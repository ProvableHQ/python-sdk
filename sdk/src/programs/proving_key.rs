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

use crate::types::ProvingKeyNative;

use pyo3::prelude::*;
use sha2::Digest;
use snarkvm::prelude::{FromBytes, ToBytes};

use std::str::FromStr;

/// Extract the `prover_checksum` field from an embedded METADATA JSON string.
fn prover_checksum_from_meta(metadata_json: &'static str) -> String {
    let meta: serde_json::Value =
        serde_json::from_str(metadata_json).expect("Metadata was not well-formatted");
    meta["prover_checksum"]
        .as_str()
        .expect("Failed to parse prover_checksum")
        .to_string()
}

/// The Aleo proving key type.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct ProvingKey(ProvingKeyNative);

#[pymethods]
impl ProvingKey {
    /// Parses a proving key from string.
    #[staticmethod]
    fn from_string(s: &str) -> anyhow::Result<Self> {
        ProvingKeyNative::from_str(s).map(Self)
    }

    /// Constructs a proving key from a byte array
    #[staticmethod]
    fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        ProvingKeyNative::from_bytes_le(bytes).map(Self)
    }

    /// Returns the byte representation of a proving key
    fn bytes(&self) -> anyhow::Result<Vec<u8>> {
        self.0.to_bytes_le()
    }

    /// Returns the proving key as a string.
    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __eq__(&self, other: &Self) -> bool {
        *self.0 == *other.0
    }

    #[classattr]
    const __hash__: Option<PyObject> = None;

    // ---- Task 8 additions ----

    /// Returns the SHA256 hex checksum of the proving key bytes.
    fn checksum(&self) -> anyhow::Result<String> {
        let bytes = self.0.to_bytes_le()?;
        Ok(hex::encode(sha2::Sha256::digest(&bytes)))
    }

    // ---- is_*_prover checkers ----

    fn is_bond_public_prover(&self) -> bool {
        #[cfg(not(feature = "testnet"))]
        {
            self.checksum()
                .map(|cs| {
                    cs == prover_checksum_from_meta(
                        snarkvm::parameters::mainnet::BondPublicProver::METADATA,
                    )
                })
                .unwrap_or(false)
        }
        #[cfg(feature = "testnet")]
        {
            self.checksum()
                .map(|cs| {
                    cs == prover_checksum_from_meta(
                        snarkvm::parameters::testnet::BondPublicProver::METADATA,
                    )
                })
                .unwrap_or(false)
        }
    }

    fn is_bond_validator_prover(&self) -> bool {
        #[cfg(not(feature = "testnet"))]
        {
            self.checksum()
                .map(|cs| {
                    cs == prover_checksum_from_meta(
                        snarkvm::parameters::mainnet::BondValidatorProver::METADATA,
                    )
                })
                .unwrap_or(false)
        }
        #[cfg(feature = "testnet")]
        {
            self.checksum()
                .map(|cs| {
                    cs == prover_checksum_from_meta(
                        snarkvm::parameters::testnet::BondValidatorProver::METADATA,
                    )
                })
                .unwrap_or(false)
        }
    }

    fn is_claim_unbond_public_prover(&self) -> bool {
        #[cfg(not(feature = "testnet"))]
        {
            self.checksum()
                .map(|cs| {
                    cs == prover_checksum_from_meta(
                        snarkvm::parameters::mainnet::ClaimUnbondPublicProver::METADATA,
                    )
                })
                .unwrap_or(false)
        }
        #[cfg(feature = "testnet")]
        {
            self.checksum()
                .map(|cs| {
                    cs == prover_checksum_from_meta(
                        snarkvm::parameters::testnet::ClaimUnbondPublicProver::METADATA,
                    )
                })
                .unwrap_or(false)
        }
    }

    fn is_fee_private_prover(&self) -> bool {
        #[cfg(not(feature = "testnet"))]
        {
            self.checksum()
                .map(|cs| {
                    cs == prover_checksum_from_meta(
                        snarkvm::parameters::mainnet::FeePrivateProver::METADATA,
                    )
                })
                .unwrap_or(false)
        }
        #[cfg(feature = "testnet")]
        {
            self.checksum()
                .map(|cs| {
                    cs == prover_checksum_from_meta(
                        snarkvm::parameters::testnet::FeePrivateProver::METADATA,
                    )
                })
                .unwrap_or(false)
        }
    }

    fn is_fee_public_prover(&self) -> bool {
        #[cfg(not(feature = "testnet"))]
        {
            self.checksum()
                .map(|cs| {
                    cs == prover_checksum_from_meta(
                        snarkvm::parameters::mainnet::FeePublicProver::METADATA,
                    )
                })
                .unwrap_or(false)
        }
        #[cfg(feature = "testnet")]
        {
            self.checksum()
                .map(|cs| {
                    cs == prover_checksum_from_meta(
                        snarkvm::parameters::testnet::FeePublicProver::METADATA,
                    )
                })
                .unwrap_or(false)
        }
    }

    fn is_inclusion_prover(&self) -> bool {
        #[cfg(not(feature = "testnet"))]
        {
            self.checksum()
                .map(|cs| {
                    cs == prover_checksum_from_meta(
                        snarkvm::parameters::mainnet::InclusionProver::METADATA,
                    )
                })
                .unwrap_or(false)
        }
        #[cfg(feature = "testnet")]
        {
            self.checksum()
                .map(|cs| {
                    cs == prover_checksum_from_meta(
                        snarkvm::parameters::testnet::InclusionProver::METADATA,
                    )
                })
                .unwrap_or(false)
        }
    }

    fn is_join_prover(&self) -> bool {
        #[cfg(not(feature = "testnet"))]
        {
            self.checksum()
                .map(|cs| {
                    cs == prover_checksum_from_meta(
                        snarkvm::parameters::mainnet::JoinProver::METADATA,
                    )
                })
                .unwrap_or(false)
        }
        #[cfg(feature = "testnet")]
        {
            self.checksum()
                .map(|cs| {
                    cs == prover_checksum_from_meta(
                        snarkvm::parameters::testnet::JoinProver::METADATA,
                    )
                })
                .unwrap_or(false)
        }
    }

    fn is_set_validator_state_prover(&self) -> bool {
        #[cfg(not(feature = "testnet"))]
        {
            self.checksum()
                .map(|cs| {
                    cs == prover_checksum_from_meta(
                        snarkvm::parameters::mainnet::SetValidatorStateProver::METADATA,
                    )
                })
                .unwrap_or(false)
        }
        #[cfg(feature = "testnet")]
        {
            self.checksum()
                .map(|cs| {
                    cs == prover_checksum_from_meta(
                        snarkvm::parameters::testnet::SetValidatorStateProver::METADATA,
                    )
                })
                .unwrap_or(false)
        }
    }

    fn is_split_prover(&self) -> bool {
        #[cfg(not(feature = "testnet"))]
        {
            self.checksum()
                .map(|cs| {
                    cs == prover_checksum_from_meta(
                        snarkvm::parameters::mainnet::SplitProver::METADATA,
                    )
                })
                .unwrap_or(false)
        }
        #[cfg(feature = "testnet")]
        {
            self.checksum()
                .map(|cs| {
                    cs == prover_checksum_from_meta(
                        snarkvm::parameters::testnet::SplitProver::METADATA,
                    )
                })
                .unwrap_or(false)
        }
    }

    fn is_transfer_private_prover(&self) -> bool {
        #[cfg(not(feature = "testnet"))]
        {
            self.checksum()
                .map(|cs| {
                    cs == prover_checksum_from_meta(
                        snarkvm::parameters::mainnet::TransferPrivateProver::METADATA,
                    )
                })
                .unwrap_or(false)
        }
        #[cfg(feature = "testnet")]
        {
            self.checksum()
                .map(|cs| {
                    cs == prover_checksum_from_meta(
                        snarkvm::parameters::testnet::TransferPrivateProver::METADATA,
                    )
                })
                .unwrap_or(false)
        }
    }

    fn is_transfer_private_to_public_prover(&self) -> bool {
        #[cfg(not(feature = "testnet"))]
        {
            self.checksum()
                .map(|cs| {
                    cs == prover_checksum_from_meta(
                        snarkvm::parameters::mainnet::TransferPrivateToPublicProver::METADATA,
                    )
                })
                .unwrap_or(false)
        }
        #[cfg(feature = "testnet")]
        {
            self.checksum()
                .map(|cs| {
                    cs == prover_checksum_from_meta(
                        snarkvm::parameters::testnet::TransferPrivateToPublicProver::METADATA,
                    )
                })
                .unwrap_or(false)
        }
    }

    fn is_transfer_public_prover(&self) -> bool {
        #[cfg(not(feature = "testnet"))]
        {
            self.checksum()
                .map(|cs| {
                    cs == prover_checksum_from_meta(
                        snarkvm::parameters::mainnet::TransferPublicProver::METADATA,
                    )
                })
                .unwrap_or(false)
        }
        #[cfg(feature = "testnet")]
        {
            self.checksum()
                .map(|cs| {
                    cs == prover_checksum_from_meta(
                        snarkvm::parameters::testnet::TransferPublicProver::METADATA,
                    )
                })
                .unwrap_or(false)
        }
    }

    fn is_transfer_public_as_signer_prover(&self) -> bool {
        #[cfg(not(feature = "testnet"))]
        {
            self.checksum()
                .map(|cs| {
                    cs == prover_checksum_from_meta(
                        snarkvm::parameters::mainnet::TransferPublicAsSignerProver::METADATA,
                    )
                })
                .unwrap_or(false)
        }
        #[cfg(feature = "testnet")]
        {
            self.checksum()
                .map(|cs| {
                    cs == prover_checksum_from_meta(
                        snarkvm::parameters::testnet::TransferPublicAsSignerProver::METADATA,
                    )
                })
                .unwrap_or(false)
        }
    }

    fn is_transfer_public_to_private_prover(&self) -> bool {
        #[cfg(not(feature = "testnet"))]
        {
            self.checksum()
                .map(|cs| {
                    cs == prover_checksum_from_meta(
                        snarkvm::parameters::mainnet::TransferPublicToPrivateProver::METADATA,
                    )
                })
                .unwrap_or(false)
        }
        #[cfg(feature = "testnet")]
        {
            self.checksum()
                .map(|cs| {
                    cs == prover_checksum_from_meta(
                        snarkvm::parameters::testnet::TransferPublicToPrivateProver::METADATA,
                    )
                })
                .unwrap_or(false)
        }
    }

    fn is_unbond_public_prover(&self) -> bool {
        #[cfg(not(feature = "testnet"))]
        {
            self.checksum()
                .map(|cs| {
                    cs == prover_checksum_from_meta(
                        snarkvm::parameters::mainnet::UnbondPublicProver::METADATA,
                    )
                })
                .unwrap_or(false)
        }
        #[cfg(feature = "testnet")]
        {
            self.checksum()
                .map(|cs| {
                    cs == prover_checksum_from_meta(
                        snarkvm::parameters::testnet::UnbondPublicProver::METADATA,
                    )
                })
                .unwrap_or(false)
        }
    }
}

impl From<ProvingKeyNative> for ProvingKey {
    fn from(value: ProvingKeyNative) -> Self {
        Self(value)
    }
}

impl From<ProvingKey> for ProvingKeyNative {
    fn from(value: ProvingKey) -> Self {
        value.0
    }
}
