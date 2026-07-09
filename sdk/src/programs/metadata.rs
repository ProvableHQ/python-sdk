// Copyright (C) 2019-2023 Aleo Systems Inc.
// This file is part of the Aleo SDK library.
//
// Licensed under GPL-3.0-or-later.

use pyo3::prelude::*;

#[cfg(not(feature = "testnet"))]
const BASE_URL: &str = "https://parameters.provable.com/mainnet/";
#[cfg(feature = "testnet")]
const BASE_URL: &str = "https://parameters.provable.com/testnet/";

fn prover_checksum(metadata_json: &'static str) -> String {
    let meta: serde_json::Value =
        serde_json::from_str(metadata_json).expect("Metadata was not well-formatted");
    meta["prover_checksum"]
        .as_str()
        .expect("Failed to parse prover checksum")
        .to_string()
}

fn verifier_checksum(metadata_json: &'static str) -> String {
    let meta: serde_json::Value =
        serde_json::from_str(metadata_json).expect("Metadata was not well-formatted");
    meta["verifier_checksum"]
        .as_str()
        .expect("Failed to parse verifier checksum")
        .to_string()
}

fn make_metadata(
    name: &str,
    verifying_key_js_name: &str,
    locator: &str,
    metadata_json: &'static str,
) -> Metadata {
    let pc = prover_checksum(metadata_json);
    let vc = verifier_checksum(metadata_json);
    Metadata {
        name: name.to_string(),
        locator: locator.to_string(),
        prover: format!("{}{}.prover.{}", BASE_URL, name, &pc[..7]),
        verifier: format!("{}.verifier.{}", name, &vc[..7]),
        verifying_key: verifying_key_js_name.to_string(),
    }
}

/// Metadata for an Aleo credits function's proving and verifying keys.
#[pyclass]
#[derive(Clone)]
pub struct Metadata {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub locator: String,
    #[pyo3(get)]
    pub prover: String,
    #[pyo3(get)]
    pub verifier: String,
    #[pyo3(get)]
    pub verifying_key: String,
}

#[pymethods]
impl Metadata {
    /// Returns the base URL for parameter downloads.
    #[staticmethod]
    fn base_url() -> String {
        BASE_URL.to_string()
    }

    #[staticmethod]
    fn bond_public() -> Metadata {
        #[cfg(not(feature = "testnet"))]
        {
            use snarkvm::parameters::mainnet::BondPublicProver;
            make_metadata(
                "bond_public",
                "bondPublicVerifier",
                "credits.aleo/bond_public",
                BondPublicProver::METADATA,
            )
        }
        #[cfg(feature = "testnet")]
        {
            use snarkvm::parameters::testnet::BondPublicProver;
            make_metadata(
                "bond_public",
                "bondPublicVerifier",
                "credits.aleo/bond_public",
                BondPublicProver::METADATA,
            )
        }
    }

    #[staticmethod]
    fn bond_validator() -> Metadata {
        #[cfg(not(feature = "testnet"))]
        {
            use snarkvm::parameters::mainnet::BondValidatorProver;
            make_metadata(
                "bond_validator",
                "bondValidatorVerifier",
                "credits.aleo/bond_validator",
                BondValidatorProver::METADATA,
            )
        }
        #[cfg(feature = "testnet")]
        {
            use snarkvm::parameters::testnet::BondValidatorProver;
            make_metadata(
                "bond_validator",
                "bondValidatorVerifier",
                "credits.aleo/bond_validator",
                BondValidatorProver::METADATA,
            )
        }
    }

    #[staticmethod]
    fn claim_unbond_public() -> Metadata {
        #[cfg(not(feature = "testnet"))]
        {
            use snarkvm::parameters::mainnet::ClaimUnbondPublicProver;
            make_metadata(
                "claim_unbond_public",
                "claimUnbondPublicVerifier",
                "credits.aleo/claim_unbond_public",
                ClaimUnbondPublicProver::METADATA,
            )
        }
        #[cfg(feature = "testnet")]
        {
            use snarkvm::parameters::testnet::ClaimUnbondPublicProver;
            make_metadata(
                "claim_unbond_public",
                "claimUnbondPublicVerifier",
                "credits.aleo/claim_unbond_public",
                ClaimUnbondPublicProver::METADATA,
            )
        }
    }

    #[staticmethod]
    fn fee_private() -> Metadata {
        #[cfg(not(feature = "testnet"))]
        {
            use snarkvm::parameters::mainnet::FeePrivateProver;
            make_metadata(
                "fee_private",
                "feePrivateVerifier",
                "credits.aleo/fee_private",
                FeePrivateProver::METADATA,
            )
        }
        #[cfg(feature = "testnet")]
        {
            use snarkvm::parameters::testnet::FeePrivateProver;
            make_metadata(
                "fee_private",
                "feePrivateVerifier",
                "credits.aleo/fee_private",
                FeePrivateProver::METADATA,
            )
        }
    }

    #[staticmethod]
    fn fee_public() -> Metadata {
        #[cfg(not(feature = "testnet"))]
        {
            use snarkvm::parameters::mainnet::FeePublicProver;
            make_metadata(
                "fee_public",
                "feePublicVerifier",
                "credits.aleo/fee_public",
                FeePublicProver::METADATA,
            )
        }
        #[cfg(feature = "testnet")]
        {
            use snarkvm::parameters::testnet::FeePublicProver;
            make_metadata(
                "fee_public",
                "feePublicVerifier",
                "credits.aleo/fee_public",
                FeePublicProver::METADATA,
            )
        }
    }

    #[staticmethod]
    fn inclusion() -> Metadata {
        #[cfg(not(feature = "testnet"))]
        {
            use snarkvm::parameters::mainnet::InclusionProver;
            make_metadata(
                "inclusion",
                "inclusionVerifier",
                "inclusion",
                InclusionProver::METADATA,
            )
        }
        #[cfg(feature = "testnet")]
        {
            use snarkvm::parameters::testnet::InclusionProver;
            make_metadata(
                "inclusion",
                "inclusionVerifier",
                "inclusion",
                InclusionProver::METADATA,
            )
        }
    }

    #[staticmethod]
    fn join() -> Metadata {
        #[cfg(not(feature = "testnet"))]
        {
            use snarkvm::parameters::mainnet::JoinProver;
            make_metadata(
                "join",
                "joinVerifier",
                "credits.aleo/join",
                JoinProver::METADATA,
            )
        }
        #[cfg(feature = "testnet")]
        {
            use snarkvm::parameters::testnet::JoinProver;
            make_metadata(
                "join",
                "joinVerifier",
                "credits.aleo/join",
                JoinProver::METADATA,
            )
        }
    }

    #[staticmethod]
    fn set_validator_state() -> Metadata {
        #[cfg(not(feature = "testnet"))]
        {
            use snarkvm::parameters::mainnet::SetValidatorStateProver;
            make_metadata(
                "set_validator_state",
                "setValidatorStateVerifier",
                "credits.aleo/set_validator_state",
                SetValidatorStateProver::METADATA,
            )
        }
        #[cfg(feature = "testnet")]
        {
            use snarkvm::parameters::testnet::SetValidatorStateProver;
            make_metadata(
                "set_validator_state",
                "setValidatorStateVerifier",
                "credits.aleo/set_validator_state",
                SetValidatorStateProver::METADATA,
            )
        }
    }

    #[staticmethod]
    fn split() -> Metadata {
        #[cfg(not(feature = "testnet"))]
        {
            use snarkvm::parameters::mainnet::SplitProver;
            make_metadata(
                "split",
                "splitVerifier",
                "credits.aleo/split",
                SplitProver::METADATA,
            )
        }
        #[cfg(feature = "testnet")]
        {
            use snarkvm::parameters::testnet::SplitProver;
            make_metadata(
                "split",
                "splitVerifier",
                "credits.aleo/split",
                SplitProver::METADATA,
            )
        }
    }

    #[staticmethod]
    fn transfer_private() -> Metadata {
        #[cfg(not(feature = "testnet"))]
        {
            use snarkvm::parameters::mainnet::TransferPrivateProver;
            make_metadata(
                "transfer_private",
                "transferPrivateVerifier",
                "credits.aleo/transfer_private",
                TransferPrivateProver::METADATA,
            )
        }
        #[cfg(feature = "testnet")]
        {
            use snarkvm::parameters::testnet::TransferPrivateProver;
            make_metadata(
                "transfer_private",
                "transferPrivateVerifier",
                "credits.aleo/transfer_private",
                TransferPrivateProver::METADATA,
            )
        }
    }

    #[staticmethod]
    fn transfer_private_to_public() -> Metadata {
        #[cfg(not(feature = "testnet"))]
        {
            use snarkvm::parameters::mainnet::TransferPrivateToPublicProver;
            make_metadata(
                "transfer_private_to_public",
                "transferPrivateToPublicVerifier",
                "credits.aleo/transfer_private_to_public",
                TransferPrivateToPublicProver::METADATA,
            )
        }
        #[cfg(feature = "testnet")]
        {
            use snarkvm::parameters::testnet::TransferPrivateToPublicProver;
            make_metadata(
                "transfer_private_to_public",
                "transferPrivateToPublicVerifier",
                "credits.aleo/transfer_private_to_public",
                TransferPrivateToPublicProver::METADATA,
            )
        }
    }

    #[staticmethod]
    fn transfer_public() -> Metadata {
        #[cfg(not(feature = "testnet"))]
        {
            use snarkvm::parameters::mainnet::TransferPublicProver;
            make_metadata(
                "transfer_public",
                "transferPublicVerifier",
                "credits.aleo/transfer_public",
                TransferPublicProver::METADATA,
            )
        }
        #[cfg(feature = "testnet")]
        {
            use snarkvm::parameters::testnet::TransferPublicProver;
            make_metadata(
                "transfer_public",
                "transferPublicVerifier",
                "credits.aleo/transfer_public",
                TransferPublicProver::METADATA,
            )
        }
    }

    #[staticmethod]
    fn transfer_public_as_signer() -> Metadata {
        #[cfg(not(feature = "testnet"))]
        {
            use snarkvm::parameters::mainnet::TransferPublicAsSignerProver;
            make_metadata(
                "transfer_public_as_signer",
                "transferPublicAsSignerVerifier",
                "credits.aleo/transfer_public_as_signer",
                TransferPublicAsSignerProver::METADATA,
            )
        }
        #[cfg(feature = "testnet")]
        {
            use snarkvm::parameters::testnet::TransferPublicAsSignerProver;
            make_metadata(
                "transfer_public_as_signer",
                "transferPublicAsSignerVerifier",
                "credits.aleo/transfer_public_as_signer",
                TransferPublicAsSignerProver::METADATA,
            )
        }
    }

    #[staticmethod]
    fn transfer_public_to_private() -> Metadata {
        #[cfg(not(feature = "testnet"))]
        {
            use snarkvm::parameters::mainnet::TransferPublicToPrivateProver;
            make_metadata(
                "transfer_public_to_private",
                "transferPublicToPrivateVerifier",
                "credits.aleo/transfer_public_to_private",
                TransferPublicToPrivateProver::METADATA,
            )
        }
        #[cfg(feature = "testnet")]
        {
            use snarkvm::parameters::testnet::TransferPublicToPrivateProver;
            make_metadata(
                "transfer_public_to_private",
                "transferPublicToPrivateVerifier",
                "credits.aleo/transfer_public_to_private",
                TransferPublicToPrivateProver::METADATA,
            )
        }
    }

    #[staticmethod]
    fn unbond_public() -> Metadata {
        #[cfg(not(feature = "testnet"))]
        {
            use snarkvm::parameters::mainnet::UnbondPublicProver;
            make_metadata(
                "unbond_public",
                "unbondPublicVerifier",
                "credits.aleo/unbond_public",
                UnbondPublicProver::METADATA,
            )
        }
        #[cfg(feature = "testnet")]
        {
            use snarkvm::parameters::testnet::UnbondPublicProver;
            make_metadata(
                "unbond_public",
                "unbondPublicVerifier",
                "credits.aleo/unbond_public",
                UnbondPublicProver::METADATA,
            )
        }
    }
}
