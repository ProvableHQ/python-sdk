// Copyright (C) 2019-2023 Aleo Systems Inc.
// This file is part of the Aleo SDK library.
//
// Licensed under GPL-3.0-or-later.

use pyo3::prelude::*;

#[cfg(not(feature = "testnet"))]
const BASE_URL: &str = "https://parameters.provable.com/mainnet/";
#[cfg(feature = "testnet")]
const BASE_URL: &str = "https://parameters.provable.com/testnet/";

pub(crate) fn read_prover_checksum(metadata_json: &'static str) -> String {
    let meta: serde_json::Value =
        serde_json::from_str(metadata_json).expect("Metadata was not well-formatted");
    meta["prover_checksum"]
        .as_str()
        .expect("Failed to parse prover_checksum")
        .to_string()
}

fn make_metadata(
    name: &str,
    verifying_key_js_name: &str,
    locator: &str,
    prover_meta: &'static str,
    verifier_meta: &'static str,
) -> Metadata {
    let pc = read_prover_checksum(prover_meta);
    let vc = read_prover_checksum(verifier_meta);
    Metadata {
        name: name.to_string(),
        locator: locator.to_string(),
        prover: format!("{}{}.prover.{}", BASE_URL, name, &pc[..7]),
        verifier: format!("{}.verifier.{}", name, &vc[..7]),
        verifying_key: verifying_key_js_name.to_string(),
    }
}

/// Metadata for an Aleo credits function's proving and verifying keys.
#[pyclass(from_py_object)]
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
            use snarkvm::parameters::mainnet::{BondPublicProver, BondPublicVerifier};
            make_metadata(
                "bond_public",
                "bondPublicVerifier",
                "credits.aleo/bond_public",
                BondPublicProver::METADATA,
                BondPublicVerifier::METADATA,
            )
        }
        #[cfg(feature = "testnet")]
        {
            use snarkvm::parameters::testnet::{BondPublicProver, BondPublicVerifier};
            make_metadata(
                "bond_public",
                "bondPublicVerifier",
                "credits.aleo/bond_public",
                BondPublicProver::METADATA,
                BondPublicVerifier::METADATA,
            )
        }
    }

    #[staticmethod]
    fn bond_validator() -> Metadata {
        #[cfg(not(feature = "testnet"))]
        {
            use snarkvm::parameters::mainnet::{BondValidatorProver, BondValidatorVerifier};
            make_metadata(
                "bond_validator",
                "bondValidatorVerifier",
                "credits.aleo/bond_validator",
                BondValidatorProver::METADATA,
                BondValidatorVerifier::METADATA,
            )
        }
        #[cfg(feature = "testnet")]
        {
            use snarkvm::parameters::testnet::{BondValidatorProver, BondValidatorVerifier};
            make_metadata(
                "bond_validator",
                "bondValidatorVerifier",
                "credits.aleo/bond_validator",
                BondValidatorProver::METADATA,
                BondValidatorVerifier::METADATA,
            )
        }
    }

    #[staticmethod]
    fn claim_unbond_public() -> Metadata {
        #[cfg(not(feature = "testnet"))]
        {
            use snarkvm::parameters::mainnet::{
                ClaimUnbondPublicProver, ClaimUnbondPublicVerifier,
            };
            make_metadata(
                "claim_unbond_public",
                "claimUnbondPublicVerifier",
                "credits.aleo/claim_unbond_public",
                ClaimUnbondPublicProver::METADATA,
                ClaimUnbondPublicVerifier::METADATA,
            )
        }
        #[cfg(feature = "testnet")]
        {
            use snarkvm::parameters::testnet::{
                ClaimUnbondPublicProver, ClaimUnbondPublicVerifier,
            };
            make_metadata(
                "claim_unbond_public",
                "claimUnbondPublicVerifier",
                "credits.aleo/claim_unbond_public",
                ClaimUnbondPublicProver::METADATA,
                ClaimUnbondPublicVerifier::METADATA,
            )
        }
    }

    #[staticmethod]
    fn fee_private() -> Metadata {
        #[cfg(not(feature = "testnet"))]
        {
            use snarkvm::parameters::mainnet::{FeePrivateProver, FeePrivateVerifier};
            make_metadata(
                "fee_private",
                "feePrivateVerifier",
                "credits.aleo/fee_private",
                FeePrivateProver::METADATA,
                FeePrivateVerifier::METADATA,
            )
        }
        #[cfg(feature = "testnet")]
        {
            use snarkvm::parameters::testnet::{FeePrivateProver, FeePrivateVerifier};
            make_metadata(
                "fee_private",
                "feePrivateVerifier",
                "credits.aleo/fee_private",
                FeePrivateProver::METADATA,
                FeePrivateVerifier::METADATA,
            )
        }
    }

    #[staticmethod]
    fn fee_public() -> Metadata {
        #[cfg(not(feature = "testnet"))]
        {
            use snarkvm::parameters::mainnet::{FeePublicProver, FeePublicVerifier};
            make_metadata(
                "fee_public",
                "feePublicVerifier",
                "credits.aleo/fee_public",
                FeePublicProver::METADATA,
                FeePublicVerifier::METADATA,
            )
        }
        #[cfg(feature = "testnet")]
        {
            use snarkvm::parameters::testnet::{FeePublicProver, FeePublicVerifier};
            make_metadata(
                "fee_public",
                "feePublicVerifier",
                "credits.aleo/fee_public",
                FeePublicProver::METADATA,
                FeePublicVerifier::METADATA,
            )
        }
    }

    #[staticmethod]
    fn inclusion() -> Metadata {
        #[cfg(not(feature = "testnet"))]
        {
            use snarkvm::parameters::mainnet::{InclusionProver, InclusionVerifier};
            make_metadata(
                "inclusion",
                "inclusionVerifier",
                "inclusion",
                InclusionProver::METADATA,
                InclusionVerifier::METADATA,
            )
        }
        #[cfg(feature = "testnet")]
        {
            use snarkvm::parameters::testnet::{InclusionProver, InclusionVerifier};
            make_metadata(
                "inclusion",
                "inclusionVerifier",
                "inclusion",
                InclusionProver::METADATA,
                InclusionVerifier::METADATA,
            )
        }
    }

    #[staticmethod]
    fn join() -> Metadata {
        #[cfg(not(feature = "testnet"))]
        {
            use snarkvm::parameters::mainnet::{JoinProver, JoinVerifier};
            make_metadata(
                "join",
                "joinVerifier",
                "credits.aleo/join",
                JoinProver::METADATA,
                JoinVerifier::METADATA,
            )
        }
        #[cfg(feature = "testnet")]
        {
            use snarkvm::parameters::testnet::{JoinProver, JoinVerifier};
            make_metadata(
                "join",
                "joinVerifier",
                "credits.aleo/join",
                JoinProver::METADATA,
                JoinVerifier::METADATA,
            )
        }
    }

    #[staticmethod]
    fn set_validator_state() -> Metadata {
        #[cfg(not(feature = "testnet"))]
        {
            use snarkvm::parameters::mainnet::{
                SetValidatorStateProver, SetValidatorStateVerifier,
            };
            make_metadata(
                "set_validator_state",
                "setValidatorStateVerifier",
                "credits.aleo/set_validator_state",
                SetValidatorStateProver::METADATA,
                SetValidatorStateVerifier::METADATA,
            )
        }
        #[cfg(feature = "testnet")]
        {
            use snarkvm::parameters::testnet::{
                SetValidatorStateProver, SetValidatorStateVerifier,
            };
            make_metadata(
                "set_validator_state",
                "setValidatorStateVerifier",
                "credits.aleo/set_validator_state",
                SetValidatorStateProver::METADATA,
                SetValidatorStateVerifier::METADATA,
            )
        }
    }

    #[staticmethod]
    fn split() -> Metadata {
        #[cfg(not(feature = "testnet"))]
        {
            use snarkvm::parameters::mainnet::{SplitProver, SplitVerifier};
            make_metadata(
                "split",
                "splitVerifier",
                "credits.aleo/split",
                SplitProver::METADATA,
                SplitVerifier::METADATA,
            )
        }
        #[cfg(feature = "testnet")]
        {
            use snarkvm::parameters::testnet::{SplitProver, SplitVerifier};
            make_metadata(
                "split",
                "splitVerifier",
                "credits.aleo/split",
                SplitProver::METADATA,
                SplitVerifier::METADATA,
            )
        }
    }

    #[staticmethod]
    fn transfer_private() -> Metadata {
        #[cfg(not(feature = "testnet"))]
        {
            use snarkvm::parameters::mainnet::{TransferPrivateProver, TransferPrivateVerifier};
            make_metadata(
                "transfer_private",
                "transferPrivateVerifier",
                "credits.aleo/transfer_private",
                TransferPrivateProver::METADATA,
                TransferPrivateVerifier::METADATA,
            )
        }
        #[cfg(feature = "testnet")]
        {
            use snarkvm::parameters::testnet::{TransferPrivateProver, TransferPrivateVerifier};
            make_metadata(
                "transfer_private",
                "transferPrivateVerifier",
                "credits.aleo/transfer_private",
                TransferPrivateProver::METADATA,
                TransferPrivateVerifier::METADATA,
            )
        }
    }

    #[staticmethod]
    fn transfer_private_to_public() -> Metadata {
        #[cfg(not(feature = "testnet"))]
        {
            use snarkvm::parameters::mainnet::{
                TransferPrivateToPublicProver, TransferPrivateToPublicVerifier,
            };
            make_metadata(
                "transfer_private_to_public",
                "transferPrivateToPublicVerifier",
                "credits.aleo/transfer_private_to_public",
                TransferPrivateToPublicProver::METADATA,
                TransferPrivateToPublicVerifier::METADATA,
            )
        }
        #[cfg(feature = "testnet")]
        {
            use snarkvm::parameters::testnet::{
                TransferPrivateToPublicProver, TransferPrivateToPublicVerifier,
            };
            make_metadata(
                "transfer_private_to_public",
                "transferPrivateToPublicVerifier",
                "credits.aleo/transfer_private_to_public",
                TransferPrivateToPublicProver::METADATA,
                TransferPrivateToPublicVerifier::METADATA,
            )
        }
    }

    #[staticmethod]
    fn transfer_public() -> Metadata {
        #[cfg(not(feature = "testnet"))]
        {
            use snarkvm::parameters::mainnet::{TransferPublicProver, TransferPublicVerifier};
            make_metadata(
                "transfer_public",
                "transferPublicVerifier",
                "credits.aleo/transfer_public",
                TransferPublicProver::METADATA,
                TransferPublicVerifier::METADATA,
            )
        }
        #[cfg(feature = "testnet")]
        {
            use snarkvm::parameters::testnet::{TransferPublicProver, TransferPublicVerifier};
            make_metadata(
                "transfer_public",
                "transferPublicVerifier",
                "credits.aleo/transfer_public",
                TransferPublicProver::METADATA,
                TransferPublicVerifier::METADATA,
            )
        }
    }

    #[staticmethod]
    fn transfer_public_as_signer() -> Metadata {
        #[cfg(not(feature = "testnet"))]
        {
            use snarkvm::parameters::mainnet::{
                TransferPublicAsSignerProver, TransferPublicAsSignerVerifier,
            };
            make_metadata(
                "transfer_public_as_signer",
                "transferPublicAsSignerVerifier",
                "credits.aleo/transfer_public_as_signer",
                TransferPublicAsSignerProver::METADATA,
                TransferPublicAsSignerVerifier::METADATA,
            )
        }
        #[cfg(feature = "testnet")]
        {
            use snarkvm::parameters::testnet::{
                TransferPublicAsSignerProver, TransferPublicAsSignerVerifier,
            };
            make_metadata(
                "transfer_public_as_signer",
                "transferPublicAsSignerVerifier",
                "credits.aleo/transfer_public_as_signer",
                TransferPublicAsSignerProver::METADATA,
                TransferPublicAsSignerVerifier::METADATA,
            )
        }
    }

    #[staticmethod]
    fn transfer_public_to_private() -> Metadata {
        #[cfg(not(feature = "testnet"))]
        {
            use snarkvm::parameters::mainnet::{
                TransferPublicToPrivateProver, TransferPublicToPrivateVerifier,
            };
            make_metadata(
                "transfer_public_to_private",
                "transferPublicToPrivateVerifier",
                "credits.aleo/transfer_public_to_private",
                TransferPublicToPrivateProver::METADATA,
                TransferPublicToPrivateVerifier::METADATA,
            )
        }
        #[cfg(feature = "testnet")]
        {
            use snarkvm::parameters::testnet::{
                TransferPublicToPrivateProver, TransferPublicToPrivateVerifier,
            };
            make_metadata(
                "transfer_public_to_private",
                "transferPublicToPrivateVerifier",
                "credits.aleo/transfer_public_to_private",
                TransferPublicToPrivateProver::METADATA,
                TransferPublicToPrivateVerifier::METADATA,
            )
        }
    }

    #[staticmethod]
    fn unbond_public() -> Metadata {
        #[cfg(not(feature = "testnet"))]
        {
            use snarkvm::parameters::mainnet::{UnbondPublicProver, UnbondPublicVerifier};
            make_metadata(
                "unbond_public",
                "unbondPublicVerifier",
                "credits.aleo/unbond_public",
                UnbondPublicProver::METADATA,
                UnbondPublicVerifier::METADATA,
            )
        }
        #[cfg(feature = "testnet")]
        {
            use snarkvm::parameters::testnet::{UnbondPublicProver, UnbondPublicVerifier};
            make_metadata(
                "unbond_public",
                "unbondPublicVerifier",
                "credits.aleo/unbond_public",
                UnbondPublicProver::METADATA,
                UnbondPublicVerifier::METADATA,
            )
        }
    }
}
