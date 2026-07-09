// Copyright (C) 2024 Provable Inc.
// SPDX-License-Identifier: GPL-3.0-or-later
//! Python bindings for leo-abi: ABI generation and compatibility checking for Aleo/Leo programs.

// pyo3 0.20 generates unsafe fn calls inside its macros; the warning is benign
// with edition 2024 where pyo3 0.20 predates the edition-2024 unsafe_op_in_unsafe_fn lint.
#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;

/// Parse the network name string into leo_ast::NetworkName.
fn parse_network(network: &str) -> anyhow::Result<leo_ast::NetworkName> {
    match network {
        "mainnet" => Ok(leo_ast::NetworkName::MainnetV0),
        "testnet" => Ok(leo_ast::NetworkName::TestnetV0),
        "canary" => Ok(leo_ast::NetworkName::CanaryV0),
        other => anyhow::bail!(
            "unknown network {:?}; expected \"mainnet\", \"testnet\", or \"canary\"",
            other
        ),
    }
}

/// Generate an ABI JSON string from Aleo bytecode.
///
/// Args:
///     program_name: The program name (e.g. "token.aleo").
///     bytecode: The Aleo bytecode string.
///     network: One of "mainnet", "testnet", or "canary".
///
/// Returns:
///     A pretty-printed JSON string representing the program ABI.
#[pyfunction]
fn generate_abi(program_name: &str, bytecode: &str, network: &str) -> anyhow::Result<String> {
    let net = parse_network(network)?;
    let abi = leo_abi::aleo::generate_from_bytecode(program_name, bytecode, net)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    Ok(serde_json::to_string_pretty(&abi)?)
}

/// Check whether a candidate ABI JSON is compatible with a standard ABI JSON.
///
/// Args:
///     candidate_abi_json: JSON string of the candidate program's ABI.
///     standard_abi_json: JSON string of the standard/interface ABI.
///
/// Returns:
///     A list of violation strings. Empty list means compatible.
#[pyfunction]
fn check_compatibility(
    candidate_abi_json: &str,
    standard_abi_json: &str,
) -> anyhow::Result<Vec<String>> {
    let candidate: leo_abi::Program = serde_json::from_str(candidate_abi_json)
        .map_err(|e| anyhow::anyhow!("failed to parse candidate ABI JSON: {}", e))?;
    let standard: leo_abi::Program = serde_json::from_str(standard_abi_json)
        .map_err(|e| anyhow::anyhow!("failed to parse standard ABI JSON: {}", e))?;
    Ok(leo_abi::compatibility::check_compatibility(
        &candidate, &standard,
    ))
}

#[pymodule]
fn _aleo_abi(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_abi, m)?)?;
    m.add_function(wrap_pyfunction!(check_compatibility, m)?)?;
    Ok(())
}
