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
    account::Address,
    coinbase::{CoinbaseVerifyingKey, EpochChallenge},
    types::ProverSolutionNative,
};

use pyo3::prelude::*;
use snarkvm::prelude::{FromBytes, ToBytes};

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

/// The prover solution for the coinbase puzzle from a prover.
#[pyclass(frozen)]
pub struct ProverSolution(ProverSolutionNative);

#[pymethods]
impl ProverSolution {
    /// Reads in a ProverSolution from a JSON string.
    #[staticmethod]
    fn from_json(json: &str) -> anyhow::Result<Self> {
        Ok(Self(serde_json::from_str(json)?))
    }

    /// Serialize the given ProverSolution as a JSON string.
    fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string(&self.0)?)
    }

    /// Constructs a ProverSolution from a byte array.
    #[staticmethod]
    fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        ProverSolutionNative::from_bytes_le(bytes).map(Self)
    }

    /// Returns the byte representation of a ProverSolution.
    fn bytes(&self) -> anyhow::Result<Vec<u8>> {
        self.0.to_bytes_le()
    }

    /// Returns the address of the prover.
    fn address(&self) -> Address {
        Address::from(self.0.address())
    }

    /// Returns `true` if the prover solution is valid.
    fn verify(
        &self,
        verifying_key: &CoinbaseVerifyingKey,
        epoch_challenge: &EpochChallenge,
        proof_target: u64,
    ) -> anyhow::Result<bool> {
        self.0.verify(verifying_key, epoch_challenge, proof_target)
    }

    /// Returns the ProverSolution as a JSON string.
    fn __str__(&self) -> anyhow::Result<String> {
        self.to_json()
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
