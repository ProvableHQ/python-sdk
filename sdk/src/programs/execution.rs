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

use crate::{programs::Transition, types::ExecutionNative, Authorization, Field};

use pyo3::prelude::*;
use snarkvm::prelude::{FromBytes, ToBytes};

use std::ops::Deref;

/// The type represents a call to an Aleo program.
#[pyclass]
#[derive(Clone)]
pub struct Execution(ExecutionNative);

#[pymethods]
impl Execution {
    /// Returns the Execution ID.
    #[getter]
    fn execution_id(&self) -> anyhow::Result<Field> {
        self.0.to_execution_id().map(Into::into)
    }

    /// Builds an UNPROVEN execution from an authorization — for devnodes
    /// only (they skip proof verification); real networks reject it.
    ///
    /// `state_root` is the node's latest global state root (`sr1…`).
    #[staticmethod]
    fn from_authorization_unproven(
        authorization: &Authorization,
        state_root: &str,
    ) -> anyhow::Result<Self> {
        use snarkvm::prelude::Network;
        use std::str::FromStr;

        let root = <crate::types::CurrentNetwork as Network>::StateRoot::from_str(state_root)?;
        let native: crate::types::AuthorizationNative = authorization.clone().into();
        ExecutionNative::from(native.transitions().values().cloned(), root, None).map(Self)
    }

    /// Reads in an Execution from a JSON string.
    #[staticmethod]
    fn from_json(json: &str) -> anyhow::Result<Self> {
        Ok(Self(serde_json::from_str(json)?))
    }

    /// Serialize the given Execution as a JSON string.
    fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string(&self.0)?)
    }

    /// Constructs an Execution from a byte array.
    #[staticmethod]
    fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        ExecutionNative::from_bytes_le(bytes).map(Self)
    }

    /// Returns the byte representation of an Execution.
    fn bytes(&self) -> anyhow::Result<Vec<u8>> {
        self.0.to_bytes_le()
    }

    /// Returns the Execution as a JSON string.
    fn __str__(&self) -> anyhow::Result<String> {
        self.to_json()
    }

    // ---- new methods ----

    /// Returns the global state root as a string.
    #[getter]
    fn global_state_root(&self) -> String {
        self.0.global_state_root().to_string()
    }

    /// Returns the proof as a string, or None if not present.
    fn proof(&self) -> Option<String> {
        self.0.proof().map(|p| p.to_string())
    }

    /// Returns the list of transitions in this execution.
    fn transitions(&self) -> Vec<Transition> {
        self.0.transitions().map(Transition::from).collect()
    }
}

impl Deref for Execution {
    type Target = ExecutionNative;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<ExecutionNative> for Execution {
    fn from(value: ExecutionNative) -> Self {
        Self(value)
    }
}

impl From<Execution> for ExecutionNative {
    fn from(value: Execution) -> Self {
        value.0
    }
}
