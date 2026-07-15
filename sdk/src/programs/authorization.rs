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
    programs::Transition,
    types::{AuthorizationNative, TransitionNative},
    Field,
};

use pyo3::prelude::*;
use snarkvm::prelude::{FromBytes, ToBytes};

/// The Aleo authorization type.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct Authorization(AuthorizationNative);

#[pymethods]
impl Authorization {
    /// Reads in an Authorization from a JSON string.
    #[staticmethod]
    fn from_json(json: &str) -> anyhow::Result<Self> {
        Ok(Self(serde_json::from_str(json)?))
    }

    /// Serialize the given Authorization as a JSON string.
    fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string(&self.0)?)
    }

    /// Constructs an Authorization from a byte array.
    #[staticmethod]
    fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        AuthorizationNative::from_bytes_le(bytes).map(Self)
    }

    /// Returns the byte representation of an Authorization.
    fn bytes(&self) -> anyhow::Result<Vec<u8>> {
        self.0.to_bytes_le()
    }

    /// Returns the Authorization as a JSON string.
    fn __str__(&self) -> anyhow::Result<String> {
        self.to_json()
    }

    // ---- new methods ----

    /// Returns the execution ID for this authorization.
    fn to_execution_id(&self) -> anyhow::Result<Field> {
        self.0.to_execution_id().map(Into::into)
    }

    /// Returns the list of transitions in this authorization.
    fn transitions(&self) -> Vec<Transition> {
        self.0
            .transitions()
            .into_iter()
            .map(|(_, t)| Transition::from(t))
            .collect()
    }

    /// Returns the function name of the first request.
    fn function_name(&self) -> anyhow::Result<String> {
        Ok(self.0.get(0)?.function_name().to_string())
    }

    /// Returns true if the authorization is for `credits.aleo/fee_private`.
    fn is_fee_private(&self) -> bool {
        self.0.is_fee_private()
    }

    /// Returns true if the authorization is for `credits.aleo/fee_public`.
    fn is_fee_public(&self) -> bool {
        self.0.is_fee_public()
    }

    /// Returns true if the authorization is for `credits.aleo/split`.
    fn is_split(&self) -> bool {
        self.0.is_split()
    }

    /// Returns the number of requests in the authorization.
    fn __len__(&self) -> usize {
        self.0.len()
    }

    /// Returns a new and independent replica of the authorization.
    fn replicate(&self) -> Authorization {
        Authorization(self.0.replicate())
    }

    /// Inserts a transition into the authorization.
    fn insert_transition(&self, transition: &Transition) -> anyhow::Result<()> {
        let native = <TransitionNative as From<&Transition>>::from(transition);
        self.0
            .insert_transition(native)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
}

impl From<AuthorizationNative> for Authorization {
    fn from(value: AuthorizationNative) -> Self {
        Self(value)
    }
}

impl From<Authorization> for AuthorizationNative {
    fn from(value: Authorization) -> Self {
        value.0
    }
}
