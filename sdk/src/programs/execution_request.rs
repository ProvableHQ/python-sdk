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
    types::{
        FieldNative, IdentifierNative, ProgramIDNative, RequestNative, ValueNative, ValueTypeNative,
    },
    Address, Field, PrivateKey, Signature,
};

use pyo3::prelude::*;
use rand::rngs::StdRng;
use snarkvm::prelude::{FromBytes, ToBytes};

use std::{ops::Deref, str::FromStr};

/// A single signed snarkVM `Request` for a program function call.
///
/// Used by the delegated proving service: a `Request` is signed client-side and
/// the server turns it into an `Authorization` via `Process::authorize_request`
/// before proving.
#[pyclass(frozen, from_py_object)]
#[derive(Clone)]
pub struct ExecutionRequest(RequestNative);

#[pymethods]
impl ExecutionRequest {
    /// Creates a new request by signing over a program ID and set of inputs.
    ///
    /// `root_tvk` is the tvk of the function at the top of the call graph;
    /// `None` for a top-level call or a single-function call graph.
    /// `program_checksum` is the checksum of the program; `None` if the call is
    /// not dynamic. `is_root` indicates the top level function in the call graph
    /// and `is_dynamic` indicates a dynamic call.
    #[staticmethod]
    #[pyo3(signature = (private_key, program_id, function_name, inputs, input_types, root_tvk, program_checksum, is_root, is_dynamic))]
    #[allow(clippy::too_many_arguments)]
    fn sign(
        private_key: &PrivateKey,
        program_id: &str,
        function_name: &str,
        inputs: Vec<String>,
        input_types: Vec<String>,
        root_tvk: Option<Field>,
        program_checksum: Option<Field>,
        is_root: bool,
        is_dynamic: bool,
    ) -> anyhow::Result<Self> {
        // Convert the ProgramID and function name to their native objects.
        let program_id = ProgramIDNative::from_str(program_id)?;
        let function_name = IdentifierNative::from_str(function_name)?;

        // Ensure the inputs are valid Aleo types.
        let inputs = inputs
            .iter()
            .map(|input| ValueNative::from_str(input))
            .collect::<anyhow::Result<Vec<ValueNative>>>()?;

        // Ensure the input types are valid Aleo value types.
        let input_types = input_types
            .iter()
            .map(|input_type| ValueTypeNative::from_str(input_type))
            .collect::<anyhow::Result<Vec<ValueTypeNative>>>()?;

        // Get the root tvk and program checksum if specified.
        let root_tvk = root_tvk.map(FieldNative::from);
        let program_checksum = program_checksum.map(FieldNative::from);

        // Generate an RNG for the function from system entropy.
        let mut rng = rand::make_rng::<StdRng>();

        // Generate the signature over the request.
        let request = RequestNative::sign(
            private_key,
            program_id,
            function_name,
            inputs.into_iter(),
            &input_types,
            root_tvk,
            is_root,
            program_checksum,
            is_dynamic,
            &mut rng,
        )?;

        Ok(ExecutionRequest(request))
    }

    /// Builds a request object from a string representation of a request.
    #[staticmethod]
    fn from_string(request: &str) -> anyhow::Result<Self> {
        Ok(ExecutionRequest(RequestNative::from_str(request)?))
    }

    /// Constructs a request from a byte array.
    #[staticmethod]
    fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        RequestNative::from_bytes_le(bytes).map(ExecutionRequest)
    }

    /// Returns the byte representation of the request.
    fn bytes(&self) -> anyhow::Result<Vec<u8>> {
        self.0.to_bytes_le()
    }

    /// Returns the request signer.
    fn signer(&self) -> Address {
        Address::from(*self.0.signer())
    }

    /// Returns the network ID.
    fn network_id(&self) -> u16 {
        **self.0.network_id()
    }

    /// Returns the program ID.
    fn program_id(&self) -> String {
        self.0.program_id().to_string()
    }

    /// Returns the function name.
    fn function_name(&self) -> String {
        self.0.function_name().to_string()
    }

    /// Returns the function inputs as a list of strings.
    fn inputs(&self) -> Vec<String> {
        self.0
            .inputs()
            .iter()
            .map(|input| input.to_string())
            .collect()
    }

    /// Returns the signature for the transition.
    fn signature(&self) -> Signature {
        Signature::from(*self.0.signature())
    }

    /// Returns the tag secret key `sk_tag`.
    fn sk_tag(&self) -> Field {
        Field::from(*self.0.sk_tag())
    }

    /// Returns the transition view key `tvk`.
    fn tvk(&self) -> Field {
        Field::from(*self.0.tvk())
    }

    /// Returns the transition commitment `tcm`.
    fn tcm(&self) -> Field {
        Field::from(*self.0.tcm())
    }

    /// Returns the signer commitment `scm`.
    fn scm(&self) -> Field {
        Field::from(*self.0.scm())
    }

    /// Returns the request as a JSON string.
    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;
}

impl Deref for ExecutionRequest {
    type Target = RequestNative;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<RequestNative> for ExecutionRequest {
    fn from(value: RequestNative) -> Self {
        Self(value)
    }
}

impl From<&RequestNative> for ExecutionRequest {
    fn from(value: &RequestNative) -> Self {
        Self(value.clone())
    }
}

impl From<ExecutionRequest> for RequestNative {
    fn from(value: ExecutionRequest) -> Self {
        value.0
    }
}

impl From<&ExecutionRequest> for RequestNative {
    fn from(value: &ExecutionRequest) -> Self {
        value.0.clone()
    }
}
