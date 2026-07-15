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
    types::{AuthorizationNative, RequestNative},
    Authorization, ExecutionRequest,
};

use pyo3::prelude::*;
use snarkvm::prelude::{FromBytes, ToBytes};

use std::{
    io,
    io::{Read, Write},
    str::FromStr,
};

/// A proving request submitted to the Delegated Proving Service.
///
/// Mirrors the variant layout used by the DPS: `Authorization` carries a
/// fully-constructed `Authorization` (and optional fee `Authorization`);
/// `Request` carries a single signed `Request` (and optional fee `Request`)
/// that the server turns into an `Authorization` via
/// `Process::authorize_request`. JSON is serialized untagged - the field shape
/// (`authorization`/`fee_authorization` vs. `request`/`fee_request`) determines
/// the variant. Bytes carry no discriminator; the reader must know the variant
/// out-of-band.
#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
// The variants intentionally hold the native types inline (no boxing) to mirror
// the wasm reference and keep the byte layout in `write_le`/`read_*_le` exact.
#[allow(clippy::large_enum_variant)]
pub enum ProvingRequestInner {
    Authorization {
        authorization: AuthorizationNative,
        fee_authorization: Option<AuthorizationNative>,
        broadcast: bool,
    },
    Request {
        request: RequestNative,
        fee_request: Option<RequestNative>,
        broadcast: bool,
    },
}

impl ProvingRequestInner {
    /// Returns the Authorization variant's inner authorization, or `None` if
    /// this is a Request variant.
    fn authorization(&self) -> Option<&AuthorizationNative> {
        match self {
            Self::Authorization { authorization, .. } => Some(authorization),
            Self::Request { .. } => None,
        }
    }

    /// Returns the Authorization variant's fee authorization, or `None` if this
    /// is a Request variant or no fee authorization is set.
    fn fee_authorization(&self) -> Option<&AuthorizationNative> {
        match self {
            Self::Authorization {
                fee_authorization, ..
            } => fee_authorization.as_ref(),
            Self::Request { .. } => None,
        }
    }

    /// Returns the Request variant's inner request, or `None` if this is an
    /// Authorization variant.
    fn request(&self) -> Option<&RequestNative> {
        match self {
            Self::Request { request, .. } => Some(request),
            Self::Authorization { .. } => None,
        }
    }

    /// Returns the Request variant's fee request, or `None` if this is an
    /// Authorization variant or no fee request is set.
    fn fee_request(&self) -> Option<&RequestNative> {
        match self {
            Self::Request { fee_request, .. } => fee_request.as_ref(),
            Self::Authorization { .. } => None,
        }
    }

    /// Returns the broadcast flag regardless of variant.
    fn broadcast(&self) -> bool {
        match self {
            Self::Authorization { broadcast, .. } | Self::Request { broadcast, .. } => *broadcast,
        }
    }

    fn is_authorization(&self) -> bool {
        matches!(self, Self::Authorization { .. })
    }

    fn is_request(&self) -> bool {
        matches!(self, Self::Request { .. })
    }

    /// Reads bytes as the Authorization variant.
    /// Layout: `authorization | bool | maybe(fee_authorization) | bool(broadcast)`.
    fn read_authorization_le<R: Read>(mut reader: R) -> io::Result<Self> {
        let authorization = AuthorizationNative::read_le(&mut reader)?;
        let has_fee_auth = bool::read_le(&mut reader)?;
        let fee_authorization = match has_fee_auth {
            false => None,
            true => Some(AuthorizationNative::read_le(&mut reader)?),
        };
        let broadcast = bool::read_le(&mut reader)?;
        Ok(Self::Authorization {
            authorization,
            fee_authorization,
            broadcast,
        })
    }

    /// Reads bytes as the Request variant.
    /// Layout: `request | bool | maybe(fee_request) | bool(broadcast)`.
    fn read_request_le<R: Read>(mut reader: R) -> io::Result<Self> {
        let request = RequestNative::read_le(&mut reader)?;
        let has_fee_request = bool::read_le(&mut reader)?;
        let fee_request = match has_fee_request {
            false => None,
            true => Some(RequestNative::read_le(&mut reader)?),
        };
        let broadcast = bool::read_le(&mut reader)?;
        Ok(Self::Request {
            request,
            fee_request,
            broadcast,
        })
    }
}

impl ToBytes for ProvingRequestInner {
    /// Variant-aware byte serialization matching the DPS layout exactly.
    /// No discriminator byte is written - the reader must know the variant
    /// out-of-band.
    fn write_le<W: Write>(&self, mut writer: W) -> io::Result<()> {
        match self {
            Self::Authorization {
                authorization,
                fee_authorization,
                broadcast,
            } => {
                authorization.write_le(&mut writer)?;
                match fee_authorization {
                    Some(auth) => {
                        true.write_le(&mut writer)?;
                        auth.write_le(&mut writer)?;
                    }
                    None => {
                        false.write_le(&mut writer)?;
                    }
                }
                broadcast.write_le(&mut writer)?;
            }
            Self::Request {
                request,
                fee_request,
                broadcast,
            } => {
                request.write_le(&mut writer)?;
                match fee_request {
                    Some(req) => {
                        true.write_le(&mut writer)?;
                        req.write_le(&mut writer)?;
                    }
                    None => {
                        false.write_le(&mut writer)?;
                    }
                }
                broadcast.write_le(&mut writer)?;
            }
        }
        Ok(())
    }
}

impl FromBytes for ProvingRequestInner {
    /// Reads bytes as the Authorization variant for back-compat. To read the
    /// Request variant, use [`ProvingRequestInner::read_request_le`] explicitly
    /// because the byte layout carries no discriminator.
    fn read_le<R: Read>(reader: R) -> io::Result<Self> {
        Self::read_authorization_le(reader)
    }
}

impl FromStr for ProvingRequestInner {
    type Err = anyhow::Error;

    fn from_str(request: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(request)?)
    }
}

impl std::fmt::Display for ProvingRequestInner {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}",
            serde_json::to_string(self).map_err(|_| std::fmt::Error)?
        )
    }
}

/// Represents a proving request to a prover.
///
/// Carries one of two variants:
/// - `Authorization` - a fully-constructed `Authorization` (plus optional fee
///   authorization).
/// - `Request` - a single signed `Request` (plus optional fee `Request`) that
///   the prover authorizes server-side.
///
/// Use `kind()` when handling a `ProvingRequest` of unknown variant (e.g. after
/// deserialization). Variant-specific accessors raise if called on the wrong
/// variant.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct ProvingRequest(ProvingRequestInner);

#[pymethods]
impl ProvingRequest {
    /// Creates a new Authorization-variant `ProvingRequest` from a function
    /// `Authorization` and an optional fee `Authorization`.
    #[new]
    #[pyo3(signature = (authorization, fee_authorization, broadcast))]
    fn new(
        authorization: Authorization,
        fee_authorization: Option<Authorization>,
        broadcast: bool,
    ) -> Self {
        ProvingRequest(ProvingRequestInner::Authorization {
            authorization: authorization.into(),
            fee_authorization: fee_authorization.map(Into::into),
            broadcast,
        })
    }

    /// Creates a new Request-variant `ProvingRequest` from a single signed
    /// `ExecutionRequest` and an optional signed fee `ExecutionRequest`.
    #[staticmethod]
    #[pyo3(signature = (request, fee_request, broadcast))]
    fn from_request(
        request: ExecutionRequest,
        fee_request: Option<ExecutionRequest>,
        broadcast: bool,
    ) -> Self {
        ProvingRequest(ProvingRequestInner::Request {
            request: request.into(),
            fee_request: fee_request.map(Into::into),
            broadcast,
        })
    }

    /// Returns the variant of this `ProvingRequest`: `"authorization"` or
    /// `"request"`.
    fn kind(&self) -> String {
        if self.0.is_request() {
            "request".to_string()
        } else {
            "authorization".to_string()
        }
    }

    /// Returns `True` if this is the Authorization variant.
    fn is_authorization(&self) -> bool {
        self.0.is_authorization()
    }

    /// Returns `True` if this is the Request variant.
    fn is_request(&self) -> bool {
        self.0.is_request()
    }

    /// Creates a `ProvingRequest` from a JSON string representation. The variant
    /// is determined automatically by the JSON shape.
    #[staticmethod]
    fn from_string(s: &str) -> anyhow::Result<Self> {
        Ok(ProvingRequest(ProvingRequestInner::from_str(s)?))
    }

    /// Reads bytes as an Authorization-variant `ProvingRequest`. For the Request
    /// variant, use `from_bytes_request` explicitly - byte layout carries no
    /// variant discriminator.
    #[staticmethod]
    fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        ProvingRequestInner::read_authorization_le(bytes)
            .map(ProvingRequest)
            .map_err(Into::into)
    }

    /// Reads bytes as a Request-variant `ProvingRequest`. Byte layout is
    /// disjoint from the Authorization variant; callers must pick the right
    /// reader for the bytes they hold.
    #[staticmethod]
    fn from_bytes_request(bytes: &[u8]) -> anyhow::Result<Self> {
        ProvingRequestInner::read_request_le(bytes)
            .map(ProvingRequest)
            .map_err(Into::into)
    }

    /// Creates a left-endian byte representation of the `ProvingRequest`,
    /// dispatching on the variant.
    fn bytes(&self) -> anyhow::Result<Vec<u8>> {
        self.0.to_bytes_le()
    }

    /// Returns the Authorization of the main function in the `ProvingRequest`.
    ///
    /// Raises if this `ProvingRequest` is a Request variant.
    fn authorization(&self) -> anyhow::Result<Authorization> {
        self.0
            .authorization()
            .map(|a| Authorization::from(a.clone()))
            .ok_or_else(|| {
                anyhow::anyhow!(
                "ProvingRequest is a Request variant; call `request()` instead of `authorization()`"
            )
            })
    }

    /// Returns the fee Authorization in the `ProvingRequest`, or `None` when no
    /// fee is set or this is a Request variant.
    fn fee_authorization(&self) -> Option<Authorization> {
        self.0
            .fee_authorization()
            .map(|a| Authorization::from(a.clone()))
    }

    /// Returns the signed `ExecutionRequest` carried by the Request variant.
    ///
    /// Raises if this `ProvingRequest` is an Authorization variant.
    fn request(&self) -> anyhow::Result<ExecutionRequest> {
        self.0.request().map(ExecutionRequest::from).ok_or_else(|| {
            anyhow::anyhow!(
                "ProvingRequest is an Authorization variant; call `authorization()` instead of `request()`"
            )
        })
    }

    /// Returns the signed fee `ExecutionRequest` in the Request variant, or
    /// `None` when no fee request is set or this is an Authorization variant.
    fn fee_request(&self) -> Option<ExecutionRequest> {
        self.0.fee_request().map(ExecutionRequest::from)
    }

    /// Returns the broadcast flag set in the `ProvingRequest`.
    #[getter]
    fn broadcast(&self) -> bool {
        self.0.broadcast()
    }

    /// Returns the `ProvingRequest` as a JSON string.
    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    #[classattr]
    const __hash__: Option<PyObject> = None;
}

impl From<ProvingRequestInner> for ProvingRequest {
    fn from(value: ProvingRequestInner) -> Self {
        Self(value)
    }
}

impl From<ProvingRequest> for ProvingRequestInner {
    fn from(value: ProvingRequest) -> Self {
        value.0
    }
}
