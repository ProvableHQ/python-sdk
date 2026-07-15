// Copyright (C) 2019-2026 Provable Inc.
// This file is part of the Aleo SDK library.
//
// Licensed under GPL-3.0-or-later.

use crate::types::ProofNative;

use pyo3::prelude::*;
use snarkvm::prelude::{FromBytes, ToBytes};

use std::{ops::Deref, str::FromStr};

/// SNARK proof for verification of program execution.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct Proof(ProofNative);

#[pymethods]
impl Proof {
    /// Construct a new proof from a byte array.
    #[staticmethod]
    fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        ProofNative::from_bytes_le(bytes).map(Self)
    }

    /// Create a proof from its string representation.
    #[staticmethod]
    fn from_string(s: &str) -> anyhow::Result<Self> {
        ProofNative::from_str(s).map(Self)
    }

    /// Return the byte representation of the proof.
    fn bytes(&self) -> anyhow::Result<Vec<u8>> {
        self.0.to_bytes_le()
    }

    /// Returns the proof as a string.
    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    #[classattr]
    const __hash__: Option<PyObject> = None;
}

impl Deref for Proof {
    type Target = ProofNative;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<ProofNative> for Proof {
    fn from(proof: ProofNative) -> Self {
        Self(proof)
    }
}

impl From<Proof> for ProofNative {
    fn from(proof: Proof) -> Self {
        proof.0
    }
}
