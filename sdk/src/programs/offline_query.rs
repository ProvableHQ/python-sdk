// Copyright (C) 2019-2023 Aleo Systems Inc.
// This file is part of the Aleo SDK library.
//
// Licensed under GPL-3.0-or-later.

use crate::types::{CurrentNetwork, FieldNative, StatePathNative};

use indexmap::IndexMap;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use snarkvm::prelude::Network;

use std::str::FromStr;

type StateRoot = <CurrentNetwork as Network>::StateRoot;

/// An offline query object used to insert the global state root and state paths
/// needed to create a valid inclusion proof offline.
#[pyclass]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OfflineQuery {
    block_height: u32,
    state_paths: IndexMap<FieldNative, StatePathNative>,
    state_root: StateRoot,
}

#[pymethods]
impl OfflineQuery {
    /// Creates a new offline query object.
    #[staticmethod]
    fn new(block_height: u32, state_root: &str) -> anyhow::Result<Self> {
        let state_root = StateRoot::from_str(state_root)?;
        Ok(Self {
            block_height,
            state_paths: IndexMap::new(),
            state_root,
        })
    }

    /// Sets the block height.
    fn add_block_height(&mut self, block_height: u32) {
        self.block_height = block_height;
    }

    /// Adds a state path for the given commitment field string.
    fn add_state_path(&mut self, commitment: &str, state_path: &str) -> anyhow::Result<()> {
        let commitment = FieldNative::from_str(commitment)?;
        let state_path = StatePathNative::from_str(state_path)?;
        self.state_paths.insert(commitment, state_path);
        Ok(())
    }

    /// Deserializes an OfflineQuery from a JSON string.
    #[staticmethod]
    fn from_string(s: &str) -> anyhow::Result<Self> {
        serde_json::from_str(s).map_err(Into::into)
    }

    /// Serializes the offline query as a JSON string.
    fn __str__(&self) -> String {
        serde_json::to_string(self).expect("OfflineQuery serialization failed")
    }
}
