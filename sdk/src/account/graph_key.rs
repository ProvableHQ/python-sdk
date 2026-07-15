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

use crate::{types::GraphKeyNative, Field, ViewKey};

use pyo3::prelude::*;

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    ops::Deref,
    str::FromStr,
};

/// The account graph key used for record scanning.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct GraphKey(GraphKeyNative);

#[pymethods]
impl GraphKey {
    /// Derives the graph key from the given view key.
    #[staticmethod]
    fn from_view_key(view_key: &ViewKey) -> anyhow::Result<Self> {
        GraphKeyNative::try_from(&**view_key).map(Self)
    }

    /// Reads in an account graph key from a base58 string.
    #[staticmethod]
    fn from_string(s: &str) -> anyhow::Result<Self> {
        GraphKeyNative::from_str(s).map(Self)
    }

    /// Returns the graph key tag `sk_tag`.
    #[getter]
    fn sk_tag(&self) -> Field {
        self.0.sk_tag().into()
    }

    /// Returns the graph key as a base58 string.
    fn __str__(&self) -> String {
        self.0.to_string()
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

impl Deref for GraphKey {
    type Target = GraphKeyNative;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<GraphKeyNative> for GraphKey {
    fn from(value: GraphKeyNative) -> Self {
        Self(value)
    }
}

impl From<GraphKey> for GraphKeyNative {
    fn from(value: GraphKey) -> Self {
        value.0
    }
}
