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
    programs::{Identifier, ProgramID},
    types::LocatorNative,
};

use pyo3::prelude::*;

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    ops::Deref,
    str::FromStr,
};

/// A locator is of the form `{program_id}/{resource}` (i.e. `howard.aleo/notify`).
#[pyclass(frozen)]
#[derive(Clone)]
pub struct Locator(LocatorNative);

#[pymethods]
impl Locator {
    /// Initializes a locator from a program ID and resource.
    #[new]
    fn new(program_id: ProgramID, resource: Identifier) -> Self {
        LocatorNative::new(program_id.into(), resource.into()).into()
    }

    /// Parses a Locator from a string.
    #[staticmethod]
    fn from_string(s: &str) -> anyhow::Result<Self> {
        LocatorNative::from_str(s).map(Self)
    }

    /// Returns the program ID.
    fn program_id(&self) -> ProgramID {
        (*self.0.program_id()).into()
    }

    /// Returns the program name.
    fn name(&self) -> Identifier {
        (*self.0.name()).into()
    }

    /// Returns the network-level domain (NLD).
    fn network(&self) -> Identifier {
        (*self.0.network()).into()
    }

    /// Returns the resource name.
    fn resource(&self) -> Identifier {
        (*self.0.resource()).into()
    }

    /// Returns the Locator as a string.
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

impl Deref for Locator {
    type Target = LocatorNative;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<LocatorNative> for Locator {
    fn from(value: LocatorNative) -> Self {
        Self(value)
    }
}

impl From<Locator> for LocatorNative {
    fn from(value: Locator) -> Self {
        value.0
    }
}
