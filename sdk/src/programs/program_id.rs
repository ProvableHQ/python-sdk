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

use crate::{programs::Identifier, types::ProgramIDNative};

use pyo3::prelude::*;

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    ops::Deref,
    str::FromStr,
};

#[pyclass(frozen)]
#[derive(Clone)]
pub struct ProgramID(ProgramIDNative);

#[pymethods]
impl ProgramID {
    /// Parses a string into a program ID.
    #[staticmethod]
    fn from_string(s: &str) -> anyhow::Result<Self> {
        let program_id = FromStr::from_str(s)?;
        Ok(Self(program_id))
    }

    /// Returns the program name.
    fn name(&self) -> Identifier {
        Identifier::from(*self.0.name())
    }

    /// Returns the network-level domain (NLD).
    fn network(&self) -> Identifier {
        Identifier::from(*self.0.network())
    }

    /// Returns `true` if the network-level domain is `aleo`.
    fn is_aleo(&self) -> bool {
        self.0.is_aleo()
    }

    /// Returns the program ID as a string.
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

impl Deref for ProgramID {
    type Target = ProgramIDNative;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<ProgramIDNative> for ProgramID {
    fn from(value: ProgramIDNative) -> Self {
        Self(value)
    }
}

impl From<ProgramID> for ProgramIDNative {
    fn from(value: ProgramID) -> Self {
        value.0
    }
}
