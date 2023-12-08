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

use crate::types::GroupNative;

use pyo3::prelude::*;
use snarkvm::prelude::Zero;

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    ops::Deref,
    str::FromStr,
};

/// The Aleo group type.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct Group(GroupNative);

#[pymethods]
impl Group {
    /// Parses a group from a string.
    #[staticmethod]
    fn from_string(s: &str) -> anyhow::Result<Self> {
        GroupNative::from_str(s).map(Self)
    }

    /// Returns the `0` element of the group.
    #[staticmethod]
    fn zero() -> Self {
        GroupNative::zero().into()
    }

    /// Returns the Group as a string.
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

impl Deref for Group {
    type Target = GroupNative;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<GroupNative> for Group {
    fn from(value: GroupNative) -> Self {
        Self(value)
    }
}

impl From<Group> for GroupNative {
    fn from(value: Group) -> Self {
        value.0
    }
}
