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
    algebra::{Field, Group, Scalar},
    types::CurrentNetwork,
};

use pyo3::prelude::*;
use snarkvm::console::algorithms::Pedersen128 as Pedersen128Native;
use snarkvm::prelude::{Commit, CommitUncompressed, Hash};

/// Pedersen128 is an additively-homomorphic collision-resistant hash function for up to 128-bit inputs.
#[pyclass(frozen)]
pub struct Pedersen128(Pedersen128Native<CurrentNetwork>);

#[pymethods]
impl Pedersen128 {
    /// Creates a Pedersen128 hasher with the default domain separator "AleoPedersen128".
    #[new]
    pub fn new() -> Self {
        Self(Pedersen128Native::setup("AleoPedersen128"))
    }

    /// Creates a Pedersen128 hasher with a custom domain separator.
    #[staticmethod]
    pub fn setup(domain_separator: &str) -> Self {
        Self(Pedersen128Native::setup(domain_separator))
    }

    /// Returns the Pedersen128 hash of the given bit-string input.
    pub fn hash(&self, input: Vec<bool>) -> anyhow::Result<Field> {
        self.0.hash(&input).map(Field::from)
    }

    /// Returns a Pedersen128 commitment for the given input and randomizer.
    pub fn commit(&self, input: Vec<bool>, randomizer: Scalar) -> anyhow::Result<Field> {
        self.0.commit(&input, &*randomizer).map(Field::from)
    }

    /// Returns a Pedersen128 commitment for the given input and randomizer as a group element.
    pub fn commit_to_group(&self, input: Vec<bool>, randomizer: Scalar) -> anyhow::Result<Group> {
        self.0
            .commit_uncompressed(&input, &*randomizer)
            .map(Group::from)
    }
}
