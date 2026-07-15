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
    types::{CurrentNetwork, FieldNative},
};

use pyo3::prelude::*;
use snarkvm::console::algorithms::Poseidon4 as Poseidon4Native;
use snarkvm::prelude::{Hash, HashMany, HashToGroup, HashToScalar};

/// Poseidon4 is a cryptographic hash function with an input rate of 4.
#[pyclass(frozen)]
pub struct Poseidon4(Poseidon4Native<CurrentNetwork>);

#[pymethods]
impl Poseidon4 {
    /// Creates a Poseidon4 hasher with the default domain separator "AleoPoseidon4".
    #[new]
    pub fn new() -> anyhow::Result<Self> {
        Poseidon4Native::setup("AleoPoseidon4").map(Self)
    }

    /// Creates a Poseidon4 hasher with a custom domain separator.
    #[staticmethod]
    pub fn setup(domain_separator: &str) -> anyhow::Result<Self> {
        Poseidon4Native::setup(domain_separator).map(Self)
    }

    /// Returns the Poseidon4 hash of the given field elements.
    pub fn hash(&self, input: Vec<Field>) -> anyhow::Result<Field> {
        let input: Vec<FieldNative> = input.into_iter().map(FieldNative::from).collect();
        self.0.hash(&input).map(Field::from)
    }

    /// Returns multiple Poseidon4 hash outputs for the given field elements.
    pub fn hash_many(&self, input: Vec<Field>, num_outputs: u16) -> Vec<Field> {
        let input: Vec<FieldNative> = input.into_iter().map(FieldNative::from).collect();
        self.0
            .hash_many(&input, num_outputs)
            .into_iter()
            .map(Field::from)
            .collect()
    }

    /// Returns the Poseidon4 hash of the given field elements projected to a scalar.
    pub fn hash_to_scalar(&self, input: Vec<Field>) -> anyhow::Result<Scalar> {
        let input: Vec<FieldNative> = input.into_iter().map(FieldNative::from).collect();
        self.0.hash_to_scalar(&input).map(Scalar::from)
    }

    /// Returns the Poseidon4 hash of the given field elements projected to a group element.
    pub fn hash_to_group(&self, input: Vec<Field>) -> anyhow::Result<Group> {
        let input: Vec<FieldNative> = input.into_iter().map(FieldNative::from).collect();
        self.0.hash_to_group(&input).map(Group::from)
    }
}
