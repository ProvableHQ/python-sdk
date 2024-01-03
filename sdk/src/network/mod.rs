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

use crate::{types::CurrentNetwork, Field};

use pyo3::prelude::*;

use snarkvm::prelude::Network as NetworkTrait;

/// The type represents the current network.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct Network;

#[pymethods]
impl Network {
    /// The network name.
    #[staticmethod]
    fn name() -> &'static str {
        CurrentNetwork::NAME
    }

    /// The network ID.
    #[staticmethod]
    fn id() -> u16 {
        CurrentNetwork::ID
    }

    /// The network version.
    #[staticmethod]
    fn edition() -> u16 {
        CurrentNetwork::EDITION
    }

    /// Returns the Poseidon hash with an input rate of 2.
    #[staticmethod]
    fn hash_psd2(input: Vec<Field>) -> anyhow::Result<Field> {
        let input: Vec<_> = input.into_iter().map(Into::into).collect();
        CurrentNetwork::hash_psd2(&input).map(Into::into)
    }
}
