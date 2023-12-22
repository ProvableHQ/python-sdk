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

/// The type represents a call to an Aleo program.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct Network(i32);

#[pymethods]
impl Network {
    /// Returns the Poseidon hash with an input rate of 2.
    #[staticmethod]
    fn hash_psd2(input: Vec<Field>) -> anyhow::Result<Field> {
        let input: Vec<_> = input.into_iter().map(Into::into).collect();
        CurrentNetwork::hash_psd2(&input).map(Into::into)
    }
}
