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

use pyo3::prelude::*;

/// The type represents the amount of Aleo credits.
#[pyclass(frozen)]
#[derive(Clone, Copy)]
pub struct Credits(f64);

#[pymethods]
impl Credits {
    #[new]
    fn new(value: f64) -> Self {
        Self(value)
    }

    fn micro(&self) -> MicroCredits {
        MicroCredits((self.0 * 1_000_000.0) as u64)
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __float__(&self) -> f64 {
        self.0
    }

    #[classattr]
    const __hash__: Option<PyObject> = None;
}

/// The type represents the amount of Aleo microcredits.
#[pyclass(frozen)]
#[derive(Clone, Copy)]
pub struct MicroCredits(u64);

#[pymethods]
impl MicroCredits {
    #[new]
    fn new(value: u64) -> Self {
        Self(value)
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __int__(&self) -> u64 {
        self.0
    }

    #[classattr]
    const __hash__: Option<PyObject> = None;
}

impl From<u64> for MicroCredits {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl From<MicroCredits> for u64 {
    fn from(value: MicroCredits) -> Self {
        value.0
    }
}
