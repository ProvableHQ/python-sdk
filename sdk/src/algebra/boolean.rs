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

use crate::types::BooleanNative;

use pyo3::prelude::*;

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

#[pyclass(frozen)]
#[derive(Copy, Clone)]
pub struct Boolean(BooleanNative);

#[pymethods]
impl Boolean {
    #[new]
    fn new(value: bool) -> Self {
        Self(BooleanNative::new(value))
    }

    /// Returns the boolean as a string.
    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __bool__(&self) -> bool {
        *self.0
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

impl From<Boolean> for BooleanNative {
    fn from(value: Boolean) -> Self {
        value.0
    }
}
