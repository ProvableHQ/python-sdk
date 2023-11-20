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

use crate::{types::ExecutionNative, Field};

use pyo3::prelude::*;

use std::ops::Deref;

#[pyclass]
#[derive(Clone)]
pub struct Execution(ExecutionNative);

#[pymethods]
impl Execution {
    /// Returns the execution ID.
    fn execution_id(&self) -> anyhow::Result<Field> {
        self.0.to_execution_id().map(Into::into)
    }
}

impl Deref for Execution {
    type Target = ExecutionNative;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<ExecutionNative> for Execution {
    fn from(value: ExecutionNative) -> Self {
        Self(value)
    }
}

impl From<Execution> for ExecutionNative {
    fn from(value: Execution) -> Self {
        value.0
    }
}
