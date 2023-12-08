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

use crate::{programs::Value, types::ResponseNative};

use pyo3::prelude::*;

/// The Aleo response type.
#[pyclass(frozen)]
pub struct Response(ResponseNative);

#[pymethods]
impl Response {
    /// Returns the function outputs.
    fn outputs(&self) -> Vec<Value> {
        self.0.outputs().iter().cloned().map(Into::into).collect()
    }
}

impl From<ResponseNative> for Response {
    fn from(value: ResponseNative) -> Self {
        Self(value)
    }
}
