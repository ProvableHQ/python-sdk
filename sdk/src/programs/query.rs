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

use crate::types::QueryNative;

use pyo3::prelude::*;

/// The Aleo query type.
#[pyclass]
#[derive(Clone)]
pub struct Query(QueryNative);

#[pymethods]
impl Query {
    /// The base URL of the node.
    #[staticmethod]
    fn rest(url: String) -> Self {
        QueryNative::REST(url).into()
    }

    #[classattr]
    const __hash__: Option<PyObject> = None;
}

impl From<QueryNative> for Query {
    fn from(value: QueryNative) -> Self {
        Self(value)
    }
}

impl From<Query> for QueryNative {
    fn from(value: Query) -> Self {
        value.0
    }
}
