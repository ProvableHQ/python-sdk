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

use crate::types::AuthorizationNative;

use pyo3::prelude::*;

#[pyclass(frozen)]
#[derive(Clone)]
pub struct Authorization(AuthorizationNative);

#[pymethods]
impl Authorization {}

impl From<AuthorizationNative> for Authorization {
    fn from(value: AuthorizationNative) -> Self {
        Self(value)
    }
}

impl From<Authorization> for AuthorizationNative {
    fn from(value: Authorization) -> Self {
        value.0
    }
}
