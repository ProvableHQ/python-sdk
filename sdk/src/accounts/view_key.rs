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

use super::*;

#[pyclass]
pub struct ViewKey {
    view_key: AleoViewKey<CurrentNetwork>
}

#[pymethods]
impl ViewKey {
    pub fn from_str(view_key_str: &str) -> PyResult<Self> {
        let view_key = AleoViewKey::from_str(view_key_str).map_err(|e| PyErr::new(e.to_string()))?;
        Ok(Self { view_key })
    }

    pub fn to_str(&self) -> PyResult<String> {
        Ok(self.view_key.to_string())
    }

    pub fn to_address(&self) -> PyResult<crate::Address> {
        Ok(self.view_key.to_address())
    }
}
