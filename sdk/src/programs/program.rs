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

use crate::{
    programs::{Identifier, ProgramID},
    types::ProgramNative,
};

use pyo3::prelude::*;

use std::{ops::Deref, str::FromStr};

#[pyclass(frozen)]
pub struct Program(ProgramNative);

#[pymethods]
impl Program {
    /// Creates a program from source code.
    #[staticmethod]
    fn from_source(s: &str) -> anyhow::Result<Self> {
        ProgramNative::from_str(s).map(Self)
    }

    /// Returns the credits.aleo program
    #[staticmethod]
    fn credits() -> Self {
        Self(ProgramNative::credits().unwrap())
    }

    /// Returns the id of the program
    fn id(&self) -> ProgramID {
        (*self.0.id()).into()
    }

    /// Returns all function names present in the program
    fn functions(&self) -> Vec<Identifier> {
        self.0
            .functions()
            .iter()
            .map(|(id, _func)| Identifier::from(*id))
            .collect()
    }

    /// Returns the imports of the program
    fn imports(&self) -> Vec<ProgramID> {
        self.0
            .imports()
            .iter()
            .map(|(id, _import)| ProgramID::from(*id))
            .collect()
    }

    /// Returns the source code of the program
    fn source(&self) -> String {
        self.0.to_string()
    }

    /// Returns the program ID as a string
    fn __str__(&self) -> String {
        self.0.id().to_string()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    #[classattr]
    const __hash__: Option<PyObject> = None;
}

impl Deref for Program {
    type Target = ProgramNative;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<ProgramNative> for Program {
    fn from(program: ProgramNative) -> Self {
        Self(program)
    }
}
