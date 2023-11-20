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
    types::TransitionNative,
};

use pyo3::prelude::*;

#[pyclass(frozen)]
pub struct Transition(TransitionNative);

#[pymethods]
impl Transition {
    /// Returns the transition ID.
    fn id(&self) -> String {
        self.0.id().to_string()
    }

    /// Returns the program ID.
    fn program_id(&self) -> ProgramID {
        (*self.0.program_id()).into()
    }

    /// Returns the function name.
    fn function_name(&self) -> Identifier {
        (*self.0.function_name()).into()
    }

    /// Returns true if this is a bond transition.
    fn is_bond(&self) -> bool {
        self.0.is_bond()
    }

    /// Returns true if this is an unbond transition.
    fn is_unbond(&self) -> bool {
        self.0.is_unbond()
    }

    /// Returns true if this is a fee_private transition.
    fn is_fee_private(&self) -> bool {
        self.0.is_fee_private()
    }

    /// Returns true if this is a fee_public transition.
    fn is_fee_public(&self) -> bool {
        self.0.is_fee_public()
    }

    /// Returns true if this is a split transition.
    fn is_split(&self) -> bool {
        self.0.is_split()
    }
}

impl From<TransitionNative> for Transition {
    fn from(value: TransitionNative) -> Self {
        Self(value)
    }
}
