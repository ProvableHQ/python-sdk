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

use crate::{types::FeeNative, Address, Transition};

use pyo3::prelude::*;

use std::ops::Deref;

#[pyclass(frozen)]
#[derive(Clone)]
pub struct Fee(FeeNative);

#[pymethods]
impl Fee {
    /// Returns true if this is a fee_private transition.
    fn is_fee_private(&self) -> bool {
        self.0.is_fee_private()
    }

    /// Returns true if this is a fee_public transition.
    fn is_fee_public(&self) -> bool {
        self.0.is_fee_public()
    }

    /// Returns the payer, if the fee is public.
    fn payer(&self) -> Option<Address> {
        self.0.payer().map(Into::into)
    }

    /// Returns the transition.
    fn transition(&self) -> Transition {
        self.0.transition().clone().into()
    }

    /// Returns the Fee as a string.
    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    #[classattr]
    const __hash__: Option<PyObject> = None;
}

impl Deref for Fee {
    type Target = FeeNative;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<FeeNative> for Fee {
    fn from(value: FeeNative) -> Self {
        Self(value)
    }
}

impl From<Fee> for FeeNative {
    fn from(value: Fee) -> Self {
        value.0
    }
}
