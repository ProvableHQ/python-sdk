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
    types::{CurrentAleo, LocatorNative, QueryNative, TraceNative},
    Execution, Fee, Locator, Query, Transition,
};

use pyo3::prelude::*;
use rand::{rngs::StdRng, SeedableRng};

/// The Aleo trace type.
#[pyclass]
pub struct Trace(TraceNative);

#[pymethods]
impl Trace {
    /// Returns true if the trace is for a fee transition.
    fn is_fee(&self) -> bool {
        self.0.is_fee()
    }

    /// Returns true if the trace is for a private fee transition.
    fn is_fee_private(&self) -> bool {
        self.0.is_fee_private()
    }

    /// Returns true if the trace is for a public fee transition.
    fn is_fee_public(&self) -> bool {
        self.0.is_fee_public()
    }

    /// Returns the list of transitions.
    fn transitions(&self) -> Vec<Transition> {
        self.0
            .transitions()
            .iter()
            .cloned()
            .map(Into::into)
            .collect()
    }

    /// Returns a new execution with a proof, for the current inclusion assignments and global state root.
    fn prove_execution(&self, locator: Locator) -> anyhow::Result<Execution> {
        let locator: LocatorNative = locator.into();
        let locator_s = locator.to_string();
        self.0
            .prove_execution::<CurrentAleo, _>(&locator_s, &mut StdRng::from_entropy())
            .map(Into::into)
    }

    /// Returns a new fee with a proof, for the current inclusion assignment and global state root.
    fn prove_fee(&self) -> anyhow::Result<Fee> {
        self.0
            .prove_fee::<CurrentAleo, _>(&mut StdRng::from_entropy())
            .map(Into::into)
    }

    fn prepare(&mut self, query: Query) -> anyhow::Result<()> {
        self.0.prepare(QueryNative::from(query))
    }
}

impl From<TraceNative> for Trace {
    fn from(value: TraceNative) -> Self {
        Self(value)
    }
}
