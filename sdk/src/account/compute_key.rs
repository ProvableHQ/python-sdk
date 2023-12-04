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

use crate::{types::ComputeKeyNative, Address, Group, Scalar};

use pyo3::prelude::*;

use std::ops::Deref;

/// The Aleo compute key type.
#[pyclass(frozen)]
pub struct ComputeKey(ComputeKeyNative);

#[pymethods]
impl ComputeKey {
    /// Returns the address from the compute key.
    fn address(&self) -> Address {
        self.0.to_address().into()
    }

    /// Returns the signature public key.
    fn pk_sig(&self) -> Group {
        self.0.pk_sig().into()
    }

    /// Returns the signature public randomizer.
    fn pr_sig(&self) -> Group {
        self.0.pr_sig().into()
    }

    /// Returns a reference to the PRF secret key.
    fn sk_prf(&self) -> Scalar {
        self.0.sk_prf().into()
    }
}

impl Deref for ComputeKey {
    type Target = ComputeKeyNative;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<ComputeKeyNative> for ComputeKey {
    fn from(value: ComputeKeyNative) -> Self {
        Self(value)
    }
}
