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

use crate::{coinbase::CoinbaseVerifyingKey, types::CoinbasePuzzleNative};

use pyo3::prelude::*;

#[pyclass]
pub struct CoinbasePuzzle(CoinbasePuzzleNative);

#[pymethods]
impl CoinbasePuzzle {
    /// Load the coinbase puzzle proving and verifying keys.
    #[staticmethod]
    fn load() -> anyhow::Result<Self> {
        let coinbase_puzzle = CoinbasePuzzleNative::load()?;
        Ok(Self(coinbase_puzzle))
    }

    /// Returns the coinbase verifying key.
    fn verifying_key(&self) -> anyhow::Result<CoinbaseVerifyingKey> {
        let verifying_key = self.0.coinbase_verifying_key().clone();
        Ok(CoinbaseVerifyingKey::from(verifying_key))
    }

    #[classattr]
    const __hash__: Option<PyObject> = None;
}
