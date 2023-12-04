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

use pyo3::prelude::*;

mod account;
mod algebra;
mod coinbase;
mod credits;
mod programs;
mod types;

use account::*;
use algebra::*;
use coinbase::*;
use credits::*;
use programs::*;

/// The Aleo Python SDK provides a set of libraries aimed at empowering
/// Python developers with zk (zero-knowledge) programming capabilities
/// via the usage of Aleo's zkSnarks.
#[pymodule]
#[pyo3(name = "aleo")]
fn register_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Account>()?;
    m.add_class::<Address>()?;
    m.add_class::<Authorization>()?;
    m.add_class::<Boolean>()?;
    m.add_class::<CoinbasePuzzle>()?;
    m.add_class::<CoinbaseVerifyingKey>()?;
    m.add_class::<ComputeKey>()?;
    m.add_class::<Credits>()?;
    m.add_class::<EpochChallenge>()?;
    m.add_class::<Execution>()?;
    m.add_class::<Fee>()?;
    m.add_class::<Field>()?;
    m.add_class::<Group>()?;
    m.add_class::<Identifier>()?;
    m.add_class::<I8>()?;
    m.add_class::<I16>()?;
    m.add_class::<I32>()?;
    m.add_class::<I64>()?;
    m.add_class::<I128>()?;
    m.add_class::<Literal>()?;
    m.add_class::<Locator>()?;
    m.add_class::<MicroCredits>()?;
    m.add_class::<PrivateKey>()?;
    m.add_class::<Process>()?;
    m.add_class::<Program>()?;
    m.add_class::<ProgramID>()?;
    m.add_class::<ProverSolution>()?;
    m.add_class::<ProvingKey>()?;
    m.add_class::<Query>()?;
    m.add_class::<RecordCiphertext>()?;
    m.add_class::<RecordPlaintext>()?;
    m.add_class::<Response>()?;
    m.add_class::<Scalar>()?;
    m.add_class::<Signature>()?;
    m.add_class::<Trace>()?;
    m.add_class::<Transaction>()?;
    m.add_class::<Transition>()?;
    m.add_class::<U8>()?;
    m.add_class::<U16>()?;
    m.add_class::<U32>()?;
    m.add_class::<U64>()?;
    m.add_class::<U128>()?;
    m.add_class::<Value>()?;
    m.add_class::<VerifyingKey>()?;
    m.add_class::<ViewKey>()?;
    Ok(())
}
