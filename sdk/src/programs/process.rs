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
    types::{CurrentAleo, ProcessNative},
    Authorization, Execution, Fee, Field, Identifier, MicroCredits, PrivateKey, Program, ProgramID,
    ProvingKey, RecordPlaintext, Response, Trace, Value, VerifyingKey,
};

use pyo3::prelude::*;
use rand::{rngs::StdRng, SeedableRng};
use snarkvm::prelude::cost_in_microcredits;

/// The Aleo process type.
#[pyclass]
pub struct Process(ProcessNative);

#[pymethods]
impl Process {
    /// Initializes a new process.
    #[staticmethod]
    fn load() -> anyhow::Result<Self> {
        ProcessNative::load().map(Self)
    }

    /// Adds a new program to the process
    fn add_program(&mut self, program: &Program) -> anyhow::Result<()> {
        self.0.add_program(program)
    }

    /// Returns true if the process contains the program with the given ID.
    fn contains_program(&self, program_id: &ProgramID) -> bool {
        self.0.contains_program(program_id)
    }

    /// Returns the proving key for the given program ID and function name.
    fn get_proving_key(
        &self,
        program_id: ProgramID,
        function_name: Identifier,
    ) -> anyhow::Result<ProvingKey> {
        self.0
            .get_proving_key(program_id, function_name)
            .map(ProvingKey::from)
    }

    /// Inserts the given proving key, for the given program ID and function name.
    fn insert_proving_key(
        &mut self,
        program_id: &ProgramID,
        function_name: &Identifier,
        proving_key: ProvingKey,
    ) -> anyhow::Result<()> {
        self.0
            .insert_proving_key(program_id, function_name, proving_key.into())
    }

    /// Returns the verifying key for the given program ID and function name.
    fn get_verifying_key(
        &self,
        program_id: ProgramID,
        function_name: Identifier,
    ) -> anyhow::Result<VerifyingKey> {
        self.0
            .get_verifying_key(program_id, function_name)
            .map(Into::into)
    }

    /// Inserts the given verifying key, for the given program ID and function name.
    fn insert_verifying_key(
        &mut self,
        program_id: &ProgramID,
        function_name: &Identifier,
        verifying_key: VerifyingKey,
    ) -> anyhow::Result<()> {
        self.0
            .insert_verifying_key(program_id, function_name, verifying_key.into())
    }

    /// Authorizes a call to the program function for the given inputs.
    fn authorize(
        &self,
        private_key: &PrivateKey,
        program_id: ProgramID,
        function_name: Identifier,
        inputs: Vec<Value>,
    ) -> anyhow::Result<Authorization> {
        self.0
            .authorize::<CurrentAleo, _>(
                private_key,
                program_id,
                function_name,
                inputs.into_iter(),
                &mut StdRng::from_entropy(),
            )
            .map(Into::into)
    }

    /// Authorizes the fee given the credits record, the fee amount (in microcredits), and the deployment or execution ID.
    fn authorize_fee_private(
        &self,
        private_key: &PrivateKey,
        credits: RecordPlaintext,
        base_fee: MicroCredits,
        deployment_or_execution_id: Field,
        priority_fee: Option<MicroCredits>,
    ) -> anyhow::Result<Authorization> {
        self.0
            .authorize_fee_private::<CurrentAleo, _>(
                private_key,
                credits.into(),
                base_fee.into(),
                priority_fee.map(Into::into).unwrap_or(0),
                deployment_or_execution_id.into(),
                &mut StdRng::from_entropy(),
            )
            .map(Into::into)
    }

    /// Authorizes the fee given the the fee amount (in microcredits) and the deployment or execution ID.
    fn authorize_fee_public(
        &self,
        private_key: &PrivateKey,
        base_fee: MicroCredits,
        deployment_or_execution_id: Field,
        priority_fee: Option<MicroCredits>,
    ) -> anyhow::Result<Authorization> {
        self.0
            .authorize_fee_public::<CurrentAleo, _>(
                private_key,
                base_fee.into(),
                priority_fee.map(Into::into).unwrap_or(0),
                deployment_or_execution_id.into(),
                &mut StdRng::from_entropy(),
            )
            .map(Into::into)
    }

    /// Executes the given authorization.
    fn execute(&self, authorization: Authorization) -> anyhow::Result<(Response, Trace)> {
        self.0
            .execute::<CurrentAleo>(authorization.into())
            .map(|(r, t)| (Response::from(r), Trace::from(t)))
    }

    /// Verifies the given execution is valid. Note: This does not check that the global state root exists in the ledger.
    fn verify_execution(&self, execution: &Execution) -> anyhow::Result<()> {
        self.0.verify_execution(execution)
    }

    /// Verifies the given fee is valid. Note: This does not check that the global state root exists in the ledger.
    fn verify_fee(&self, fee: &Fee, deployment_or_execution_id: Field) -> anyhow::Result<()> {
        self.0.verify_fee(fee, deployment_or_execution_id.into())
    }

    /// Returns the *minimum* cost in microcredits to publish the given execution (total cost, (storage cost, finalize cost)).
    fn execution_cost(
        &self,
        execution: &Execution,
    ) -> anyhow::Result<(MicroCredits, (MicroCredits, MicroCredits))> {
        // Compute the storage cost in microcredits.
        let storage_cost = execution.size_in_bytes()?;

        // Compute the finalize cost in microcredits.
        let mut finalize_cost = 0u64;

        for transition in execution.transitions() {
            let program = self.0.get_program(transition.program_id())?;
            let function = program.get_function(transition.function_name())?;
            let cost = match function.finalize_logic() {
                Some(finalize) => cost_in_microcredits(finalize)?,
                None => continue,
            };
            // Accumulate the finalize cost.
            finalize_cost = finalize_cost.checked_add(cost).ok_or(anyhow::anyhow!(
                "The finalize cost computation overflowed for an execution"
            ))?;
        }

        // Compute the total cost in microcredits.
        let total_cost = storage_cost
            .checked_add(finalize_cost)
            .ok_or(anyhow::anyhow!(
                "The total cost computation overflowed for an execution"
            ))?;

        Ok((
            MicroCredits::from(total_cost),
            (
                MicroCredits::from(storage_cost),
                MicroCredits::from(finalize_cost),
            ),
        ))
    }
}
