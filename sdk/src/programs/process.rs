// Copyright (C) 2019-2026 Provable Inc.
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
    Address, Authorization, Deployment, Execution, Fee, Field, Identifier, PrivateKey, Program,
    ProgramID, ProvingKey, RecordPlaintext, Response, Trace, Value, VerifyingKey,
};

use indexmap::IndexMap;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use snarkvm::algorithms::snark::varuna::VarunaVersion;
use snarkvm::console::network::ConsensusVersion;
use snarkvm::synthesizer::process::{deployment_cost, execution_cost, InclusionVersion};

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
    fn add_program(&self, program: &Program) -> anyhow::Result<()> {
        self.0.lock().add_program(program)
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
        &self,
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
        &self,
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
                &mut rand::make_rng::<StdRng>(),
            )
            .map(Into::into)
            .map_err(anyhow::Error::from)
    }

    /// Authorizes the fee given the credits record, the base and priority fee amounts (in
    /// microcredits), and the deployment or execution ID.
    fn authorize_fee_private(
        &self,
        private_key: &PrivateKey,
        credits: RecordPlaintext,
        base_fee: u64,
        priority_fee: u64,
        deployment_or_execution_id: Field,
    ) -> anyhow::Result<Authorization> {
        self.0
            .authorize_fee_private::<CurrentAleo, _>(
                private_key,
                credits.into(),
                base_fee,
                priority_fee,
                deployment_or_execution_id.into(),
                &mut rand::make_rng::<StdRng>(),
            )
            .map(Into::into)
            .map_err(anyhow::Error::from)
    }

    /// Authorizes the fee given the base and priority fee amounts (in microcredits) and the
    /// deployment or execution ID.
    fn authorize_fee_public(
        &self,
        private_key: &PrivateKey,
        base_fee: u64,
        priority_fee: u64,
        deployment_or_execution_id: Field,
    ) -> anyhow::Result<Authorization> {
        self.0
            .authorize_fee_public::<CurrentAleo, _>(
                private_key,
                base_fee,
                priority_fee,
                deployment_or_execution_id.into(),
                &mut rand::make_rng::<StdRng>(),
            )
            .map(Into::into)
            .map_err(anyhow::Error::from)
    }

    /// Executes the given authorization.
    fn execute(&self, authorization: Authorization) -> anyhow::Result<(Response, Trace)> {
        self.0
            .execute::<CurrentAleo, _>(authorization.into(), &mut rand::make_rng::<StdRng>())
            .map(|(r, t)| (Response::from(r), Trace::from(t)))
            .map_err(anyhow::Error::from)
    }

    /// Verifies the given execution is valid. Note: This does not check that the global state root exists in the ledger.
    fn verify_execution(&self, execution: &Execution) -> anyhow::Result<()> {
        let execution: crate::types::ExecutionNative = execution.clone().into();
        // Build the map of program stacks referenced by the execution's transitions.
        let execution_stacks = execution
            .transitions()
            .map(|transition| {
                Ok((
                    *transition.program_id(),
                    self.0.get_stack(transition.program_id())?,
                ))
            })
            .collect::<anyhow::Result<IndexMap<_, _>>>()?;
        ProcessNative::verify_execution(
            ConsensusVersion::V17,
            VarunaVersion::V2,
            InclusionVersion::V1,
            &execution,
            &execution_stacks,
        )
    }

    /// Verifies the given fee is valid. Note: This does not check that the global state root exists in the ledger.
    fn verify_fee(&self, fee: &Fee, deployment_or_execution_id: Field) -> anyhow::Result<()> {
        self.0.verify_fee(
            ConsensusVersion::V17,
            VarunaVersion::V2,
            InclusionVersion::V1,
            fee,
            deployment_or_execution_id.into(),
        )
    }

    /// Returns the *minimum* cost in microcredits to publish the given execution (total cost, (storage cost, finalize cost)).
    fn execution_cost(&self, execution: &Execution) -> anyhow::Result<(u64, (u64, u64))> {
        execution_cost(&self.0, &execution.clone().into(), ConsensusVersion::V17)
    }

    /// Synthesizes a deployment for the given program (V9+ semantics: the
    /// program checksum and owner address are set on the deployment).
    ///
    /// The program's imports must already be present in this process.  Key
    /// synthesis is expensive — expect seconds to minutes for large programs.
    fn deploy(&self, program: &Program, owner: &Address) -> anyhow::Result<Deployment> {
        let mut deployment = self
            .0
            .deploy::<CurrentAleo, _>(program, &mut rand::make_rng::<StdRng>())?;
        deployment.set_program_checksum_raw(Some(deployment.program().to_checksum()));
        deployment.set_program_owner_raw(Some(**owner));
        Ok(deployment.into())
    }

    /// Returns the *minimum* cost in microcredits to publish the given deployment.
    fn deployment_cost(&self, deployment: &Deployment) -> anyhow::Result<u64> {
        let (minimum_cost, _) =
            deployment_cost(&self.0, deployment.as_ref(), ConsensusVersion::V17)?;
        Ok(minimum_cost)
    }
}
