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
    programs::{Deployment, Execution, Fee, Program, Transition},
    types::TransactionNative,
    Field, PrivateKey, RecordCiphertext, RecordPlaintext, ViewKey,
};

use pyo3::{
    prelude::*,
    types::{PyDict, PyList, PyTuple},
};
use snarkvm::prelude::{FromBytes, ToBytes};

use std::str::FromStr;

/// Represents a transaction of a deploy, execute or fee type.
#[pyclass(frozen)]
pub struct Transaction(TransactionNative);

#[pymethods]
impl Transaction {
    #[staticmethod]
    fn from_execution(execution: Execution, fee: Option<Fee>) -> anyhow::Result<Self> {
        TransactionNative::from_execution(execution.into(), fee.map(Into::into)).map(Self)
    }

    /// Constructs a deployment transaction: the deployer signs program
    /// ownership over the deployment ID, and the fee pays for publication.
    #[staticmethod]
    fn from_deployment(
        private_key: &PrivateKey,
        deployment: Deployment,
        fee: Fee,
    ) -> anyhow::Result<Self> {
        use rand::rngs::StdRng;
        use snarkvm::prelude::ProgramOwner;

        let deployment_id = deployment.as_ref().to_deployment_id()?;
        let owner = ProgramOwner::new(private_key, deployment_id, &mut rand::make_rng::<StdRng>())?;
        TransactionNative::from_deployment(owner, deployment.into(), fee.into()).map(Self)
    }

    /// Parses a Transaction from a JSON string.
    #[staticmethod]
    fn from_json(json: &str) -> anyhow::Result<Self> {
        TransactionNative::from_str(json).map(Self)
    }

    /// Serialize the given Transaction as a JSON string.
    fn to_json(&self) -> String {
        self.0.to_string()
    }

    /// Constructs a Transation from a byte array.
    #[staticmethod]
    fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        TransactionNative::from_bytes_le(bytes).map(Self)
    }

    /// Returns the byte representation of a Transaction.
    fn bytes(&self) -> anyhow::Result<Vec<u8>> {
        self.0.to_bytes_le()
    }

    /// Returns the Transaction as a JSON string.
    fn __str__(&self) -> String {
        self.to_json()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    #[classattr]
    const __hash__: Option<PyObject> = None;

    // ---- new methods ----

    /// Returns the transaction ID string.
    #[getter]
    fn id(&self) -> String {
        self.0.id().to_string()
    }

    /// Returns "deploy", "execute", or "fee".
    #[getter]
    fn transaction_type(&self) -> String {
        match &self.0 {
            TransactionNative::Deploy(..) => "deploy".to_string(),
            TransactionNative::Execute(..) => "execute".to_string(),
            TransactionNative::Fee(..) => "fee".to_string(),
        }
    }

    /// Returns true if the transaction is a deployment transaction.
    fn is_deploy(&self) -> bool {
        self.0.is_deploy()
    }

    /// Returns true if the transaction is an execution transaction.
    fn is_execute(&self) -> bool {
        self.0.is_execute()
    }

    /// Returns true if the transaction is a fee transaction.
    fn is_fee(&self) -> bool {
        self.0.is_fee()
    }

    /// Returns the transaction's base fee amount.
    #[getter]
    fn base_fee_amount(&self) -> u64 {
        self.0.base_fee_amount().map(|f| *f).unwrap_or(0)
    }

    /// Returns the transaction's total fee amount.
    #[getter]
    fn fee_amount(&self) -> u64 {
        self.0.fee_amount().map(|f| *f).unwrap_or(0)
    }

    /// Returns the transaction's priority fee amount.
    #[getter]
    fn priority_fee_amount(&self) -> u64 {
        self.0.priority_fee_amount().map(|f| *f).unwrap_or(0)
    }

    /// Returns a list of (Field, RecordCiphertext) tuples for all records in the transaction.
    fn records(&self, py: Python) -> PyObject {
        let list = PyList::empty(py);
        for (commitment, record) in self.0.records() {
            let f: Field = (*commitment).into();
            let rc: RecordCiphertext = record.clone().into();
            let t = PyTuple::new(py, [f.into_py(py), rc.into_py(py)]);
            list.append(t).unwrap();
        }
        list.into_py(py)
    }

    /// Returns list of RecordPlaintext owned by view_key.
    fn owned_records(&self, view_key: &ViewKey) -> Vec<RecordPlaintext> {
        self.0
            .records()
            .filter_map(|(_, rc)| rc.decrypt(view_key).ok().map(RecordPlaintext::from))
            .collect()
    }

    /// Returns the RecordCiphertext for a commitment, or None.
    fn find_record(&self, commitment: &Field) -> Option<RecordCiphertext> {
        self.0.find_record(commitment).map(|rc| rc.clone().into())
    }

    /// Returns true if the transaction contains the given serial number.
    fn contains_serial_number(&self, serial_number: &Field) -> bool {
        self.0.contains_serial_number(serial_number)
    }

    /// Returns true if the transaction contains the given commitment.
    fn contains_commitment(&self, commitment: &Field) -> bool {
        self.0.contains_commitment(commitment)
    }

    /// Returns the execution within the transaction (if present).
    fn execution(&self) -> Option<Execution> {
        self.0.execution().map(|e| Execution::from(e.clone()))
    }

    /// Returns the list of transitions in the transaction.
    fn transitions(&self) -> Vec<Transition> {
        self.0.transitions().map(Transition::from).collect()
    }

    /// Returns the program deployed within the transaction (if it is a deployment transaction).
    fn deployed_program(&self) -> Option<Program> {
        self.0
            .deployment()
            .map(|d| Program::from(d.program().clone()))
    }

    /// Returns a list of dicts with keys "program", "function", "verifying_key", "certificate".
    fn verifying_keys(&self, py: Python) -> PyObject {
        let list = PyList::empty(py);
        if let Some(deployment) = self.0.deployment() {
            for (fname, (vk, cert)) in deployment.verifying_keys() {
                let d = PyDict::new(py);
                d.set_item("program", deployment.program_id().to_string())
                    .unwrap();
                d.set_item("function", fname.to_string()).unwrap();
                d.set_item("verifying_key", vk.to_string()).unwrap();
                d.set_item("certificate", cert.to_string()).unwrap();
                list.append(d).unwrap();
            }
        }
        list.into_py(py)
    }

    /// Returns a Python dict summarizing the transaction (id, type, fee fields).
    fn summary(&self, py: Python) -> PyObject {
        let d = PyDict::new(py);
        d.set_item("id", self.id()).unwrap();
        d.set_item("type", self.transaction_type()).unwrap();
        d.set_item("fee_amount", self.fee_amount()).unwrap();
        d.set_item("base_fee", self.base_fee_amount()).unwrap();
        d.set_item("priority_fee", self.priority_fee_amount())
            .unwrap();
        d.into_py(py)
    }
}

impl From<TransactionNative> for Transaction {
    fn from(value: TransactionNative) -> Self {
        Self(value)
    }
}
