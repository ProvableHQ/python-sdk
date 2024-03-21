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
    programs::{Execution, Fee},
    types::{
        DeploymentNative,
        TransactionNative,
        VerifyingKeyNative,
    }
};

use pyo3::prelude::*;
use snarkvm::prelude::{FromBytes, ToBytes};

use std::{ops::Deref, str::FromStr, sync::Arc};

/// Represents a transaction of a deploy, execute or fee type.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct Transaction(TransactionNative);

#[pymethods]
impl Transaction {
    #[staticmethod]
    fn from_execution(execution: Execution, fee: Option<Fee>) -> anyhow::Result<Self> {
        TransactionNative::from_execution(execution.into(), fee.map(Into::into)).map(Self)
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

    #[staticmethod]
    fn underreport_constraints(transaction: Transaction, num_constraints: usize) -> Transaction {
        // Destructure the deployment transaction.
        let transaction_native = TransactionNative::from(transaction);
        let TransactionNative::Deploy(txid, program_owner, deployment, fee) = transaction_native else {
            panic!("Expected a deployment transaction");
        };

        // Decrease the number of constraints in the verifying keys.
        let mut vks_with_underreport = Vec::with_capacity(deployment.verifying_keys().len());
        for (id, (vk, cert)) in deployment.verifying_keys() {
            let mut vk = vk.deref().clone();
            vk.circuit_info.num_constraints -= 2;
            let vk = VerifyingKeyNative::new(Arc::new(vk));
            vks_with_underreport.push((*id, (vk, cert.clone())));
        }

        // Create a new deployment transaction with the underreported verifying keys.
        let adjusted_deployment =
            DeploymentNative::new(deployment.edition(), deployment.program().clone(), vks_with_underreport).unwrap();
        let adjusted_transaction = TransactionNative::Deploy(txid, program_owner, Box::new(adjusted_deployment), fee);

        Transaction(adjusted_transaction)
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
}

impl From<TransactionNative> for Transaction {
    fn from(value: TransactionNative) -> Self {
        Self(value)
    }
}

impl From<Transaction> for TransactionNative {
    fn from(value: Transaction) -> Self {
        value.0
    }
}
