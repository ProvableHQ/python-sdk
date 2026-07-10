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
    types::{
        ArgumentNative, CurrentNetwork, FieldNative, FutureNative, InputNative, OutputNative,
        TransitionNative, U16Native,
    },
    Field, Group, RecordCiphertext, RecordPlaintext, ViewKey,
};

use pyo3::{
    prelude::*,
    types::{PyDict, PyList, PyTuple},
};
use snarkvm::prelude::{compute_function_id, FromBytes, Network, ToBytes};

/// The Aleo transition type.
#[pyclass(frozen)]
pub struct Transition(TransitionNative);

#[pymethods]
impl Transition {
    /// Returns the transition ID.
    #[getter]
    fn id(&self) -> String {
        self.0.id().to_string()
    }

    /// Returns the program ID.
    #[getter]
    fn program_id(&self) -> ProgramID {
        (*self.0.program_id()).into()
    }

    /// Returns the function name.
    #[getter]
    fn function_name(&self) -> Identifier {
        (*self.0.function_name()).into()
    }

    /// Returns true if this is a bond_public transition.
    fn is_bond_public(&self) -> bool {
        self.0.is_bond_public()
    }

    /// Returns true if this is an unbond_public transition.
    fn is_unbond_public(&self) -> bool {
        self.0.is_unbond_public()
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

    /// Reads in a Transition from a JSON string.
    #[staticmethod]
    fn from_json(json: &str) -> anyhow::Result<Self> {
        Ok(Self(serde_json::from_str(json)?))
    }

    /// Serialize the given Transition as a JSON string.
    fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string(&self.0)?)
    }

    /// Constructs a Transition from a byte array.
    #[staticmethod]
    fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        TransitionNative::from_bytes_le(bytes).map(Self)
    }

    /// Returns the byte representation of a Transition.
    fn bytes(&self) -> anyhow::Result<Vec<u8>> {
        self.0.to_bytes_le()
    }

    /// Returns the Transition as a JSON string.
    fn __str__(&self) -> anyhow::Result<String> {
        self.to_json()
    }

    // ---- new methods ----

    /// Returns the transition public key (tpk).
    #[getter]
    fn tpk(&self) -> Group {
        Group::from(self.0.tpk())
    }

    /// Returns the transition commitment (tcm).
    #[getter]
    fn tcm(&self) -> Field {
        (*self.0.tcm()).into()
    }

    /// Returns the transition signer commitment (scm).
    #[getter]
    fn scm(&self) -> Field {
        (*self.0.scm()).into()
    }

    /// Returns the transition view key: `(tpk * view_key_scalar).to_x_coordinate()`.
    fn tvk(&self, view_key: &ViewKey) -> Field {
        (*self.0.tpk() * ***view_key).to_x_coordinate().into()
    }

    /// Returns a list of (Field, RecordCiphertext) tuples for all records in this transition.
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

    /// Returns true if the transition contains the given commitment.
    fn contains_commitment(&self, commitment: &Field) -> bool {
        self.0.contains_commitment(commitment)
    }

    /// Returns true if the transition contains the given serial number.
    fn contains_serial_number(&self, serial_number: &Field) -> bool {
        self.0.contains_serial_number(serial_number)
    }

    /// Returns a list of dicts representing the transition inputs.
    fn inputs(&self, py: Python) -> PyObject {
        let list = PyList::empty(py);
        for input in self.0.inputs().iter() {
            let d = input_to_py_dict(py, input);
            list.append(d).unwrap();
        }
        list.into_py(py)
    }

    /// Returns a list of dicts representing the transition outputs.
    fn outputs(&self, py: Python) -> PyObject {
        let list = PyList::empty(py);
        for output in self.0.outputs().iter() {
            let d = output_to_py_dict(py, output);
            list.append(d).unwrap();
        }
        list.into_py(py)
    }

    /// Decrypt private inputs/outputs using the transition view key.
    fn decrypt_transition(&self, tvk: &Field) -> anyhow::Result<Transition> {
        let function_id = compute_function_id(
            &U16Native::new(CurrentNetwork::ID),
            self.0.program_id(),
            self.0.function_name(),
        )?;

        let mut decrypted_inputs = Vec::with_capacity(self.0.inputs().len());
        for (index, input) in self.0.inputs().iter().enumerate() {
            decrypted_inputs.push(match input {
                InputNative::Private(input_id, Some(ciphertext)) => {
                    let index_field = FieldNative::from_u16(index as u16);
                    let input_view_key =
                        CurrentNetwork::hash_psd4(&[function_id, **tvk, index_field])
                            .map_err(|e| anyhow::anyhow!("{e}"))?;
                    let plaintext = ciphertext
                        .decrypt_symmetric(input_view_key)
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    InputNative::Public(*input_id, Some(plaintext))
                }
                _ => input.clone(),
            });
        }

        let outputs = self.0.outputs();
        let num_inputs = self.0.inputs().len();
        let mut decrypted_outputs = Vec::with_capacity(outputs.len());
        for (index, output) in outputs.iter().enumerate() {
            decrypted_outputs.push(match output {
                OutputNative::Private(output_id, Some(ciphertext)) => {
                    let index_field = FieldNative::from_u16((num_inputs + index) as u16);
                    let output_view_key =
                        CurrentNetwork::hash_psd4(&[function_id, **tvk, index_field])
                            .map_err(|e| anyhow::anyhow!("{e}"))?;
                    let plaintext = ciphertext
                        .decrypt_symmetric(output_view_key)
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    OutputNative::Public(*output_id, Some(plaintext))
                }
                _ => output.clone(),
            });
        }

        TransitionNative::new(
            *self.0.program_id(),
            *self.0.function_name(),
            decrypted_inputs,
            decrypted_outputs,
            *self.0.tpk(),
            *self.0.tcm(),
            *self.0.scm(),
        )
        .map(Self)
        .map_err(|e| anyhow::anyhow!("{e}"))
    }
}

// ---------- helpers ----------

fn input_to_py_dict<'py>(py: Python<'py>, input: &InputNative) -> &'py PyDict {
    let d = PyDict::new(py);
    match input {
        InputNative::Constant(id, plaintext) => {
            d.set_item("type", "constant").unwrap();
            d.set_item("id", id.to_string()).unwrap();
            d.set_item("value", plaintext.as_ref().map(|p| p.to_string()))
                .unwrap();
        }
        InputNative::Public(id, plaintext) => {
            d.set_item("type", "public").unwrap();
            d.set_item("id", id.to_string()).unwrap();
            d.set_item("value", plaintext.as_ref().map(|p| p.to_string()))
                .unwrap();
        }
        InputNative::Private(id, ciphertext) => {
            d.set_item("type", "private").unwrap();
            d.set_item("id", id.to_string()).unwrap();
            d.set_item("value", ciphertext.as_ref().map(|c| c.to_string()))
                .unwrap();
        }
        InputNative::Record(serial_number, tag) => {
            d.set_item("type", "record").unwrap();
            d.set_item("id", serial_number.to_string()).unwrap();
            d.set_item("tag", tag.to_string()).unwrap();
        }
        InputNative::ExternalRecord(id) => {
            d.set_item("type", "externalRecord").unwrap();
            d.set_item("id", id.to_string()).unwrap();
        }
        InputNative::DynamicRecord(id) => {
            d.set_item("type", "record_dynamic").unwrap();
            d.set_item("id", id.to_string()).unwrap();
        }
        InputNative::RecordWithDynamicID(serial_number, tag, dynamic_id) => {
            d.set_item("type", "record_with_dynamic_id").unwrap();
            d.set_item("id", serial_number.to_string()).unwrap();
            d.set_item("tag", tag.to_string()).unwrap();
            d.set_item("dynamic_id", dynamic_id.to_string()).unwrap();
        }
        InputNative::ExternalRecordWithDynamicID(hash, dynamic_id) => {
            d.set_item("type", "external_record_with_dynamic_id")
                .unwrap();
            d.set_item("id", hash.to_string()).unwrap();
            d.set_item("dynamic_id", dynamic_id.to_string()).unwrap();
        }
    }
    d
}

fn output_to_py_dict<'py>(py: Python<'py>, output: &OutputNative) -> &'py PyDict {
    let d = PyDict::new(py);
    match output {
        OutputNative::Constant(id, plaintext) => {
            d.set_item("type", "constant").unwrap();
            d.set_item("id", id.to_string()).unwrap();
            d.set_item("value", plaintext.as_ref().map(|p| p.to_string()))
                .unwrap();
        }
        OutputNative::Public(id, plaintext) => {
            d.set_item("type", "public").unwrap();
            d.set_item("id", id.to_string()).unwrap();
            d.set_item("value", plaintext.as_ref().map(|p| p.to_string()))
                .unwrap();
        }
        OutputNative::Private(id, ciphertext) => {
            d.set_item("type", "private").unwrap();
            d.set_item("id", id.to_string()).unwrap();
            d.set_item("value", ciphertext.as_ref().map(|c| c.to_string()))
                .unwrap();
        }
        OutputNative::Record(commitment, checksum, record_ciphertext, sender_ciphertext) => {
            d.set_item("type", "record").unwrap();
            d.set_item("id", commitment.to_string()).unwrap();
            d.set_item("checksum", checksum.to_string()).unwrap();
            d.set_item("value", record_ciphertext.as_ref().map(|r| r.to_string()))
                .unwrap();
            d.set_item(
                "sender_ciphertext",
                sender_ciphertext.as_ref().map(|c| c.to_string()),
            )
            .unwrap();
        }
        OutputNative::ExternalRecord(id) => {
            d.set_item("type", "external_record").unwrap();
            d.set_item("id", id.to_string()).unwrap();
        }
        OutputNative::Future(id, future) => {
            if let Some(future) = future {
                let obj = future_to_py_dict(py, future, id);
                // copy all items from obj into d
                for (k, v) in obj.iter() {
                    d.set_item(k, v).unwrap();
                }
            } else {
                d.set_item("type", "future").unwrap();
                d.set_item("id", id.to_string()).unwrap();
            }
        }
        OutputNative::DynamicRecord(id) => {
            d.set_item("type", "record_dynamic").unwrap();
            d.set_item("id", id.to_string()).unwrap();
        }
        OutputNative::RecordWithDynamicID(
            commitment,
            checksum,
            record_ciphertext,
            sender_ciphertext,
            dynamic_id,
        ) => {
            d.set_item("type", "record_with_dynamic_id").unwrap();
            d.set_item("id", commitment.to_string()).unwrap();
            d.set_item("checksum", checksum.to_string()).unwrap();
            d.set_item("value", record_ciphertext.as_ref().map(|r| r.to_string()))
                .unwrap();
            d.set_item(
                "sender_ciphertext",
                sender_ciphertext.as_ref().map(|c| c.to_string()),
            )
            .unwrap();
            d.set_item("dynamic_id", dynamic_id.to_string()).unwrap();
        }
        OutputNative::ExternalRecordWithDynamicID(hash, dynamic_id) => {
            d.set_item("type", "external_record_with_dynamic_id")
                .unwrap();
            d.set_item("id", hash.to_string()).unwrap();
            d.set_item("dynamic_id", dynamic_id.to_string()).unwrap();
        }
    }
    d
}

fn future_to_py_dict<'py>(py: Python<'py>, future: &FutureNative, id: &FieldNative) -> &'py PyDict {
    let arguments: Vec<PyObject> = future
        .arguments()
        .iter()
        .map(|arg| argument_to_py_object(py, arg, id))
        .collect();
    let args_list = PyList::new(py, arguments);
    let d = PyDict::new(py);
    d.set_item("type", "future").unwrap();
    d.set_item("id", id.to_string()).unwrap();
    d.set_item("program", future.program_id().to_string())
        .unwrap();
    d.set_item("function", future.function_name().to_string())
        .unwrap();
    d.set_item("arguments", args_list).unwrap();
    d
}

fn argument_to_py_object(py: Python<'_>, arg: &ArgumentNative, id: &FieldNative) -> PyObject {
    match arg {
        ArgumentNative::Plaintext(plaintext) => plaintext.to_string().into_py(py),
        ArgumentNative::Future(future) => future_to_py_dict(py, future, id).into_py(py),
        ArgumentNative::DynamicFuture(dynamic_future) => {
            if let Ok(future) = dynamic_future.to_future() {
                future_to_py_dict(py, &future, id).into_py(py)
            } else {
                let d = PyDict::new(py);
                d.set_item("type", "dynamic_future").unwrap();
                d.set_item("checksum", dynamic_future.checksum().to_string())
                    .unwrap();
                d.into_py(py)
            }
        }
    }
}

impl From<TransitionNative> for Transition {
    fn from(value: TransitionNative) -> Self {
        Self(value)
    }
}

impl From<Transition> for TransitionNative {
    fn from(value: Transition) -> Self {
        value.0
    }
}

impl From<&TransitionNative> for Transition {
    fn from(value: &TransitionNative) -> Self {
        Self(value.clone())
    }
}

impl From<&Transition> for TransitionNative {
    fn from(value: &Transition) -> Self {
        value.0.clone()
    }
}
