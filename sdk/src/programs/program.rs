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
    programs::{Identifier, ProgramID},
    types::{CurrentNetwork, IdentifierNative, ProgramNative},
    Address,
};

use pyo3::{
    prelude::*,
    types::{PyDict, PyList},
};
use snarkvm::prelude::{EntryType, PlaintextType, ValueType};

use std::{ops::Deref, str::FromStr};

/// The Aleo program type.
#[pyclass(frozen)]
pub struct Program(ProgramNative);

#[pymethods]
impl Program {
    /// Creates a program from source code.
    #[staticmethod]
    fn from_source(s: &str) -> anyhow::Result<Self> {
        ProgramNative::from_str(s).map(Self)
    }

    /// Returns the credits.aleo program
    #[staticmethod]
    fn credits() -> Self {
        Self(ProgramNative::credits().unwrap())
    }

    /// Returns the id of the program
    #[getter]
    fn id(&self) -> ProgramID {
        (*self.0.id()).into()
    }

    /// Returns all function names present in the program
    #[getter]
    fn functions(&self) -> Vec<Identifier> {
        self.0
            .functions()
            .iter()
            .map(|(id, _func)| Identifier::from(*id))
            .collect()
    }

    /// Returns the imports of the program
    #[getter]
    fn imports(&self) -> Vec<ProgramID> {
        self.0
            .imports()
            .iter()
            .map(|(id, _import)| ProgramID::from(*id))
            .collect()
    }

    /// Returns the source code of the program
    #[getter]
    fn source(&self) -> String {
        self.0.to_string()
    }

    /// Returns the program ID as a string
    fn __str__(&self) -> String {
        self.0.id().to_string()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    #[classattr]
    const __hash__: Option<PyObject> = None;

    /// Returns true if the program contains a function with the given name.
    fn has_function(&self, function_name: &str) -> bool {
        IdentifierNative::from_str(function_name).is_ok_and(|id| self.0.contains_function(&id))
    }

    /// Returns a list of dicts describing each input of the given function.
    fn get_function_inputs<'py>(
        &self,
        py: Python<'py>,
        function_name: &str,
    ) -> anyhow::Result<PyObject> {
        let id = IdentifierNative::from_str(function_name)?;
        let function = self
            .0
            .functions()
            .get(&id)
            .ok_or_else(|| anyhow::anyhow!("function {} not found", function_name))?;
        let list = PyList::empty(py);
        for input in function.inputs().iter() {
            let register = input.register().to_string();
            match input.value_type() {
                ValueType::Constant(p) => {
                    let d = plaintext_input_to_dict(py, &self.0, p, Some("constant"), None)?;
                    d.set_item("register", &register)?;
                    list.append(d)?;
                }
                ValueType::Public(p) => {
                    let d = plaintext_input_to_dict(py, &self.0, p, Some("public"), None)?;
                    d.set_item("register", &register)?;
                    list.append(d)?;
                }
                ValueType::Private(p) => {
                    let d = plaintext_input_to_dict(py, &self.0, p, Some("private"), None)?;
                    d.set_item("register", &register)?;
                    list.append(d)?;
                }
                ValueType::Record(identifier) => {
                    let d = record_members_dict(py, &self.0, &identifier.to_string())?;
                    d.set_item("register", &register)?;
                    list.append(d)?;
                }
                ValueType::ExternalRecord(locator) => {
                    let d = PyDict::new(py);
                    d.set_item("type", "external_record")?;
                    d.set_item("locator", locator.to_string())?;
                    d.set_item("register", &register)?;
                    list.append(d)?;
                }
                ValueType::Future(locator) => {
                    let d = PyDict::new(py);
                    d.set_item("type", "future")?;
                    d.set_item("locator", locator.to_string())?;
                    d.set_item("register", &register)?;
                    list.append(d)?;
                }
                ValueType::DynamicRecord => {
                    let d = PyDict::new(py);
                    d.set_item("type", "dynamic.record")?;
                    d.set_item("register", &register)?;
                    list.append(d)?;
                }
                ValueType::DynamicFuture => {
                    let d = PyDict::new(py);
                    d.set_item("type", "dynamic.future")?;
                    d.set_item("register", &register)?;
                    list.append(d)?;
                }
            }
        }
        Ok(list.into_py(py))
    }

    /// Returns a list of dicts describing each mapping in the program.
    fn get_mappings<'py>(&self, py: Python<'py>) -> anyhow::Result<PyObject> {
        let list = PyList::empty(py);
        for (name, mapping) in self.0.mappings().iter() {
            let d = PyDict::new(py);
            d.set_item("name", name.to_string())?;
            d.set_item("key_type", mapping.key().plaintext_type().to_string())?;
            d.set_item("value_type", mapping.value().plaintext_type().to_string())?;
            list.append(d)?;
        }
        Ok(list.into_py(py))
    }

    /// Returns a dict describing the members of the given record type.
    fn get_record_members<'py>(
        &self,
        py: Python<'py>,
        record_name: &str,
    ) -> anyhow::Result<PyObject> {
        let d = record_members_dict(py, &self.0, record_name)?;
        Ok(d.into_py(py))
    }

    /// Returns a list of dicts describing the members of the given struct type.
    fn get_struct_members<'py>(
        &self,
        py: Python<'py>,
        struct_name: &str,
    ) -> anyhow::Result<PyObject> {
        let list = struct_members_list(py, &self.0, struct_name)?;
        Ok(list.into_py(py))
    }

    /// Returns the address corresponding to this program's ID.
    fn address(&self) -> anyhow::Result<Address> {
        self.0.id().to_address().map(Address::from)
    }
}

impl Deref for Program {
    type Target = ProgramNative;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<ProgramNative> for Program {
    fn from(program: ProgramNative) -> Self {
        Self(program)
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn plaintext_input_to_dict<'py>(
    py: Python<'py>,
    program: &ProgramNative,
    plaintext: &PlaintextType<CurrentNetwork>,
    visibility: Option<&str>,
    name: Option<&str>,
) -> anyhow::Result<&'py PyDict> {
    let d = PyDict::new(py);
    match plaintext {
        PlaintextType::Literal(lit) => {
            if let Some(n) = name {
                d.set_item("name", n)?;
            }
            d.set_item("type", lit.to_string())?;
        }
        PlaintextType::Struct(struct_id) => {
            if let Some(n) = name {
                d.set_item("name", n)?;
            }
            d.set_item("type", "struct")?;
            d.set_item("struct_id", struct_id.to_string())?;
            // Recursively expand struct members (mirrors wasm get_plaintext_input).
            let members = struct_members_list(py, program, &struct_id.to_string())?;
            d.set_item("members", members)?;
        }
        PlaintextType::Array(array_type) => {
            if let Some(n) = name {
                d.set_item("name", n)?;
            }
            d.set_item("type", "array")?;
            let elem =
                plaintext_input_to_dict(py, program, array_type.base_element_type(), None, None)?;
            d.set_item("element_type", elem)?;
            let length: u32 = **array_type.length();
            d.set_item("length", length)?;
        }
        PlaintextType::ExternalStruct(locator) => {
            if let Some(n) = name {
                d.set_item("name", n)?;
            }
            d.set_item("type", "struct")?;
            d.set_item("struct_id", locator.name().to_string())?;
            // External structs live in another program; best-effort expansion only if locally available.
            if let Ok(members) = struct_members_list(py, program, &locator.name().to_string()) {
                d.set_item("members", members)?;
            }
        }
    }
    if let Some(v) = visibility {
        d.set_item("visibility", v)?;
    }
    Ok(d)
}

fn record_members_dict<'py>(
    py: Python<'py>,
    program: &ProgramNative,
    record_name: &str,
) -> anyhow::Result<&'py PyDict> {
    let id = IdentifierNative::from_str(record_name)?;
    let record_type = program
        .get_record(&id)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let members_list = PyList::empty(py);
    for (member_name, entry_type) in record_type.entries().iter() {
        let (pt, vis) = match entry_type {
            EntryType::Constant(pt) => (pt, "constant"),
            EntryType::Public(pt) => (pt, "public"),
            EntryType::Private(pt) => (pt, "private"),
        };
        let md =
            plaintext_input_to_dict(py, program, pt, Some(vis), Some(&member_name.to_string()))?;
        members_list.append(md)?;
    }
    // Append _nonce as per wasm spec.
    let nonce_d = PyDict::new(py);
    nonce_d.set_item("name", "_nonce")?;
    nonce_d.set_item("type", "group")?;
    nonce_d.set_item("visibility", "public")?;
    members_list.append(nonce_d)?;

    let d = PyDict::new(py);
    d.set_item("type", "record")?;
    d.set_item("record", record_name)?;
    d.set_item("members", members_list)?;
    Ok(d)
}

fn struct_members_list<'py>(
    py: Python<'py>,
    program: &ProgramNative,
    struct_name: &str,
) -> anyhow::Result<&'py PyList> {
    let id = IdentifierNative::from_str(struct_name)?;
    let struct_type = program
        .get_struct(&id)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let list = PyList::empty(py);
    for (member_name, plaintext_type) in struct_type.members().iter() {
        let d = plaintext_input_to_dict(
            py,
            program,
            plaintext_type,
            None,
            Some(&member_name.to_string()),
        )?;
        list.append(d)?;
    }
    Ok(list)
}
