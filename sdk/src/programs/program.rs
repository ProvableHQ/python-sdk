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

/// The required signature of an interface function or view function, expressed in the
/// canonical `.aleo` bytecode type notation (e.g. "address.public", "Token.record").
/// The special output marker "future" matches any future type, since a future's locator
/// embeds the id of the program being checked.
struct InterfaceFunction {
    name: &'static str,
    inputs: &'static [&'static str],
    outputs: &'static [&'static str],
}

const FUTURE: &str = "future";

/// Input marker for the ARC-22 Merkle proof array, which is matched structurally rather
/// than by exact type string since the MerkleProof struct may be local or imported.
const MERKLE_PROOFS: &str = "[MerkleProof; 2u32].private";

/// The functions required by the ARC-20 token interface (IARC20).
const ARC20_FUNCTIONS: &[InterfaceFunction] = &[
    InterfaceFunction {
        name: "transfer_public",
        inputs: &["address.public", "u128.public"],
        outputs: &[FUTURE],
    },
    InterfaceFunction {
        name: "transfer_private",
        inputs: &["Token.record", "address.private", "u128.private"],
        outputs: &["Token.record", "Token.record"],
    },
    InterfaceFunction {
        name: "transfer_private_to_public",
        inputs: &["Token.record", "address.public", "u128.public"],
        outputs: &["Token.record", FUTURE],
    },
    InterfaceFunction {
        name: "transfer_public_to_private",
        inputs: &["address.private", "u128.public"],
        outputs: &["Token.record", FUTURE],
    },
    InterfaceFunction {
        name: "transfer_public_as_signer",
        inputs: &["address.public", "u128.public"],
        outputs: &[FUTURE],
    },
    InterfaceFunction {
        name: "transfer_from_public",
        inputs: &["address.public", "address.public", "u128.public"],
        outputs: &[FUTURE],
    },
    InterfaceFunction {
        name: "transfer_from_public_to_private",
        inputs: &["address.public", "address.private", "u128.public"],
        outputs: &["Token.record", FUTURE],
    },
    InterfaceFunction {
        name: "approve_public",
        inputs: &["address.public", "u128.public"],
        outputs: &[FUTURE],
    },
    InterfaceFunction {
        name: "unapprove_public",
        inputs: &["address.public", "u128.public"],
        outputs: &[FUTURE],
    },
    InterfaceFunction {
        name: "join",
        inputs: &["Token.record", "Token.record"],
        outputs: &["Token.record"],
    },
    InterfaceFunction {
        name: "split",
        inputs: &["Token.record", "u128.private"],
        outputs: &["Token.record", "Token.record"],
    },
];

/// The view functions required by both the ARC-20 (IARC20) and ARC-22 (IARC22) token
/// interfaces. View function inputs and outputs are always public.
const ARC20_VIEWS: &[InterfaceFunction] = &[
    InterfaceFunction {
        name: "balance_of",
        inputs: &["address.public"],
        outputs: &["u128.public"],
    },
    InterfaceFunction {
        name: "allowance",
        inputs: &["address.public", "address.public"],
        outputs: &["u128.public"],
    },
    InterfaceFunction {
        name: "supply",
        inputs: &[],
        outputs: &["u128.public"],
    },
    InterfaceFunction {
        name: "max_supply",
        inputs: &[],
        outputs: &["u128.public"],
    },
    InterfaceFunction {
        name: "decimals",
        inputs: &[],
        outputs: &["u8.public"],
    },
    InterfaceFunction {
        name: "name",
        inputs: &[],
        outputs: &["identifier.public"],
    },
    InterfaceFunction {
        name: "symbol",
        inputs: &[],
        outputs: &["identifier.public"],
    },
];

/// The functions required by the ARC-22 compliant token interface (IARC22).
const ARC22_FUNCTIONS: &[InterfaceFunction] = &[
    InterfaceFunction {
        name: "approve_public",
        inputs: &["address.public", "u128.public"],
        outputs: &[FUTURE],
    },
    InterfaceFunction {
        name: "unapprove_public",
        inputs: &["address.public", "u128.public"],
        outputs: &[FUTURE],
    },
    InterfaceFunction {
        name: "transfer_public",
        inputs: &["address.public", "u128.public"],
        outputs: &[FUTURE],
    },
    InterfaceFunction {
        name: "transfer_private",
        inputs: &[
            "address.private",
            "u128.private",
            "Token.record",
            MERKLE_PROOFS,
        ],
        outputs: &[
            "ComplianceRecord.record",
            "Token.record",
            "Token.record",
            FUTURE,
        ],
    },
    InterfaceFunction {
        name: "transfer_private_to_public",
        inputs: &[
            "address.public",
            "u128.public",
            "Token.record",
            MERKLE_PROOFS,
        ],
        outputs: &["ComplianceRecord.record", "Token.record", FUTURE],
    },
    InterfaceFunction {
        name: "transfer_public_to_private",
        inputs: &["address.private", "u128.public"],
        outputs: &["ComplianceRecord.record", "Token.record", FUTURE],
    },
    InterfaceFunction {
        name: "transfer_from_public",
        inputs: &["address.public", "address.public", "u128.public"],
        outputs: &[FUTURE],
    },
    InterfaceFunction {
        name: "transfer_from_public_to_private",
        inputs: &["address.public", "address.private", "u128.public"],
        outputs: &["ComplianceRecord.record", "Token.record", FUTURE],
    },
    InterfaceFunction {
        name: "transfer_public_as_signer",
        inputs: &["address.public", "u128.public"],
        outputs: &[FUTURE],
    },
    InterfaceFunction {
        name: "join",
        inputs: &["Token.record", "Token.record"],
        outputs: &["Token.record"],
    },
    InterfaceFunction {
        name: "split",
        inputs: &["Token.record", "u128.private"],
        outputs: &["Token.record", "Token.record"],
    },
];

/// The members of the MerkleProof struct required by the ARC-22 interface, where
/// the siblings array is sized by `MAX_TREE_DEPTH + 1` with `MAX_TREE_DEPTH = 15`.
const ARC22_MERKLE_PROOF_MEMBERS: &[(&str, &str)] =
    &[("siblings", "[field; 16u32]"), ("leaf_index", "u32")];

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

    /// Returns true if the program implements the ARC-20 fungible token interface (IARC20).
    ///
    /// This checks that the program defines a `Token` record with an `amount: u128` entry
    /// (additional entries are permitted, per the interface's open record definition) and
    /// that every function and view function required by ARC-20 is present with the exact
    /// input and output signature defined by the standard.
    ///
    /// See https://github.com/ProvableHQ/ARCs/blob/master/arc-0020/README.md
    fn is_arc20(&self) -> bool {
        self.record_has_entries("Token", &[("amount", "u128")])
            && ARC20_FUNCTIONS
                .iter()
                .all(|function| self.matches_function(function))
            && ARC20_VIEWS.iter().all(|view| self.matches_view(view))
    }

    /// Returns true if the program implements the ARC-22 compliant token interface (IARC22).
    ///
    /// This checks that the program defines `Token` and `ComplianceRecord` records with the
    /// entries required by the standard (additional entries are permitted, per the
    /// interface's open record definitions), and that every function and view function
    /// required by ARC-22 is present with the exact input and output signature defined by
    /// the standard. The `MerkleProof` struct used by the private transfer functions may be
    /// declared locally (in which case its shape must match the standard exactly) or
    /// imported from another program such as a freeze list registry.
    ///
    /// Note: this checks the token interface (IARC22) only. The freeze list registry
    /// interface (IARC22Freezelist) is typically implemented by a separate program and is
    /// not required for a token program to be considered ARC-22 compliant.
    ///
    /// See https://github.com/ProvableHQ/ARCs/blob/master/arc-0022/README.md
    fn is_arc22(&self) -> bool {
        self.record_has_entries("Token", &[("amount", "u128")])
            && self.record_has_entries(
                "ComplianceRecord",
                &[
                    ("amount", "u128"),
                    ("sender", "address"),
                    ("recipient", "address"),
                ],
            )
            && ARC22_FUNCTIONS
                .iter()
                .all(|function| self.matches_function(function))
            && ARC20_VIEWS.iter().all(|view| self.matches_view(view))
    }
}

// Private helpers backing the ARC-20/ARC-22 interface checks (not exposed to Python).
impl Program {
    // Check that a function exists with the exact interface signature.
    fn matches_function(&self, interface_function: &InterfaceFunction) -> bool {
        let Ok(name) = IdentifierNative::from_str(interface_function.name) else {
            return false;
        };
        let Some(function) = self.0.functions().get(&name) else {
            return false;
        };
        let inputs = function.input_types();
        let outputs = function.output_types();
        inputs.len() == interface_function.inputs.len()
            && outputs.len() == interface_function.outputs.len()
            && inputs
                .iter()
                .zip(interface_function.inputs)
                .all(|(input, expected)| {
                    if *expected == MERKLE_PROOFS {
                        self.is_merkle_proof_array(input)
                    } else {
                        input.to_string() == *expected
                    }
                })
            && outputs
                .iter()
                .zip(interface_function.outputs)
                .all(|(output, expected)| {
                    if *expected == FUTURE {
                        // A `Final` output must be the function's own future — a future pointing
                        // at another program's function does not satisfy the interface.
                        matches!(output, ValueType::Future(locator)
                        if locator.program_id() == self.0.id()
                            && locator.resource().to_string() == interface_function.name)
                    } else {
                        output.to_string() == *expected
                    }
                })
    }

    // Check that a function input is a private two-element array of the ARC-22 MerkleProof
    // struct. The struct may be declared locally, in which case its shape must match the
    // standard exactly, or imported from another program (e.g. a freeze list registry),
    // whose definition cannot be resolved from this program alone.
    fn is_merkle_proof_array(&self, input: &ValueType<CurrentNetwork>) -> bool {
        let ValueType::Private(PlaintextType::Array(array)) = input else {
            return false;
        };
        if **array.length() != 2u32 {
            return false;
        }
        match array.next_element_type() {
            PlaintextType::Struct(name) => {
                name.to_string() == "MerkleProof"
                    && self.struct_matches("MerkleProof", ARC22_MERKLE_PROOF_MEMBERS)
            }
            PlaintextType::ExternalStruct(locator) => {
                locator.resource().to_string() == "MerkleProof"
            }
            _ => false,
        }
    }

    // Check that a view function exists with the exact interface signature.
    fn matches_view(&self, interface_view: &InterfaceFunction) -> bool {
        let Ok(name) = IdentifierNative::from_str(interface_view.name) else {
            return false;
        };
        let Some(view) = self.0.views().get(&name) else {
            return false;
        };
        let inputs = view.input_types();
        let outputs = view.output_types();
        inputs.len() == interface_view.inputs.len()
            && outputs.len() == interface_view.outputs.len()
            && inputs
                .iter()
                .zip(interface_view.inputs)
                .all(|(input, expected)| input.to_string() == *expected)
            && outputs
                .iter()
                .zip(interface_view.outputs)
                .all(|(output, expected)| output.to_string() == *expected)
    }

    // Check that a record exists with a private owner, containing at least the given
    // private entries; additional entries are permitted, matching the `..` in interface
    // record definitions.
    fn record_has_entries(&self, record_name: &str, entries: &[(&str, &str)]) -> bool {
        let Ok(name) = IdentifierNative::from_str(record_name) else {
            return false;
        };
        let Ok(record) = self.0.get_record(&name) else {
            return false;
        };
        if !record.owner().is_private() {
            return false;
        }
        entries.iter().all(|(entry_name, entry_type)| {
            record.entries().iter().any(|(name, ty)| {
                name.to_string() == *entry_name
                    && matches!(ty, EntryType::Private(plaintext) if plaintext.to_string() == *entry_type)
            })
        })
    }

    // Check that a struct exists with exactly the given members in order.
    fn struct_matches(&self, struct_name: &str, members: &[(&str, &str)]) -> bool {
        let Ok(name) = IdentifierNative::from_str(struct_name) else {
            return false;
        };
        let Ok(struct_) = self.0.get_struct(&name) else {
            return false;
        };
        struct_.members().len() == members.len()
            && struct_.members().iter().zip(members).all(
                |((name, ty), (expected_name, expected_type))| {
                    name.to_string() == *expected_name && ty.to_string() == *expected_type
                },
            )
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
            d.set_item("struct_id", locator.resource().to_string())?;
            // External structs live in another program; best-effort expansion only if locally available.
            if let Ok(members) = struct_members_list(py, program, &locator.resource().to_string()) {
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
