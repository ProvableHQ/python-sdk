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
    types::{
        CiphertextNative, CurrentNetwork, FieldNative, IdentifierNative, LiteralNative,
        PlaintextNative, ProgramIDNative, U16Native,
    },
    Address, Field, Group, Identifier, Literal, Scalar, ViewKey,
};
use std::ops::Deref;

use pyo3::{exceptions::PyTypeError, prelude::*};

use snarkvm::prelude::{
    compute_function_id, FromBits, FromBytes, FromFields, Network, ToBits, ToBitsRaw, ToBytes,
    ToFields, ToFieldsRaw,
};
use std::{collections::HashMap, str::FromStr};

/// The Aleo ciphertext type.
#[pyclass(frozen)]
pub struct Ciphertext(CiphertextNative);

#[pymethods]
impl Ciphertext {
    /// Reads in the ciphertext string.
    #[staticmethod]
    fn from_string(s: &str) -> anyhow::Result<Self> {
        CiphertextNative::from_str(s).map(Self)
    }

    /// Decrypts self into plaintext using the given account view key & nonce.
    pub fn decrypt(&self, view_key: ViewKey, nonce: Group) -> anyhow::Result<Plaintext> {
        self.0
            .decrypt(view_key.into(), nonce.into())
            .map(Into::into)
    }

    /// Decrypts self into plaintext using the given plaintext view key.
    pub fn decrypt_symmetric(&self, plaintext_view_key: Field) -> anyhow::Result<Plaintext> {
        self.0
            .decrypt_symmetric(plaintext_view_key.into())
            .map(Into::into)
    }

    /// Decrypt a ciphertext using the transition view key and (program, function, index).
    ///
    /// Computes: function_id = compute_function_id(network_id, program_id, function_name);
    /// ivk = hash_psd4([function_id, tvk, Field::from_u16(index)]);
    /// then decrypts symmetrically with ivk.
    fn decrypt_with_transition_view_key(
        &self,
        tvk: &Field,
        program: &str,
        function_name: &str,
        index: u16,
    ) -> anyhow::Result<Plaintext> {
        let program_id = ProgramIDNative::from_str(program)?;
        let function_ident = IdentifierNative::from_str(function_name)?;
        let function_id = compute_function_id(
            &U16Native::new(CurrentNetwork::ID),
            &program_id,
            &function_ident,
        )?;
        let index_field = FieldNative::from_u16(index);
        let input_view_key = CurrentNetwork::hash_psd4(&[function_id, **tvk, index_field])
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        self.0
            .decrypt_symmetric(input_view_key)
            .map(Plaintext::from)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Decrypt a ciphertext using the view key, transition public key, and (program, function, index).
    ///
    /// Derives tvk = (tpk * view_key_scalar).to_x_coordinate(), then calls
    /// decrypt_with_transition_view_key logic internally.
    fn decrypt_with_transition_info(
        &self,
        view_key: &ViewKey,
        tpk: &Group,
        program: &str,
        function_name: &str,
        index: u16,
    ) -> anyhow::Result<Plaintext> {
        let program_id = ProgramIDNative::from_str(program)?;
        let function_ident = IdentifierNative::from_str(function_name)?;
        let function_id = compute_function_id(
            &U16Native::new(CurrentNetwork::ID),
            &program_id,
            &function_ident,
        )?;
        let tvk = (**tpk * ***view_key).to_x_coordinate();
        let index_field = FieldNative::from_u16(index);
        let input_view_key = CurrentNetwork::hash_psd4(&[function_id, tvk, index_field])
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        self.0
            .decrypt_symmetric(input_view_key)
            .map(Plaintext::from)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Returns the ciphertext as a string.
    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Deref for Ciphertext {
    type Target = CiphertextNative;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<CiphertextNative> for Ciphertext {
    fn from(value: CiphertextNative) -> Self {
        Self(value)
    }
}

/// The Aleo plaintext type.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct Plaintext(PlaintextNative);

#[pymethods]
impl Plaintext {
    /// Returns a plaintext from a string literal.
    #[staticmethod]
    fn from_string(s: &str) -> anyhow::Result<Self> {
        PlaintextNative::from_str(s).map(Self)
    }

    /// Returns a new Plaintext from a Literal.
    #[staticmethod]
    fn new_literal(literal: Literal) -> Self {
        PlaintextNative::from(LiteralNative::from(literal)).into()
    }

    /// Returns a new Plaintext::Struct from a list of (key, value).
    #[staticmethod]
    fn new_struct(kv: Vec<(Identifier, Plaintext)>) -> Self {
        let kv: Vec<_> = kv.into_iter().map(|(k, v)| (k.into(), v.into())).collect();
        PlaintextNative::Struct(
            indexmap::IndexMap::from_iter(kv),
            std::sync::OnceLock::new(),
        )
        .into()
    }

    /// Returns a new Plaintext::Array from a list of values.
    #[staticmethod]
    fn new_array(values: Vec<Plaintext>) -> Self {
        let values: Vec<_> = values.into_iter().map(Into::into).collect();
        PlaintextNative::Array(values, std::sync::OnceLock::new()).into()
    }

    /// Encrypts self to the given address under the given randomizer.
    fn encrypt(&self, address: Address, randomizer: Scalar) -> anyhow::Result<Ciphertext> {
        self.0.encrypt(&address, randomizer.into()).map(Into::into)
    }

    /// Encrypts self under the given plaintext view key.
    fn encrypt_symmetric(&self, plaintext_view_key: Field) -> anyhow::Result<Ciphertext> {
        self.0
            .encrypt_symmetric(plaintext_view_key.into())
            .map(Into::into)
    }

    /// Returns true if self if Plaintext::Literal.
    fn is_literal(&self) -> bool {
        matches!(self.0, PlaintextNative::Literal(..))
    }

    /// Returns true if self if Plaintext::Struct.
    fn is_struct(&self) -> bool {
        matches!(self.0, PlaintextNative::Struct(..))
    }

    /// Returns true if self if Plaintext::Array
    fn is_array(&self) -> bool {
        matches!(self.0, PlaintextNative::Array(..))
    }

    /// Unboxes the underlying Plaintext::Literal.
    fn as_literal(&self) -> PyResult<Literal> {
        match &self.0 {
            PlaintextNative::Literal(literal, _) => Ok(literal.clone().into()),
            _ => Err(PyTypeError::new_err("Plaintext is not a literal")),
        }
    }

    /// Unboxes the underlying Plaintext::Struct.
    fn as_struct(&self) -> PyResult<HashMap<Identifier, Plaintext>> {
        match &self.0 {
            PlaintextNative::Struct(s, _) => {
                let res: HashMap<Identifier, Plaintext> = s
                    .clone()
                    .into_iter()
                    .map(|(k, v)| (k.into(), v.into()))
                    .collect();
                Ok(res)
            }
            _ => Err(PyTypeError::new_err("Plaintext is not a struct")),
        }
    }

    /// Unboxes the underlying Plaintext::Array.
    fn as_array(&self) -> PyResult<Vec<Plaintext>> {
        match &self.0 {
            PlaintextNative::Array(s, _) => {
                let res: Vec<Plaintext> = s.clone().into_iter().map(|v| v.into()).collect();
                Ok(res)
            }
            _ => Err(PyTypeError::new_err("Plaintext is not an array")),
        }
    }

    /// Finds a member in a struct plaintext by name.
    /// Returns an error if the plaintext is not a struct or the member doesn't exist.
    pub fn find(&self, path: Vec<String>) -> anyhow::Result<Self> {
        let identifiers: Vec<IdentifierNative> = path
            .iter()
            .map(|s| IdentifierNative::from_str(s))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self(self.0.find(&identifiers)?))
    }

    /// Returns the byte representation of the plaintext (little-endian).
    pub fn bytes(&self) -> anyhow::Result<Vec<u8>> {
        self.0.to_bytes_le()
    }

    /// Parses a plaintext from its little-endian byte representation.
    #[staticmethod]
    pub fn from_bytes(bytes: Vec<u8>) -> anyhow::Result<Self> {
        PlaintextNative::from_bytes_le(&bytes).map(Self)
    }

    /// Returns the little-endian bit representation of the plaintext.
    pub fn to_bits_le(&self) -> Vec<bool> {
        self.0.to_bits_le()
    }

    /// Returns the raw little-endian bit representation (no length prefix).
    pub fn to_bits_raw_le(&self) -> Vec<bool> {
        self.0.to_bits_raw_le()
    }

    /// Returns the raw big-endian bit representation (no length prefix).
    pub fn to_bits_raw_be(&self) -> Vec<bool> {
        self.0.to_bits_raw_be()
    }

    /// Returns the raw little-endian byte representation (packed from raw LE bits).
    pub fn to_bytes_raw_le(&self) -> anyhow::Result<Vec<u8>> {
        let bits_le = self.0.to_bits_raw_le();
        let bytes: Vec<u8> = bits_le
            .chunks(8)
            .map(|chunk| {
                crate::types::U8Native::from_bits_le(chunk)
                    .map(|u8_val| u8_val.to_bytes_le().unwrap()[0])
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(bytes)
    }

    /// Returns the raw big-endian byte representation (packed from raw BE bits).
    pub fn to_bytes_raw_be(&self) -> anyhow::Result<Vec<u8>> {
        let bits_be = self.0.to_bits_raw_be();
        let mut bytes: Vec<u8> = bits_be
            .chunks(8)
            .map(|chunk| {
                crate::types::U8Native::from_bits_be(chunk)
                    .map(|u8_val| u8_val.to_bytes_le().unwrap()[0])
            })
            .collect::<Result<Vec<_>, _>>()?;
        bytes.reverse();
        Ok(bytes)
    }

    /// Returns the field element encoding of the plaintext.
    pub fn to_fields(&self) -> anyhow::Result<Vec<Field>> {
        Ok(self.0.to_fields()?.into_iter().map(Into::into).collect())
    }

    /// Recovers a plaintext from field elements.
    #[staticmethod]
    pub fn from_fields(fields: Vec<Field>) -> anyhow::Result<Self> {
        let native: Vec<FieldNative> = fields.into_iter().map(Into::into).collect();
        PlaintextNative::from_fields(&native).map(Self)
    }

    /// Returns the raw field element encoding of the plaintext (no metadata).
    pub fn to_fields_raw(&self) -> anyhow::Result<Vec<Field>> {
        Ok(self
            .0
            .to_fields_raw()?
            .into_iter()
            .map(Into::into)
            .collect())
    }

    /// Returns the type of the plaintext: the literal type name, "struct", or "array".
    #[getter]
    pub fn plaintext_type(&self) -> String {
        match &self.0 {
            PlaintextNative::Literal(literal, _) => literal.to_type().type_name().to_string(),
            PlaintextNative::Struct(..) => "struct".to_string(),
            PlaintextNative::Array(..) => "array".to_string(),
        }
    }

    /// Converts the plaintext to a native Python object.
    ///
    /// Mapping:
    /// - Literal(boolean) → bool
    /// - Literal(u8/u16/u32/u64/u128/i8/i16/i32/i64/i128) → int
    /// - Literal(address/field/group/scalar/signature) → str
    /// - Struct → dict[str, <recursive>]
    /// - Array → list[<recursive>]
    pub fn to_python(&self, py: Python<'_>) -> PyObject {
        plaintext_to_pyobject(&self.0, py)
    }

    /// Returns the plaintext as a string.
    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Deref for Plaintext {
    type Target = PlaintextNative;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<PlaintextNative> for Plaintext {
    fn from(value: PlaintextNative) -> Self {
        Self(value)
    }
}

impl From<Plaintext> for PlaintextNative {
    fn from(value: Plaintext) -> Self {
        value.0
    }
}

fn plaintext_to_pyobject(p: &PlaintextNative, py: Python<'_>) -> PyObject {
    match p {
        PlaintextNative::Literal(lit, _) => match lit {
            LiteralNative::Boolean(b) => (**b).into_py(py),
            LiteralNative::U8(n) => (**n).into_py(py),
            LiteralNative::U16(n) => (**n).into_py(py),
            LiteralNative::U32(n) => (**n).into_py(py),
            LiteralNative::U64(n) => (**n).into_py(py),
            LiteralNative::U128(n) => (**n).into_py(py),
            LiteralNative::I8(n) => (**n).into_py(py),
            LiteralNative::I16(n) => (**n).into_py(py),
            LiteralNative::I32(n) => (**n).into_py(py),
            LiteralNative::I64(n) => (**n).into_py(py),
            LiteralNative::I128(n) => (**n).into_py(py),
            _ => lit.to_string().into_py(py),
        },
        PlaintextNative::Struct(members, _) => {
            let dict = pyo3::types::PyDict::new(py);
            for (k, v) in members.iter() {
                let _ = dict.set_item(k.to_string(), plaintext_to_pyobject(v, py));
            }
            dict.into_py(py)
        }
        PlaintextNative::Array(elems, _) => {
            let list: Vec<PyObject> = elems.iter().map(|e| plaintext_to_pyobject(e, py)).collect();
            list.into_py(py)
        }
    }
}
