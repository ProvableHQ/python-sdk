// Copyright (C) 2019-2026 Provable Inc.
// This file is part of the Aleo SDK library.
//
// Licensed under GPL-3.0-or-later.

use crate::{types::DynamicRecordNative, Address, Field, Group, RecordPlaintext};

use pyo3::prelude::*;
use snarkvm::prelude::{FromBytes, ToBits, ToBytes, ToFields};

use std::{ops::Deref, str::FromStr};

/// A fixed-size representation of an Aleo record that stores the Merkle root
/// of the record data rather than the full data.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct DynamicRecord(DynamicRecordNative);

#[pymethods]
impl DynamicRecord {
    /// Creates a DynamicRecord from its string representation.
    #[staticmethod]
    fn from_string(s: &str) -> anyhow::Result<Self> {
        DynamicRecordNative::from_str(s).map(Self)
    }

    /// Creates a DynamicRecord from a RecordPlaintext.
    #[staticmethod]
    fn from_record(record: &RecordPlaintext) -> anyhow::Result<Self> {
        DynamicRecordNative::from_record(&**record).map(Self)
    }

    /// Deserializes a DynamicRecord from a little-endian byte array.
    #[staticmethod]
    fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        DynamicRecordNative::from_bytes_le(bytes).map(Self)
    }

    /// Converts this DynamicRecord back to a RecordPlaintext.
    fn to_record(&self, owner_is_private: bool) -> anyhow::Result<RecordPlaintext> {
        self.0
            .to_record(owner_is_private)
            .map(RecordPlaintext::from)
    }

    /// Serializes the dynamic record to a little-endian byte array.
    fn bytes(&self) -> anyhow::Result<Vec<u8>> {
        self.0.to_bytes_le()
    }

    /// Returns the dynamic record as a vector of field elements.
    fn to_fields(&self) -> anyhow::Result<Vec<Field>> {
        self.0
            .to_fields()
            .map(|fs| fs.into_iter().map(Field::from).collect())
    }

    /// Returns the dynamic record as a little-endian bit vector.
    fn to_bits_le(&self) -> Vec<bool> {
        self.0.to_bits_le()
    }

    /// Returns the owner address of the dynamic record.
    #[getter]
    fn owner(&self) -> Address {
        Address::from(*self.0.owner())
    }

    /// Returns the Merkle root of the record data as a Field.
    #[getter]
    fn root(&self) -> Field {
        Field::from(*self.0.root())
    }

    /// Returns the nonce of the record as a Group.
    #[getter]
    fn nonce(&self) -> Group {
        Group::from(*self.0.nonce())
    }

    /// Returns true if the dynamic record is a hiding variant (version != 0).
    fn is_hiding(&self) -> bool {
        self.0.is_hiding()
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    #[classattr]
    const __hash__: Option<PyObject> = None;
}

impl Deref for DynamicRecord {
    type Target = DynamicRecordNative;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<DynamicRecordNative> for DynamicRecord {
    fn from(native: DynamicRecordNative) -> Self {
        Self(native)
    }
}

impl From<DynamicRecord> for DynamicRecordNative {
    fn from(dr: DynamicRecord) -> Self {
        dr.0
    }
}
