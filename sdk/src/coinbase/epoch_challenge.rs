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

use crate::types::EpochChallengeNative;

use pyo3::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use std::ops::Deref;

#[pyclass(frozen)]
pub struct EpochChallenge(EpochChallengeNative);

impl Serialize for EpochChallenge {
    /// Serializes the solutions to a JSON-string or buffer.
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        use snarkvm::prelude::ToBytesSerializer;
        match serializer.is_human_readable() {
            true => {
                let mut s = serializer.serialize_struct("EpochChallenge", 1)?;
                s.serialize_field("epoch_number", &self.0.epoch_number())?;
                s.serialize_field("epoch_block_hash", &self.0.epoch_block_hash())?;
                s.serialize_field("degree", &self.0.degree())?;
                s.end()
            }
            false => ToBytesSerializer::serialize_with_size_encoding(&self.0, serializer),
        }
    }
}

impl<'de> Deserialize<'de> for EpochChallenge {
    /// Deserializes the partial solution from a JSON-string or buffer.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use snarkvm::prelude::FromBytesDeserializer;
        use snarkvm::utilities::DeserializeExt;

        let native = match deserializer.is_human_readable() {
            true => {
                let mut v = serde_json::Value::deserialize(deserializer)?;
                EpochChallengeNative::new(
                    DeserializeExt::take_from_value::<D>(&mut v, "epoch_number")?,
                    DeserializeExt::take_from_value::<D>(&mut v, "epoch_block_hash")?,
                    DeserializeExt::take_from_value::<D>(&mut v, "degree")?,
                )
                .map_err(|e| {
                    serde::de::Error::custom(format!("Could not create EpochChallenge: {e}"))
                })?
            }
            false => FromBytesDeserializer::<EpochChallengeNative>::deserialize_with_size_encoding(
                deserializer,
                "epoch challenge",
            )?,
        };
        Ok(Self(native))
    }
}

impl Deref for EpochChallenge {
    type Target = EpochChallengeNative;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[pymethods]
impl EpochChallenge {
    /// Reads in an epoch challenge from a json string.
    #[staticmethod]
    fn from_json(json: String) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(&json)?)
    }

    /// Serialize the given epoch challenge as a JSON string.
    fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string(&self)?)
    }

    fn __str__(&self) -> anyhow::Result<String> {
        self.to_json()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    #[classattr]
    const __hash__: Option<PyObject> = None;
}
