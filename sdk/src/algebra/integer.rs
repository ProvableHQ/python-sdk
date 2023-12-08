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

use crate::types::{
    I128Native, I16Native, I32Native, I64Native, I8Native, U128Native, U16Native, U32Native,
    U64Native, U8Native,
};

use pyo3::prelude::*;
use snarkvm::prelude::Zero;

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

macro_rules! integer {
    ($export_ty:ident, $native:ident, $machine:ident, $desc:literal) => {
        #[doc = concat!("The Aleo ", $desc, " type.")]
        #[pyclass(frozen)]
        #[derive(Copy, Clone)]
        pub struct $export_ty($native);
        #[pymethods]
        impl $export_ty {
            #[new]
            fn new(value: $machine) -> Self {
                Self($native::new(value))
            }

            /// Returns the `0` element of the integer.
            #[staticmethod]
            fn zero() -> Self {
                Self($native::zero())
            }

            /// Returns the integer as a string.
            fn __str__(&self) -> String {
                self.0.to_string()
            }

            fn __int__(&self) -> $machine {
                *self.0
            }

            fn __eq__(&self, other: &Self) -> bool {
                self.0 == other.0
            }

            fn __hash__(&self) -> u64 {
                let mut hasher = DefaultHasher::new();
                self.0.hash(&mut hasher);
                hasher.finish()
            }
        }

        impl From<$export_ty> for $native {
            fn from(value: $export_ty) -> Self {
                value.0
            }
        }
    };
}

integer!(I8, I8Native, i8, "signed I8");
integer!(I16, I16Native, i16, "signed I16");
integer!(I32, I32Native, i32, "signed I32");
integer!(I64, I64Native, i64, "signed I64");
integer!(I128, I128Native, i128, "signed I128");

integer!(U8, U8Native, u8, "unsigned U8");
integer!(U16, U16Native, u16, "unsigned U16");
integer!(U32, U32Native, u32, "unsigned U32");
integer!(U64, U64Native, u64, "unsigned U64");
integer!(U128, U128Native, u128, "unsigned U128");
