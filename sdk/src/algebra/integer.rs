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
    types::{
        BooleanNative, FieldNative, I128Native, I16Native, I32Native, I64Native, I8Native,
        LiteralNative, PlaintextNative, U128Native, U16Native, U32Native, U64Native, U8Native,
    },
    Boolean, Field, Scalar,
};

use pyo3::{
    exceptions::{PyOverflowError, PyZeroDivisionError},
    prelude::*,
};
use rand::rngs::StdRng;
use snarkvm::prelude::{
    AddWrapped, DivWrapped, FromBits, FromBytes, FromField, FromFields, MulWrapped, One,
    RemWrapped, SubWrapped, ToBits, ToBytes, ToField, Uniform, Zero,
};

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    str::FromStr,
    sync::OnceLock,
};

// ── Unsigned integer macro ────────────────────────────────────────────────────
// All methods for unsigned integers in a single #[pymethods] block.

macro_rules! integer_unsigned {
    ($export_ty:ident, $native:ident, $machine:ident, $literal_variant:ident, $desc:literal) => {
        #[doc = concat!("The Aleo ", $desc, " type.")]
        #[pyclass(frozen)]
        #[derive(Copy, Clone)]
        pub struct $export_ty($native);

        #[allow(clippy::wrong_self_convention)]
        #[pymethods]
        impl $export_ty {
            #[new]
            fn new(value: $machine) -> Self {
                Self($native::new(value))
            }

            /// Parses an integer from a string (e.g. "42u32").
            #[staticmethod]
            fn from_string(s: &str) -> anyhow::Result<Self> {
                $native::from_str(s).map(Self)
            }

            /// Returns the `0` element.
            #[staticmethod]
            fn zero() -> Self {
                Self($native::zero())
            }

            /// Returns the `1` element.
            #[staticmethod]
            fn one() -> Self {
                Self($native::one())
            }

            /// Returns a random value.
            #[staticmethod]
            fn random() -> Self {
                Self(<$native as Uniform>::rand(&mut rand::make_rng::<StdRng>()))
            }

            // ── serialization ──────────────────────────────────────────────

            fn to_bytes_le(&self) -> anyhow::Result<Vec<u8>> {
                self.0.to_bytes_le()
            }

            #[staticmethod]
            fn from_bytes_le(bytes: Vec<u8>) -> anyhow::Result<Self> {
                $native::from_bytes_le(&bytes).map(Self)
            }

            fn to_bits_le(&self) -> Vec<bool> {
                self.0.to_bits_le()
            }

            #[staticmethod]
            fn from_bits_le(bits: Vec<bool>) -> anyhow::Result<Self> {
                $native::from_bits_le(&bits).map(Self)
            }

            // ── dunders ────────────────────────────────────────────────────

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

            fn __add__(&self, other: Self) -> PyResult<Self> {
                self.add(other)
            }

            fn __sub__(&self, other: Self) -> PyResult<Self> {
                self.subtract(other)
            }

            fn __mul__(&self, other: Self) -> PyResult<Self> {
                self.multiply(other)
            }

            fn __floordiv__(&self, other: Self) -> PyResult<Self> {
                self.divide(other)
            }

            fn __pow__(&self, exp: U32, _modulo: Option<u32>) -> PyResult<Self> {
                self.pow_u32(exp)
            }

            // ── checked arithmetic ─────────────────────────────────────────

            fn add(&self, other: Self) -> PyResult<Self> {
                (*self.0)
                    .checked_add(*other.0)
                    .map(|v| Self($native::new(v)))
                    .ok_or_else(|| {
                        PyOverflowError::new_err(concat!(
                            stringify!($machine),
                            " addition overflowed"
                        ))
                    })
            }

            fn subtract(&self, other: Self) -> PyResult<Self> {
                (*self.0)
                    .checked_sub(*other.0)
                    .map(|v| Self($native::new(v)))
                    .ok_or_else(|| {
                        PyOverflowError::new_err(concat!(
                            stringify!($machine),
                            " subtraction overflowed"
                        ))
                    })
            }

            fn multiply(&self, other: Self) -> PyResult<Self> {
                (*self.0)
                    .checked_mul(*other.0)
                    .map(|v| Self($native::new(v)))
                    .ok_or_else(|| {
                        PyOverflowError::new_err(concat!(
                            stringify!($machine),
                            " multiplication overflowed"
                        ))
                    })
            }

            fn divide(&self, other: Self) -> PyResult<Self> {
                if *other.0 == 0 {
                    return Err(PyZeroDivisionError::new_err("integer division by zero"));
                }
                (*self.0)
                    .checked_div(*other.0)
                    .map(|v| Self($native::new(v)))
                    .ok_or_else(|| {
                        PyOverflowError::new_err(concat!(
                            stringify!($machine),
                            " division overflowed"
                        ))
                    })
            }

            // ── wrapped ────────────────────────────────────────────────────

            fn add_wrapped(&self, other: Self) -> Self {
                Self(self.0.add_wrapped(&other.0))
            }

            fn sub_wrapped(&self, other: Self) -> Self {
                Self(self.0.sub_wrapped(&other.0))
            }

            fn mul_wrapped(&self, other: Self) -> Self {
                Self(self.0.mul_wrapped(&other.0))
            }

            fn div_wrapped(&self, other: Self) -> Self {
                Self(self.0.div_wrapped(&other.0))
            }

            // ── remainder ─────────────────────────────────────────────────

            fn rem(&self, other: Self) -> PyResult<Self> {
                if *other.0 == 0 {
                    return Err(PyZeroDivisionError::new_err("integer remainder by zero"));
                }
                // For unsigned types, checked_rem only fails on zero divisor (already handled).
                (*self.0)
                    .checked_rem(*other.0)
                    .map(|v| Self($native::new(v)))
                    .ok_or_else(|| {
                        PyOverflowError::new_err(concat!(
                            stringify!($machine),
                            " remainder overflowed"
                        ))
                    })
            }

            fn rem_wrapped(&self, other: Self) -> PyResult<Self> {
                if *other.0 == 0 {
                    return Err(PyZeroDivisionError::new_err("integer remainder by zero"));
                }
                Ok(Self(self.0.rem_wrapped(&other.0)))
            }

            // ── power ──────────────────────────────────────────────────────

            fn pow_u8(&self, exp: U8) -> PyResult<Self> {
                let exp_u32 = *exp.0 as u32;
                (*self.0)
                    .checked_pow(exp_u32)
                    .map(|v| Self($native::new(v)))
                    .ok_or_else(|| {
                        PyOverflowError::new_err(concat!(stringify!($machine), " power overflowed"))
                    })
            }

            fn pow_u16(&self, exp: U16) -> PyResult<Self> {
                let exp_u32 = *exp.0 as u32;
                (*self.0)
                    .checked_pow(exp_u32)
                    .map(|v| Self($native::new(v)))
                    .ok_or_else(|| {
                        PyOverflowError::new_err(concat!(stringify!($machine), " power overflowed"))
                    })
            }

            fn pow_u32(&self, exp: U32) -> PyResult<Self> {
                let exp_u32 = *exp.0;
                (*self.0)
                    .checked_pow(exp_u32)
                    .map(|v| Self($native::new(v)))
                    .ok_or_else(|| {
                        PyOverflowError::new_err(concat!(stringify!($machine), " power overflowed"))
                    })
            }

            // ── conversions ────────────────────────────────────────────────

            fn to_field(&self) -> anyhow::Result<Field> {
                self.0.to_field().map(Into::into)
            }

            #[staticmethod]
            fn from_field(f: &Field) -> anyhow::Result<Self> {
                $native::from_field(&**f).map(Self)
            }

            #[staticmethod]
            fn from_field_lossy(f: &Field) -> Self {
                Self($native::from_field_lossy(&**f))
            }

            #[staticmethod]
            fn from_fields(fields: Vec<Field>) -> anyhow::Result<Self> {
                let native: Vec<FieldNative> = fields.into_iter().map(Into::into).collect();
                $native::from_fields(&native).map(Self)
            }

            fn to_scalar(&self) -> Scalar {
                self.0.to_scalar().into()
            }

            fn to_plaintext(&self) -> crate::Plaintext {
                crate::Plaintext::from(PlaintextNative::Literal(
                    LiteralNative::$literal_variant(self.0),
                    OnceLock::new(),
                ))
            }

            fn to_boolean_lossy(&self) -> Boolean {
                Boolean::from(BooleanNative::new(self.0.to_bits_le()[0]))
            }

            // ── cross-casts ────────────────────────────────────────────────

            fn to_u8_lossy(&self) -> PyResult<U8> {
                let f = self
                    .0
                    .to_field()
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(U8(U8Native::from_field_lossy(&f)))
            }

            fn to_u16_lossy(&self) -> PyResult<U16> {
                let f = self
                    .0
                    .to_field()
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(U16(U16Native::from_field_lossy(&f)))
            }

            fn to_u32_lossy(&self) -> PyResult<U32> {
                let f = self
                    .0
                    .to_field()
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(U32(U32Native::from_field_lossy(&f)))
            }

            fn to_u64_lossy(&self) -> PyResult<U64> {
                let f = self
                    .0
                    .to_field()
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(U64(U64Native::from_field_lossy(&f)))
            }

            fn to_u128_lossy(&self) -> PyResult<U128> {
                let f = self
                    .0
                    .to_field()
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(U128(U128Native::from_field_lossy(&f)))
            }

            fn to_i8_lossy(&self) -> PyResult<I8> {
                let f = self
                    .0
                    .to_field()
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(I8(I8Native::from_field_lossy(&f)))
            }

            fn to_i16_lossy(&self) -> PyResult<I16> {
                let f = self
                    .0
                    .to_field()
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(I16(I16Native::from_field_lossy(&f)))
            }

            fn to_i32_lossy(&self) -> PyResult<I32> {
                let f = self
                    .0
                    .to_field()
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(I32(I32Native::from_field_lossy(&f)))
            }

            fn to_i64_lossy(&self) -> PyResult<I64> {
                let f = self
                    .0
                    .to_field()
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(I64(I64Native::from_field_lossy(&f)))
            }

            fn to_i128_lossy(&self) -> PyResult<I128> {
                let f = self
                    .0
                    .to_field()
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(I128(I128Native::from_field_lossy(&f)))
            }
        }

        impl From<$export_ty> for $native {
            fn from(value: $export_ty) -> Self {
                value.0
            }
        }

        impl From<$native> for $export_ty {
            fn from(value: $native) -> Self {
                Self(value)
            }
        }
    };
}

// ── Signed integer macro ──────────────────────────────────────────────────────
// Identical to unsigned but includes negate/abs_checked/abs_wrapped.

macro_rules! integer_signed {
    ($export_ty:ident, $native:ident, $machine:ident, $literal_variant:ident, $desc:literal) => {
        #[doc = concat!("The Aleo ", $desc, " type.")]
        #[pyclass(frozen)]
        #[derive(Copy, Clone)]
        pub struct $export_ty($native);

        #[allow(clippy::wrong_self_convention)]
        #[pymethods]
        impl $export_ty {
            #[new]
            fn new(value: $machine) -> Self {
                Self($native::new(value))
            }

            #[staticmethod]
            fn from_string(s: &str) -> anyhow::Result<Self> {
                $native::from_str(s).map(Self)
            }

            #[staticmethod]
            fn zero() -> Self {
                Self($native::zero())
            }

            #[staticmethod]
            fn one() -> Self {
                Self($native::one())
            }

            #[staticmethod]
            fn random() -> Self {
                Self(<$native as Uniform>::rand(&mut rand::make_rng::<StdRng>()))
            }

            fn to_bytes_le(&self) -> anyhow::Result<Vec<u8>> {
                self.0.to_bytes_le()
            }

            #[staticmethod]
            fn from_bytes_le(bytes: Vec<u8>) -> anyhow::Result<Self> {
                $native::from_bytes_le(&bytes).map(Self)
            }

            fn to_bits_le(&self) -> Vec<bool> {
                self.0.to_bits_le()
            }

            #[staticmethod]
            fn from_bits_le(bits: Vec<bool>) -> anyhow::Result<Self> {
                $native::from_bits_le(&bits).map(Self)
            }

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

            fn __add__(&self, other: Self) -> PyResult<Self> {
                self.add(other)
            }

            fn __sub__(&self, other: Self) -> PyResult<Self> {
                self.subtract(other)
            }

            fn __mul__(&self, other: Self) -> PyResult<Self> {
                self.multiply(other)
            }

            fn __floordiv__(&self, other: Self) -> PyResult<Self> {
                self.divide(other)
            }

            fn __pow__(&self, exp: U32, _modulo: Option<u32>) -> PyResult<Self> {
                self.pow_u32(exp)
            }

            fn __neg__(&self) -> PyResult<Self> {
                self.negate()
            }

            // ── checked arithmetic ─────────────────────────────────────────

            fn add(&self, other: Self) -> PyResult<Self> {
                (*self.0)
                    .checked_add(*other.0)
                    .map(|v| Self($native::new(v)))
                    .ok_or_else(|| {
                        PyOverflowError::new_err(concat!(
                            stringify!($machine),
                            " addition overflowed"
                        ))
                    })
            }

            fn subtract(&self, other: Self) -> PyResult<Self> {
                (*self.0)
                    .checked_sub(*other.0)
                    .map(|v| Self($native::new(v)))
                    .ok_or_else(|| {
                        PyOverflowError::new_err(concat!(
                            stringify!($machine),
                            " subtraction overflowed"
                        ))
                    })
            }

            fn multiply(&self, other: Self) -> PyResult<Self> {
                (*self.0)
                    .checked_mul(*other.0)
                    .map(|v| Self($native::new(v)))
                    .ok_or_else(|| {
                        PyOverflowError::new_err(concat!(
                            stringify!($machine),
                            " multiplication overflowed"
                        ))
                    })
            }

            fn divide(&self, other: Self) -> PyResult<Self> {
                if *other.0 == 0 {
                    return Err(PyZeroDivisionError::new_err("integer division by zero"));
                }
                (*self.0)
                    .checked_div(*other.0)
                    .map(|v| Self($native::new(v)))
                    .ok_or_else(|| {
                        PyOverflowError::new_err(concat!(
                            stringify!($machine),
                            " division overflowed"
                        ))
                    })
            }

            fn negate(&self) -> PyResult<Self> {
                (*self.0)
                    .checked_neg()
                    .map(|v| Self($native::new(v)))
                    .ok_or_else(|| {
                        PyOverflowError::new_err(concat!(
                            stringify!($machine),
                            " negation overflowed"
                        ))
                    })
            }

            fn abs_checked(&self) -> PyResult<Self> {
                (*self.0)
                    .checked_abs()
                    .map(|v| Self($native::new(v)))
                    .ok_or_else(|| {
                        PyOverflowError::new_err(concat!(stringify!($machine), " abs overflowed"))
                    })
            }

            fn abs_wrapped(&self) -> Self {
                use snarkvm::prelude::AbsWrapped;
                Self(self.0.abs_wrapped())
            }

            // ── wrapped ────────────────────────────────────────────────────

            fn add_wrapped(&self, other: Self) -> Self {
                Self(self.0.add_wrapped(&other.0))
            }

            fn sub_wrapped(&self, other: Self) -> Self {
                Self(self.0.sub_wrapped(&other.0))
            }

            fn mul_wrapped(&self, other: Self) -> Self {
                Self(self.0.mul_wrapped(&other.0))
            }

            fn div_wrapped(&self, other: Self) -> Self {
                Self(self.0.div_wrapped(&other.0))
            }

            // ── remainder ─────────────────────────────────────────────────

            fn rem(&self, other: Self) -> PyResult<Self> {
                if *other.0 == 0 {
                    return Err(PyZeroDivisionError::new_err("integer remainder by zero"));
                }
                // For signed types, checked_rem also returns None for MIN % -1 (overflow).
                (*self.0)
                    .checked_rem(*other.0)
                    .map(|v| Self($native::new(v)))
                    .ok_or_else(|| {
                        PyOverflowError::new_err(concat!(
                            stringify!($machine),
                            " remainder overflowed"
                        ))
                    })
            }

            fn rem_wrapped(&self, other: Self) -> PyResult<Self> {
                if *other.0 == 0 {
                    return Err(PyZeroDivisionError::new_err("integer remainder by zero"));
                }
                Ok(Self(self.0.rem_wrapped(&other.0)))
            }

            // ── power ──────────────────────────────────────────────────────

            fn pow_u8(&self, exp: U8) -> PyResult<Self> {
                let exp_u32 = *exp.0 as u32;
                (*self.0)
                    .checked_pow(exp_u32)
                    .map(|v| Self($native::new(v)))
                    .ok_or_else(|| {
                        PyOverflowError::new_err(concat!(stringify!($machine), " power overflowed"))
                    })
            }

            fn pow_u16(&self, exp: U16) -> PyResult<Self> {
                let exp_u32 = *exp.0 as u32;
                (*self.0)
                    .checked_pow(exp_u32)
                    .map(|v| Self($native::new(v)))
                    .ok_or_else(|| {
                        PyOverflowError::new_err(concat!(stringify!($machine), " power overflowed"))
                    })
            }

            fn pow_u32(&self, exp: U32) -> PyResult<Self> {
                let exp_u32 = *exp.0;
                (*self.0)
                    .checked_pow(exp_u32)
                    .map(|v| Self($native::new(v)))
                    .ok_or_else(|| {
                        PyOverflowError::new_err(concat!(stringify!($machine), " power overflowed"))
                    })
            }

            // ── conversions ────────────────────────────────────────────────

            fn to_field(&self) -> anyhow::Result<Field> {
                self.0.to_field().map(Into::into)
            }

            #[staticmethod]
            fn from_field(f: &Field) -> anyhow::Result<Self> {
                $native::from_field(&**f).map(Self)
            }

            #[staticmethod]
            fn from_field_lossy(f: &Field) -> Self {
                Self($native::from_field_lossy(&**f))
            }

            #[staticmethod]
            fn from_fields(fields: Vec<Field>) -> anyhow::Result<Self> {
                let native: Vec<FieldNative> = fields.into_iter().map(Into::into).collect();
                $native::from_fields(&native).map(Self)
            }

            fn to_scalar(&self) -> Scalar {
                self.0.to_scalar().into()
            }

            fn to_plaintext(&self) -> crate::Plaintext {
                crate::Plaintext::from(PlaintextNative::Literal(
                    LiteralNative::$literal_variant(self.0),
                    OnceLock::new(),
                ))
            }

            fn to_boolean_lossy(&self) -> Boolean {
                Boolean::from(BooleanNative::new(self.0.to_bits_le()[0]))
            }

            // ── cross-casts ────────────────────────────────────────────────

            fn to_u8_lossy(&self) -> PyResult<U8> {
                let f = self
                    .0
                    .to_field()
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(U8(U8Native::from_field_lossy(&f)))
            }

            fn to_u16_lossy(&self) -> PyResult<U16> {
                let f = self
                    .0
                    .to_field()
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(U16(U16Native::from_field_lossy(&f)))
            }

            fn to_u32_lossy(&self) -> PyResult<U32> {
                let f = self
                    .0
                    .to_field()
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(U32(U32Native::from_field_lossy(&f)))
            }

            fn to_u64_lossy(&self) -> PyResult<U64> {
                let f = self
                    .0
                    .to_field()
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(U64(U64Native::from_field_lossy(&f)))
            }

            fn to_u128_lossy(&self) -> PyResult<U128> {
                let f = self
                    .0
                    .to_field()
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(U128(U128Native::from_field_lossy(&f)))
            }

            fn to_i8_lossy(&self) -> PyResult<I8> {
                let f = self
                    .0
                    .to_field()
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(I8(I8Native::from_field_lossy(&f)))
            }

            fn to_i16_lossy(&self) -> PyResult<I16> {
                let f = self
                    .0
                    .to_field()
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(I16(I16Native::from_field_lossy(&f)))
            }

            fn to_i32_lossy(&self) -> PyResult<I32> {
                let f = self
                    .0
                    .to_field()
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(I32(I32Native::from_field_lossy(&f)))
            }

            fn to_i64_lossy(&self) -> PyResult<I64> {
                let f = self
                    .0
                    .to_field()
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(I64(I64Native::from_field_lossy(&f)))
            }

            fn to_i128_lossy(&self) -> PyResult<I128> {
                let f = self
                    .0
                    .to_field()
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(I128(I128Native::from_field_lossy(&f)))
            }
        }

        impl From<$export_ty> for $native {
            fn from(value: $export_ty) -> Self {
                value.0
            }
        }

        impl From<$native> for $export_ty {
            fn from(value: $native) -> Self {
                Self(value)
            }
        }
    };
}

// ── Instantiate unsigned types ─────────────────────────────────────────────────
integer_unsigned!(U8, U8Native, u8, U8, "unsigned U8");
integer_unsigned!(U16, U16Native, u16, U16, "unsigned U16");
integer_unsigned!(U32, U32Native, u32, U32, "unsigned U32");
integer_unsigned!(U64, U64Native, u64, U64, "unsigned U64");
integer_unsigned!(U128, U128Native, u128, U128, "unsigned U128");

// ── Instantiate signed types ───────────────────────────────────────────────────
integer_signed!(I8, I8Native, i8, I8, "signed I8");
integer_signed!(I16, I16Native, i16, I16, "signed I16");
integer_signed!(I32, I32Native, i32, I32, "signed I32");
integer_signed!(I64, I64Native, i64, I64, "signed I64");
integer_signed!(I128, I128Native, i128, I128, "signed I128");
