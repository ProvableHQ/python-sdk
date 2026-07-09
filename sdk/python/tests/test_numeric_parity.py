"""
Wasm parity tests for integer, boolean, field, group, and scalar types.

Verifies:
  - Checked overflow raises OverflowError
  - Wrapped arithmetic wraps correctly
  - dunder ≡ named method equivalence
  - to_field/from_field round-trips
  - from_field_lossy truncation (pinned KATs from wasm test suite)
  - Cross-cast pinned values
  - Logic ops truth tables for Boolean
  - Conversions from Boolean to numeric types
  - New methods on Field, Group, Scalar
"""

import pytest
from aleo.mainnet import (
    Boolean,
    Field,
    Group,
    I8,
    I16,
    I32,
    I64,
    I128,
    U8,
    U16,
    U32,
    U64,
    U128,
    Scalar,
)


# ═══════════════════════════════════════════════════════════════
# Integer — U8
# ═══════════════════════════════════════════════════════════════


class TestU8:
    def test_checked_overflow_raises(self):
        with pytest.raises(OverflowError):
            U8(255) + U8(1)

    def test_dunder_add_eq_add(self):
        a, b = U8(10), U8(20)
        assert a + b == a.add(b)

    def test_checked_sub_underflow_raises(self):
        with pytest.raises(OverflowError):
            U8(0) - U8(1)

    def test_dunder_sub_eq_subtract(self):
        a, b = U8(30), U8(10)
        assert a - b == a.subtract(b)

    def test_checked_mul_overflow_raises(self):
        with pytest.raises(OverflowError):
            U8(200) * U8(2)

    def test_dunder_mul_eq_multiply(self):
        a, b = U8(3), U8(4)
        assert a * b == a.multiply(b)

    def test_div_zero_raises(self):
        with pytest.raises(ZeroDivisionError):
            U8(5) // U8(0)

    def test_floordiv_eq_divide(self):
        a, b = U8(10), U8(3)
        assert a // b == a.divide(b)

    def test_add_wrapped(self):
        assert U8(255).add_wrapped(U8(1)) == U8(0)

    def test_sub_wrapped(self):
        assert U8(0).sub_wrapped(U8(1)) == U8(255)

    def test_mul_wrapped(self):
        assert U8(200).mul_wrapped(U8(2)) == U8(144)  # 400 % 256 = 144

    def test_div_wrapped(self):
        assert U8(10).div_wrapped(U8(3)) == U8(3)

    def test_rem(self):
        assert U8(10).rem(U8(3)) == U8(1)

    def test_rem_zero_divisor_raises(self):
        with pytest.raises(ZeroDivisionError):
            U8(10).rem(U8(0))

    def test_rem_wrapped(self):
        assert U8(10).rem_wrapped(U8(3)) == U8(1)

    def test_rem_wrapped_zero_divisor_raises(self):
        with pytest.raises(ZeroDivisionError):
            U8(10).rem_wrapped(U8(0))

    def test_pow_u8(self):
        # 2^8 = 256 overflows u8
        with pytest.raises(OverflowError):
            U8(2).pow_u8(U8(8))
        assert U8(2).pow_u8(U8(7)) == U8(128)

    def test_pow_u32_dunder(self):
        assert U8(2) ** U32(3) == U8(8)

    def test_to_field_from_field_roundtrip(self):
        v = U8(200)
        f = v.to_field()
        back = U8.from_field(f)
        assert back == v

    def test_from_field_lossy_truncation(self):
        # KAT from wasm: field(256).to_u8_lossy() == 0u8
        f256 = Field.from_string("256field")
        assert U8.from_field_lossy(f256) == U8(0)
        # KAT: field(257).to_u8_lossy() == 1u8
        f257 = Field.from_string("257field")
        assert U8.from_field_lossy(f257) == U8(1)

    def test_to_scalar(self):
        s = U8(42).to_scalar()
        assert str(s) == "42scalar"

    def test_to_boolean_lossy_even(self):
        assert bool(U8(4).to_boolean_lossy()) is False

    def test_to_boolean_lossy_odd(self):
        assert bool(U8(5).to_boolean_lossy()) is True

    def test_from_bits_le_roundtrip(self):
        v = U8(42)
        bits = v.to_bits_le()
        assert U8.from_bits_le(bits) == v

    def test_from_bytes_le_roundtrip(self):
        v = U8(200)
        bts = v.to_bytes_le()
        assert U8.from_bytes_le(bts) == v

    def test_one(self):
        assert U8.one() == U8(1)

    def test_random_produces_values(self):
        # just sanity-check that it returns an U8
        r = U8.random()
        assert isinstance(r, U8)


# ═══════════════════════════════════════════════════════════════
# Integer — U64 (64-bit boundary tests)
# ═══════════════════════════════════════════════════════════════


class TestU64:
    def test_overflow_raises(self):
        with pytest.raises(OverflowError):
            U64(2**64 - 1) + U64(1)

    def test_add_wrapped(self):
        assert U64(2**64 - 1).add_wrapped(U64(1)) == U64(0)

    def test_to_field_from_field(self):
        v = U64(12345678901234)
        assert U64.from_field(v.to_field()) == v


# ═══════════════════════════════════════════════════════════════
# Integer — U128 (boundary, cross-cast)
# ═══════════════════════════════════════════════════════════════


class TestU128:
    def test_overflow_raises(self):
        with pytest.raises(OverflowError):
            U128(2**128 - 1) + U128(1)

    def test_add_wrapped(self):
        assert U128(2**128 - 1).add_wrapped(U128(1)) == U128(0)

    def test_cross_cast_identity(self):
        # KAT from wasm: U32(42).to_u32_lossy() == U32(42)
        v = U32(42)
        assert v.to_u32_lossy() == v

    def test_cross_cast_widening(self):
        # KAT from wasm: U8(255).to_u32_lossy() == U32(255)
        assert U8(255).to_u32_lossy() == U32(255)

    def test_cross_cast_narrowing(self):
        # KAT from wasm: U32(256).to_u8_lossy() == U8(0)
        assert U32(256).to_u8_lossy() == U8(0)
        # KAT from wasm: U32(257).to_u8_lossy() == U8(1)
        assert U32(257).to_u8_lossy() == U8(1)

    def test_cross_cast_signed_unsigned(self):
        # KAT from wasm: I32(42).to_u32_lossy() == U32(42)
        assert I32(42).to_u32_lossy() == U32(42)


# ═══════════════════════════════════════════════════════════════
# Integer — I8 (signed)
# ═══════════════════════════════════════════════════════════════


class TestI8:
    def test_negate_dunder_eq_negate(self):
        v = I8(10)
        assert -v == v.negate()

    def test_negate(self):
        assert I8(10).negate() == I8(-10)

    def test_negate_min_raises(self):
        with pytest.raises(OverflowError):
            I8(-128).negate()

    def test_abs_checked_normal(self):
        assert I8(-5).abs_checked() == I8(5)

    def test_abs_checked_min_raises(self):
        with pytest.raises(OverflowError):
            I8(-128).abs_checked()

    def test_abs_wrapped_min(self):
        assert I8(-128).abs_wrapped() == I8(-128)  # MIN stays MIN

    def test_overflow_add(self):
        with pytest.raises(OverflowError):
            I8(127) + I8(1)

    def test_dunder_eq_named(self):
        a, b = I8(10), I8(5)
        assert a + b == a.add(b)
        assert a - b == a.subtract(b)
        assert a * b == a.multiply(b)

    def test_to_field_from_field(self):
        v = I8(-42)
        assert I8.from_field(v.to_field()) == v

    def test_rem_zero_divisor_raises(self):
        with pytest.raises(ZeroDivisionError):
            I8(10).rem(I8(0))

    def test_rem_wrapped_zero_divisor_raises(self):
        with pytest.raises(ZeroDivisionError):
            I8(10).rem_wrapped(I8(0))

    def test_rem_min_neg1_raises_overflow(self):
        # i8::MIN % -1 overflows (checked_rem returns None); should raise OverflowError
        with pytest.raises(OverflowError):
            I8(-128).rem(I8(-1))

    def test_rem_wrapped_min_neg1(self):
        # rem_wrapped uses wrapping_rem: i8::MIN wrapping_rem -1 == 0 in Rust stdlib
        assert I8(-128).rem_wrapped(I8(-1)) == I8(0)


# ═══════════════════════════════════════════════════════════════
# Integer — I128 (signed, large)
# ═══════════════════════════════════════════════════════════════


class TestI128:
    def test_overflow_add(self):
        with pytest.raises(OverflowError):
            I128(2**127 - 1) + I128(1)

    def test_underflow_sub(self):
        with pytest.raises(OverflowError):
            I128(-(2**127)) - I128(1)

    def test_negate_min_raises(self):
        with pytest.raises(OverflowError):
            I128(-(2**127)).negate()

    def test_add_wrapped(self):
        assert I128(2**127 - 1).add_wrapped(I128(1)) == I128(-(2**127))

    def test_to_scalar(self):
        v = I128(1000)
        s = v.to_scalar()
        assert "1000" in str(s)

    def test_to_plaintext_roundtrip(self):
        v = I128(9999)
        pt = v.to_plaintext()
        assert pt is not None
        assert "9999" in str(pt)


# ═══════════════════════════════════════════════════════════════
# Boolean
# ═══════════════════════════════════════════════════════════════


class TestBoolean:
    # Truth table for AND
    @pytest.mark.parametrize(
        "a,b,expected",
        [(False, False, False), (False, True, False), (True, False, False), (True, True, True)],
    )
    def test_and(self, a, b, expected):
        assert bool(Boolean(a).and_(Boolean(b))) == expected

    @pytest.mark.parametrize(
        "a,b,expected",
        [(False, False, False), (False, True, False), (True, False, False), (True, True, True)],
    )
    def test_and_dunder(self, a, b, expected):
        assert bool(Boolean(a) & Boolean(b)) == expected

    @pytest.mark.parametrize(
        "a,b,expected",
        [(False, False, False), (False, True, True), (True, False, True), (True, True, True)],
    )
    def test_or(self, a, b, expected):
        assert bool(Boolean(a).or_(Boolean(b))) == expected

    @pytest.mark.parametrize(
        "a,b,expected",
        [(False, False, False), (False, True, True), (True, False, True), (True, True, True)],
    )
    def test_or_dunder(self, a, b, expected):
        assert bool(Boolean(a) | Boolean(b)) == expected

    @pytest.mark.parametrize(
        "a,b,expected",
        [(False, False, False), (False, True, True), (True, False, True), (True, True, False)],
    )
    def test_xor(self, a, b, expected):
        assert bool(Boolean(a).xor(Boolean(b))) == expected

    @pytest.mark.parametrize(
        "a,b,expected",
        [(False, False, False), (False, True, True), (True, False, True), (True, True, False)],
    )
    def test_xor_dunder(self, a, b, expected):
        assert bool(Boolean(a) ^ Boolean(b)) == expected

    @pytest.mark.parametrize(
        "a,b,expected",
        [(False, False, True), (False, True, True), (True, False, True), (True, True, False)],
    )
    def test_nand(self, a, b, expected):
        assert bool(Boolean(a).nand(Boolean(b))) == expected

    @pytest.mark.parametrize(
        "a,b,expected",
        [(False, False, True), (False, True, False), (True, False, False), (True, True, False)],
    )
    def test_nor(self, a, b, expected):
        assert bool(Boolean(a).nor(Boolean(b))) == expected

    def test_not(self):
        assert bool(Boolean(True).not_()) is False
        assert bool(Boolean(False).not_()) is True

    def test_not_dunder(self):
        assert bool(~Boolean(True)) is False
        assert bool(~Boolean(False)) is True

    def test_dunder_not_eq_not_(self):
        b = Boolean(True)
        assert ~b == b.not_()

    def test_dunder_and_eq_and_(self):
        a, b = Boolean(True), Boolean(False)
        assert (a & b) == a.and_(b)

    def test_dunder_or_eq_or_(self):
        a, b = Boolean(True), Boolean(False)
        assert (a | b) == a.or_(b)

    def test_dunder_xor_eq_xor(self):
        a, b = Boolean(True), Boolean(False)
        assert (a ^ b) == a.xor(b)

    def test_to_field_false(self):
        f = Boolean(False).to_field()
        assert str(f) == "0field"

    def test_to_field_true(self):
        f = Boolean(True).to_field()
        assert str(f) == "1field"

    def test_to_scalar_false(self):
        s = Boolean(False).to_scalar()
        assert str(s) == "0scalar"

    def test_to_scalar_true(self):
        s = Boolean(True).to_scalar()
        assert str(s) == "1scalar"

    def test_to_u8_false(self):
        assert Boolean(False).to_u8() == U8(0)

    def test_to_u8_true(self):
        assert Boolean(True).to_u8() == U8(1)

    def test_to_i8_false(self):
        assert Boolean(False).to_i8() == I8(0)

    def test_to_i8_true(self):
        assert Boolean(True).to_i8() == I8(1)

    def test_to_u128(self):
        assert Boolean(True).to_u128() == U128(1)

    def test_to_i128(self):
        assert Boolean(True).to_i128() == I128(1)

    def test_to_plaintext(self):
        pt = Boolean(True).to_plaintext()
        assert "true" in str(pt)

    def test_from_string(self):
        assert bool(Boolean.from_string("true")) is True
        assert bool(Boolean.from_string("false")) is False

    def test_from_bits_le_roundtrip(self):
        for v in [True, False]:
            b = Boolean(v)
            bits = b.to_bits_le()
            assert Boolean.from_bits_le(bits) == b

    def test_from_bytes_le_roundtrip(self):
        for v in [True, False]:
            b = Boolean(v)
            bts = b.to_bytes_le()
            assert Boolean.from_bytes_le(bts) == b

    def test_random(self):
        r = Boolean.random()
        assert isinstance(r, Boolean)


# ═══════════════════════════════════════════════════════════════
# Field (new methods)
# ═══════════════════════════════════════════════════════════════


class TestFieldNewMethods:
    def test_to_boolean_strict_zero(self):
        b = Field.from_string("0field").to_boolean()
        assert bool(b) is False

    def test_to_boolean_strict_one(self):
        b = Field.from_string("1field").to_boolean()
        assert bool(b) is True

    def test_to_boolean_strict_non_binary_raises(self):
        with pytest.raises(Exception):
            Field.from_string("2field").to_boolean()

    def test_to_boolean_lossy(self):
        # KAT from wasm field tests
        assert bool(Field.from_string("0field").to_boolean_lossy()) is False
        assert bool(Field.from_string("1field").to_boolean_lossy()) is True
        assert bool(Field.from_string("2field").to_boolean_lossy()) is False
        assert bool(Field.from_string("3field").to_boolean_lossy()) is True

    def test_to_group_lossy_never_fails(self):
        for s in ["0field", "1field", "12345field"]:
            g = Field.from_string(s).to_group_lossy()
            assert isinstance(g, Group)

    def test_to_address_lossy(self):
        from aleo.mainnet import Address
        addr = Field.from_string("42field").to_address_lossy()
        assert str(addr).startswith("aleo1")

    def test_to_plaintext(self):
        f = Field.from_string("42field")
        pt = f.to_plaintext()
        assert "42field" in str(pt)

    def test_to_u8_lossy_kats(self):
        # KAT from wasm field tests
        assert Field.from_string("255field").to_u8_lossy() == U8(255)
        assert Field.from_string("256field").to_u8_lossy() == U8(0)
        assert Field.from_string("257field").to_u8_lossy() == U8(1)

    def test_to_u32_lossy(self):
        assert Field.from_string("255field").to_u32_lossy() == U32(255)

    def test_to_i8_lossy(self):
        f = Field.from_string("127field")
        assert Field.from_string("127field").to_i8_lossy() == I8(127)

    def test_random_distinct(self):
        a, b = Field.random(), Field.random()
        assert isinstance(a, Field)
        assert isinstance(b, Field)
        # Astronomically unlikely to collide over BLS12-377 field
        assert a != b


# ═══════════════════════════════════════════════════════════════
# Group (new methods)
# ═══════════════════════════════════════════════════════════════


class TestGroupNewMethods:
    def test_random(self):
        g = Group.random()
        assert isinstance(g, Group)

    def test_to_field_eq_x_coordinate(self):
        g = Group.generator()
        assert g.to_field() == g.to_x_coordinate()

    def test_to_scalar_lossy(self):
        g = Group.generator()
        s = g.to_scalar_lossy()
        assert isinstance(s, Scalar)

    def test_to_boolean_lossy(self):
        g = Group.generator()
        b = g.to_boolean_lossy()
        assert isinstance(b, Boolean)

    def test_to_plaintext(self):
        g = Group.generator()
        pt = g.to_plaintext()
        assert "group" in str(pt).lower()

    def test_to_fields(self):
        g = Group.generator()
        fields = g.to_fields()
        assert len(fields) >= 1
        assert isinstance(fields[0], Field)

    def test_to_u8_lossy(self):
        g = Group.generator()
        v = g.to_u8_lossy()
        assert isinstance(v, U8)

    def test_to_bits_le_from_bits_le_roundtrip(self):
        g = Group.generator()
        bits = g.to_bits_le()
        back = Group.from_bits_le(bits)
        assert back == g

    def test_to_address(self):
        from aleo.mainnet import Address
        g = Group.generator()
        addr = g.to_address()
        assert str(addr).startswith("aleo1")


# ═══════════════════════════════════════════════════════════════
# Scalar (new methods)
# ═══════════════════════════════════════════════════════════════


class TestScalarNewMethods:
    def test_random(self):
        s = Scalar.random()
        assert isinstance(s, Scalar)

    def test_to_plaintext(self):
        s = Scalar.from_string("42scalar")
        pt = s.to_plaintext()
        assert "42scalar" in str(pt)

    def test_to_boolean_strict_zero(self):
        b = Scalar.zero().to_boolean()
        assert bool(b) is False

    def test_to_boolean_strict_one(self):
        b = Scalar.one().to_boolean()
        assert bool(b) is True

    def test_to_boolean_strict_other_raises(self):
        with pytest.raises(Exception):
            Scalar.from_string("42scalar").to_boolean()

    def test_to_boolean_lossy(self):
        assert bool(Scalar.zero().to_boolean_lossy()) is False
        assert bool(Scalar.one().to_boolean_lossy()) is True

    def test_to_bits_le_roundtrip(self):
        s = Scalar.from_string("42scalar")
        bits = s.to_bits_le()
        back = Scalar.from_bits_le(bits)
        assert back == s

    def test_double(self):
        s = Scalar.one()
        d = s.double()
        assert str(d) == str(s + s)

    def test_to_u8_lossy(self):
        s = Scalar.from_string("200scalar")
        assert s.to_u8_lossy() == U8(200)

    def test_to_u32_lossy(self):
        s = Scalar.from_string("42scalar")
        assert s.to_u32_lossy() == U32(42)

    def test_to_i8_lossy(self):
        s = Scalar.from_string("42scalar")
        # 42 fits in i8
        assert s.to_i8_lossy() == I8(42)

    def test_to_group_lossy(self):
        s = Scalar.from_string("1scalar")
        g = s.to_group_lossy()
        assert isinstance(g, Group)

    def test_to_field_roundtrip(self):
        # KAT from wasm scalar tests
        s = Scalar.from_string("42scalar")
        f = s.to_field()
        back = f.to_scalar_lossy()
        assert str(back) == str(s)


# ═══════════════════════════════════════════════════════════════
# Cross-type round-trip: integer → field → integer
# ═══════════════════════════════════════════════════════════════


class TestCrossTypeRoundtrips:
    def test_u8_to_field_roundtrip(self):
        v = U8(200)
        assert U8.from_field(v.to_field()) == v

    def test_u16_to_field_roundtrip(self):
        v = U16(60000)
        assert U16.from_field(v.to_field()) == v

    def test_u32_to_field_roundtrip(self):
        # KAT from wasm integer tests
        v = U32.from_string("42u32")
        f = v.to_field()
        back = U32.from_field(f)
        assert back == v

    def test_i8_to_field_roundtrip(self):
        v = I8(-100)
        assert I8.from_field(v.to_field()) == v

    def test_i128_to_field_roundtrip(self):
        v = I128(-(2**63))
        assert I128.from_field(v.to_field()) == v

    def test_u8_from_field_lossy_in_range(self):
        # KAT from wasm: in-range round-trips losslessly
        v = U8.from_string("200u8")
        f = v.to_field()
        back = U8.from_field_lossy(f)
        assert back == v

    def test_to_boolean_lossy_u32(self):
        # KATs from wasm tests
        assert bool(U32.from_string("0u32").to_boolean_lossy()) is False
        assert bool(U32.from_string("1u32").to_boolean_lossy()) is True
        assert bool(U32.from_string("4u32").to_boolean_lossy()) is False
        assert bool(U32.from_string("5u32").to_boolean_lossy()) is True
