"""
Tests for Plaintext construction, classification, encrypt/decrypt, and
related Literal, Ciphertext, and Value types.
"""

import pytest
from aleo.mainnet import (
    Address,
    Boolean,
    Ciphertext,
    Field,
    Group,
    I128,
    Identifier,
    Literal,
    Plaintext,
    Scalar,
    Signature,
    U8,
    U64,
    Value,
)


# ---------------------------------------------------------------------------
# Literal constructors and type_name
# ---------------------------------------------------------------------------

class TestLiteral:
    def test_from_field(self):
        lit = Literal.from_field(Field.from_string("42field"))
        assert lit.type_name() == "field"

    def test_from_address(self):
        addr = Address.from_string(
            "aleo1rhgdu77hgyqd3xjj8ucu3jj9r2krwz6mnzyd80gncr5fxcwlh5rsvzp9px"
        )
        lit = Literal.from_address(addr)
        assert lit.type_name() == "address"

    def test_from_boolean(self):
        lit = Literal.from_boolean(Boolean(True))
        assert lit.type_name() == "boolean"

    def test_from_u8(self):
        lit = Literal.from_u8(U8(255))
        assert lit.type_name() == "u8"

    def test_from_u64(self):
        lit = Literal.from_u64(U64(1_000_000))
        assert lit.type_name() == "u64"

    def test_from_i128(self):
        lit = Literal.from_i128(I128(-1))
        assert lit.type_name() == "i128"

    def test_from_scalar(self):
        lit = Literal.from_scalar(Scalar.zero())
        assert lit.type_name() == "scalar"

    def test_from_signature(self):
        sig = Signature.from_string(
            "sign1lcpxtgqkp238x45fk79lkx5xz7sx37f56wl0hyemhv78dgzxyspykg6u26l"
            "x2a02tvat6zaflx530qtnme34gh702wclwr20rdxrsqcl7shvwsyhygt2yvkgzeq7"
            "zz2rdat4rrsr0cd9kwm6jddjcs9lps8s80v35rwvtkgg2gxprf4dge0tcet3pe7nf"
            "xupkvfuvh3sw2gpyv0km46"
        )
        lit = Literal.from_signature(sig)
        assert lit.type_name() == "signature"

    def test_parse_u64(self):
        lit = Literal.parse("99u64")
        assert lit.type_name() == "u64"

    def test_parse_address(self):
        lit = Literal.parse(
            "aleo1rhgdu77hgyqd3xjj8ucu3jj9r2krwz6mnzyd80gncr5fxcwlh5rsvzp9px"
        )
        assert lit.type_name() == "address"


# ---------------------------------------------------------------------------
# Plaintext: new_literal / is_literal / as_literal
# ---------------------------------------------------------------------------

class TestPlaintextLiteral:
    def test_new_literal_classification(self):
        p = Plaintext.new_literal(Literal.from_field(Field.from_string("1field")))
        assert p.is_literal() is True
        assert p.is_struct() is False
        assert p.is_array() is False

    def test_as_literal_round_trip(self):
        lit = Literal.from_field(Field.from_string("77field"))
        p = Plaintext.new_literal(lit)
        lit2 = p.as_literal()
        assert lit2.type_name() == "field"

    def test_from_string_literal(self):
        p = Plaintext.from_string("42u64")
        assert p.is_literal() is True


# ---------------------------------------------------------------------------
# Plaintext: new_struct / is_struct / as_struct
# ---------------------------------------------------------------------------

class TestPlaintextStruct:
    def _make_struct(self):
        k1 = Identifier.from_string("x")
        v1 = Plaintext.new_literal(Literal.from_field(Field.from_string("1field")))
        k2 = Identifier.from_string("y")
        v2 = Plaintext.new_literal(Literal.from_u64(U64(99)))
        return Plaintext.new_struct([(k1, v1), (k2, v2)])

    def test_new_struct_classification(self):
        ps = self._make_struct()
        assert ps.is_struct() is True
        assert ps.is_literal() is False
        assert ps.is_array() is False

    def test_as_struct_has_correct_members(self):
        ps = self._make_struct()
        members = ps.as_struct()
        assert len(members) == 2

    def test_struct_singleton(self):
        k = Identifier.from_string("a")
        v = Plaintext.new_literal(Literal.from_boolean(Boolean(False)))
        ps = Plaintext.new_struct([(k, v)])
        assert ps.is_struct() is True
        assert len(ps.as_struct()) == 1


# ---------------------------------------------------------------------------
# Plaintext: new_array / is_array / as_array
# ---------------------------------------------------------------------------

class TestPlaintextArray:
    def test_new_array_classification(self):
        elem = Plaintext.new_literal(Literal.from_u64(U64(0)))
        pa = Plaintext.new_array([elem, elem, elem])
        assert pa.is_array() is True
        assert pa.is_literal() is False
        assert pa.is_struct() is False

    def test_as_array_length(self):
        elems = [
            Plaintext.new_literal(Literal.from_u8(U8(i))) for i in range(4)
        ]
        pa = Plaintext.new_array(elems)
        assert len(pa.as_array()) == 4

    def test_empty_array(self):
        pa = Plaintext.new_array([])
        assert pa.is_array() is True
        assert len(pa.as_array()) == 0


# ---------------------------------------------------------------------------
# Plaintext: encrypt_symmetric / decrypt_symmetric round-trip
# ---------------------------------------------------------------------------

class TestSymmetricEncryption:
    def _key(self):
        return Field.from_string("12345678field")

    def test_literal_round_trip(self):
        key = self._key()
        p = Plaintext.new_literal(Literal.from_field(Field.from_string("99field")))
        ct = p.encrypt_symmetric(key)
        assert isinstance(ct, Ciphertext)
        p2 = ct.decrypt_symmetric(key)
        assert str(p2) == str(p)

    def test_struct_round_trip(self):
        key = self._key()
        k = Identifier.from_string("v")
        v = Plaintext.new_literal(Literal.from_u64(U64(42_000)))
        ps = Plaintext.new_struct([(k, v)])
        ct = ps.encrypt_symmetric(key)
        ps2 = ct.decrypt_symmetric(key)
        assert str(ps2) == str(ps)

    def test_wrong_key_produces_different_plaintext_or_errors(self):
        """Decrypting with the wrong key must not silently return the original plaintext.
        snarkvm may raise a RuntimeError (malformed struct) or return a different value;
        either outcome is correct – the ciphertext must not decrypt to the original."""
        key1 = Field.from_string("1field")
        key2 = Field.from_string("2field")
        p = Plaintext.new_literal(Literal.from_field(Field.from_string("7field")))
        ct = p.encrypt_symmetric(key1)
        try:
            p_wrong = ct.decrypt_symmetric(key2)
            assert str(p_wrong) != str(p)
        except RuntimeError:
            pass  # snarkvm raised an error – wrong key rejected


# ---------------------------------------------------------------------------
# Value
# ---------------------------------------------------------------------------

class TestValue:
    def test_parse_u64(self):
        v = Value.parse("42u64")
        assert "42u64" in str(v)

    def test_parse_address(self):
        addr_str = "aleo1rhgdu77hgyqd3xjj8ucu3jj9r2krwz6mnzyd80gncr5fxcwlh5rsvzp9px"
        v = Value.parse(addr_str)
        assert addr_str in str(v)

    def test_from_literal(self):
        lit = Literal.from_field(Field.from_string("7field"))
        v = Value.from_literal(lit)
        assert "7field" in str(v)
