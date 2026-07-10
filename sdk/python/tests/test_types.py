"""
Tests for SDK primitive and structural types:
  - Integer types (U8, U16, U32, U64, U128, I8, I16, I32, I64, I128)
  - ProgramID, Identifier, Locator
  - Program.from_source
  - Bytes round-trips (Authorization)
"""

import pytest
from aleo.mainnet import (
    Authorization,
    I8,
    I16,
    I32,
    I64,
    I128,
    Identifier,
    Locator,
    Network,
    PrivateKey,
    Process,
    Program,
    ProgramID,
    U8,
    U16,
    U32,
    U64,
    U128,
    Value,
)


# ---------------------------------------------------------------------------
# Integer types
# ---------------------------------------------------------------------------

class TestUnsignedIntegers:
    def test_u8_construction_and_int(self):
        assert int(U8(0)) == 0
        assert int(U8(255)) == 255

    def test_u8_zero(self):
        assert int(U8.zero()) == 0

    def test_u16_construction(self):
        assert int(U16(65535)) == 65535
        assert int(U16.zero()) == 0

    def test_u32_construction(self):
        assert int(U32(4_294_967_295)) == 4_294_967_295
        assert int(U32.zero()) == 0

    def test_u64_construction(self):
        assert int(U64(1_000_000)) == 1_000_000
        assert int(U64.zero()) == 0

    def test_u128_construction(self):
        large = (1 << 64) + 1
        assert int(U128(large)) == large
        assert int(U128.zero()) == 0


class TestSignedIntegers:
    def test_i8_construction_and_int(self):
        assert int(I8(-128)) == -128
        assert int(I8(127)) == 127

    def test_i8_zero(self):
        assert int(I8.zero()) == 0

    def test_i16_construction(self):
        assert int(I16(-32768)) == -32768
        assert int(I16.zero()) == 0

    def test_i32_construction(self):
        assert int(I32(-1)) == -1
        assert int(I32.zero()) == 0

    def test_i64_construction(self):
        assert int(I64(-(1 << 62))) == -(1 << 62)
        assert int(I64.zero()) == 0

    def test_i128_construction(self):
        assert int(I128(-999_999_999)) == -999_999_999
        assert int(I128.zero()) == 0


# ---------------------------------------------------------------------------
# ProgramID
# ---------------------------------------------------------------------------

class TestProgramID:
    def test_credits_parse(self):
        pid = ProgramID.from_string("credits.aleo")
        assert str(pid.name) == "credits"
        assert str(pid.network) == "aleo"

    def test_str_round_trip(self):
        pid = ProgramID.from_string("hello.aleo")
        assert str(pid) == "hello.aleo"

    def test_is_aleo(self):
        pid = ProgramID.from_string("credits.aleo")
        assert pid.is_aleo() is True

    def test_hashable_as_dict_key(self):
        pid1 = ProgramID.from_string("credits.aleo")
        pid2 = ProgramID.from_string("credits.aleo")
        d = {pid1: "ok"}
        assert pid2 in d

    def test_equality(self):
        pid1 = ProgramID.from_string("credits.aleo")
        pid2 = ProgramID.from_string("credits.aleo")
        assert pid1 == pid2

    def test_inequality(self):
        pid1 = ProgramID.from_string("credits.aleo")
        pid2 = ProgramID.from_string("token.aleo")
        assert pid1 != pid2


# ---------------------------------------------------------------------------
# Identifier
# ---------------------------------------------------------------------------

class TestIdentifier:
    def test_from_string_round_trip(self):
        ident = Identifier.from_string("transfer_public")
        assert str(ident) == "transfer_public"

    def test_equality(self):
        a = Identifier.from_string("noop")
        b = Identifier.from_string("noop")
        assert a == b

    def test_inequality(self):
        a = Identifier.from_string("foo")
        b = Identifier.from_string("bar")
        assert a != b

    def test_hashable(self):
        a = Identifier.from_string("mint")
        b = Identifier.from_string("mint")
        s = {a}
        assert b in s


# ---------------------------------------------------------------------------
# Locator
# ---------------------------------------------------------------------------

class TestLocator:
    def test_from_string_properties(self):
        loc = Locator.from_string("credits.aleo/transfer_public")
        assert str(loc.program_id) == "credits.aleo"
        assert str(loc.resource) == "transfer_public"
        assert str(loc.name) == "credits"
        assert str(loc.network) == "aleo"

    def test_constructor(self):
        pid = ProgramID.from_string("credits.aleo")
        res = Identifier.from_string("fee_public")
        loc = Locator(pid, res)
        assert str(loc.resource) == "fee_public"

    def test_str_round_trip(self):
        loc = Locator.from_string("credits.aleo/transfer_public")
        assert str(loc) == "credits.aleo/transfer_public"


# ---------------------------------------------------------------------------
# Program.from_source
# ---------------------------------------------------------------------------

_NOOP_SRC = "program test.aleo;\nfunction noop:\n"
_HELLO_SRC = (
    "program hellothere.aleo;\n"
    "\n"
    "function hello:\n"
    "    input r0 as u32.public;\n"
    "    input r1 as u32.private;\n"
    "    add r0 r1 into r2;\n"
    "    output r2 as u32.private;\n"
)


class TestProgram:
    def test_noop_program_parses(self):
        prog = Program.from_source(_NOOP_SRC)
        assert str(prog.id) == "test.aleo"

    def test_noop_function_list(self):
        prog = Program.from_source(_NOOP_SRC)
        assert Identifier.from_string("noop") in prog.functions

    def test_noop_source_round_trip(self):
        prog = Program.from_source(_NOOP_SRC)
        assert "program test.aleo" in prog.source

    def test_hello_program_from_ts_sdk(self):
        """hellothere.aleo – exact program from TS SDK account-data.ts helloProgram."""
        prog = Program.from_source(_HELLO_SRC)
        assert str(prog.id) == "hellothere.aleo"
        assert Identifier.from_string("hello") in prog.functions

    def test_credits_is_builtin(self):
        prog = Program.credits()
        assert str(prog.id) == "credits.aleo"
        assert len(prog.functions) > 0

    def test_network_name_and_id(self):
        # Network.name() returns the full display name (e.g. "Aleo Mainnet (v0)");
        # Network.id() is 0 for MainnetV0.
        assert "mainnet" in Network.name().lower() or "aleo" in Network.name().lower()
        assert Network.id() == 0


# ---------------------------------------------------------------------------
# Bytes round-trips
# ---------------------------------------------------------------------------

class TestBytesRoundTrips:
    """bytes() → from_bytes() for Authorization."""

    def _make_auth(self):
        process = Process.load()
        pk = PrivateKey.random()
        return process.authorize(
            pk,
            ProgramID.from_string("credits.aleo"),
            Identifier.from_string("transfer_public"),
            [Value.parse(str(pk.address)), Value.parse("10u64")],
        )

    def test_authorization_bytes_round_trip(self):
        auth = self._make_auth()
        raw = auth.bytes()
        assert isinstance(raw, list)
        auth2 = Authorization.from_bytes(bytes(raw))
        assert auth.to_json() == auth2.to_json()
