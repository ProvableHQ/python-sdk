"""
Tests for new Plaintext and Value methods added for W4b parity.

These tests target methods beyond the baseline snarkvm bindings — find(),
bytes(), to_bits_*, to_bytes_raw_*, to_fields*, plaintext_type, to_python(),
Value.from_plaintext(), Value.is_*, Value.value_type, etc.  They act as a
specification / acceptance suite: they are expected to fail until the
corresponding Rust pyo3 surface is wired up.
"""

import pytest
from aleo.mainnet import Field, Plaintext, RecordPlaintext, Value


# ---------------------------------------------------------------------------
# Known-answer strings
# ---------------------------------------------------------------------------

STRUCT = "{\n  microcredits: 100000000u64,\n  height: 1653124u32\n}"
NESTED_STRUCT = (
    "{ player: aleo13nnjqa7h2u4mpl95guz97nhzkhlde750zsjnw59tkgdwc85lyurs295lxc,"
    " health: 100u8,"
    " inventory: { coins: 5u32, snacks: { candies: 5u64, vegetals: 6u64 } },"
    " secret: 2group, cipher: 2scalar, is_alive: true }"
)

PLAINTEXT_LITERAL = "100u64"
PLAINTEXT_STRUCT = "{\n  microcredits: 100000000u64,\n  height: 1653124u32\n}"
RECORD_VALUE = (
    "{ owner: aleo1d5hg2z3ma00382pngntdp68e74zv54jdxy249qhaujhks9c72yrs33ddah.private,"
    " token_amount: 100u64.private,"
    " _nonce: 0group.public }"
)
FUTURE_VALUE = (
    "{\n"
    "  program_id: credits.aleo,\n"
    "  function_name: transfer,\n"
    "  arguments: [\n"
    "    aleo1d5hg2z3ma00382pngntdp68e74zv54jdxy249qhaujhks9c72yrs33ddah,\n"
    "    100000000u64\n"
    "  ]\n"
    "}"
)


# ---------------------------------------------------------------------------
# Plaintext parity
# ---------------------------------------------------------------------------

class TestPlaintextParity:
    def test_find(self):
        p = Plaintext.from_string(STRUCT)
        mc = p.find(["microcredits"])
        assert isinstance(mc, Plaintext)
        assert mc.is_literal()

    def test_find_nested(self):
        p = Plaintext.from_string(NESTED_STRUCT)
        inv = p.find(["inventory"])
        assert isinstance(inv, Plaintext)
        assert inv.is_struct()

    def test_find_not_found(self):
        p = Plaintext.from_string(STRUCT)
        with pytest.raises(Exception):
            p.find(["nonexistent"])

    def test_bytes_roundtrip(self):
        p = Plaintext.from_string(STRUCT)
        b = p.bytes()
        p2 = Plaintext.from_bytes(b)
        assert p == p2

    def test_to_bits_le(self):
        p = Plaintext.from_string(PLAINTEXT_LITERAL)
        bits = p.to_bits_le()
        assert isinstance(bits, list)
        assert all(isinstance(b, bool) for b in bits)

    def test_to_bits_raw_le(self):
        p = Plaintext.from_string(PLAINTEXT_LITERAL)
        raw = p.to_bits_raw_le()
        assert isinstance(raw, list)
        assert all(isinstance(b, bool) for b in raw)

    def test_to_bits_raw_be(self):
        p = Plaintext.from_string(PLAINTEXT_LITERAL)
        raw = p.to_bits_raw_be()
        assert isinstance(raw, list)
        assert all(isinstance(b, bool) for b in raw)

    def test_to_bytes_raw_le(self):
        p = Plaintext.from_string(PLAINTEXT_LITERAL)
        raw = p.to_bytes_raw_le()
        assert isinstance(raw, bytes)  # pyo3 0.23+: Vec<u8> maps to bytes
        assert len(raw) > 0

    def test_to_bytes_raw_be(self):
        p = Plaintext.from_string(PLAINTEXT_LITERAL)
        raw = p.to_bytes_raw_be()
        assert isinstance(raw, bytes)  # pyo3 0.23+: Vec<u8> maps to bytes
        assert len(raw) > 0

    def test_to_fields_roundtrip(self):
        p = Plaintext.from_string(STRUCT)
        fields = p.to_fields()
        assert isinstance(fields, list)
        assert len(fields) > 0

    def test_to_fields_raw(self):
        p = Plaintext.from_string(PLAINTEXT_LITERAL)
        raw = p.to_fields_raw()
        assert isinstance(raw, list)
        assert len(raw) > 0

    def test_plaintext_type_literal(self):
        p = Plaintext.from_string("100u64")
        assert p.plaintext_type == "u64"

    def test_plaintext_type_struct(self):
        p = Plaintext.from_string(STRUCT)
        assert p.plaintext_type == "struct"

    def test_plaintext_type_array(self):
        p = Plaintext.from_string("[1u8, 2u8, 3u8]")
        assert p.plaintext_type == "array"

    def test_to_python_literal_int(self):
        p = Plaintext.from_string("42u64")
        v = p.to_python()
        assert v == 42
        assert isinstance(v, int)

    def test_to_python_literal_bool(self):
        p = Plaintext.from_string("true")
        v = p.to_python()
        assert v is True
        assert isinstance(v, bool)

    def test_to_python_struct(self):
        p = Plaintext.from_string(STRUCT)
        v = p.to_python()
        assert isinstance(v, dict)
        assert "microcredits" in v
        assert v["microcredits"] == 100_000_000

    def test_to_python_nested_struct(self):
        p = Plaintext.from_string(NESTED_STRUCT)
        v = p.to_python()
        assert isinstance(v, dict)
        assert "inventory" in v
        assert isinstance(v["inventory"], dict)

    def test_to_python_array(self):
        p = Plaintext.from_string("[1u8, 2u8, 3u8]")
        v = p.to_python()
        assert isinstance(v, list)
        assert len(v) == 3
        assert v[0] == 1

    def test_to_python_address_str(self):
        addr = "aleo1rhgdu77hgyqd3xjj8ucu3jj9r2krwz6mnzyd80gncr5fxcwlh5rsvzp9px"
        p = Plaintext.from_string(addr)
        v = p.to_python()
        assert isinstance(v, str)
        assert addr in v


# ---------------------------------------------------------------------------
# Value parity
# ---------------------------------------------------------------------------

class TestValueParity:
    def test_from_plaintext_literal(self):
        p = Plaintext.from_string("100u64")
        v = Value.from_plaintext(p)
        assert isinstance(v, Value)
        assert v.is_plaintext()
        assert v.value_type == "plaintext"

    def test_to_plaintext_roundtrip(self):
        p = Plaintext.from_string("100u64")
        v = Value.from_plaintext(p)
        p2 = v.to_plaintext()
        assert p == p2

    def test_to_plaintext_from_non_plaintext_fails(self):
        v = Value.parse(RECORD_VALUE)
        with pytest.raises(Exception):
            v.to_plaintext()

    def test_to_record_plaintext(self):
        v = Value.parse(RECORD_VALUE)
        assert v.is_record()
        r = v.to_record_plaintext()
        assert isinstance(r, RecordPlaintext)

    def test_is_future(self):
        v = Value.parse(FUTURE_VALUE)
        assert v.is_future()
        assert not v.is_plaintext()
        assert not v.is_record()

    def test_value_type_plaintext(self):
        v = Value.parse(PLAINTEXT_LITERAL)
        assert v.value_type == "plaintext"

    def test_value_type_record(self):
        v = Value.parse(RECORD_VALUE)
        assert v.value_type == "record"

    def test_value_type_future(self):
        v = Value.parse(FUTURE_VALUE)
        assert v.value_type == "future"

    def test_bytes_roundtrip(self):
        v = Value.parse(PLAINTEXT_LITERAL)
        b = v.bytes()
        v2 = Value.from_bytes(b)
        assert str(v) == str(v2)

    def test_to_bits_le(self):
        v = Value.parse(PLAINTEXT_LITERAL)
        bits = v.to_bits_le()
        assert isinstance(bits, list)
        assert all(isinstance(b, bool) for b in bits)

    def test_to_fields(self):
        v = Value.parse(PLAINTEXT_LITERAL)
        fields = v.to_fields()
        assert isinstance(fields, list)
        assert len(fields) > 0
        assert all(isinstance(f, Field) for f in fields)
