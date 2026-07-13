"""Tests for aleo.codegen.runtime — plaintext parsing and literal formatting."""
import pytest

from aleo.codegen.runtime import (
    fmt_address,
    fmt_bool,
    fmt_fieldlike,
    fmt_int,
    parse_plaintext,
)


def test_parse_scalar_literals():
    assert parse_plaintext("4055i32") == 4055
    assert parse_plaintext("-4055i32") == -4055
    assert parse_plaintext("183051202759u128") == 183051202759
    assert parse_plaintext("true") is True
    assert parse_plaintext("false") is False
    assert parse_plaintext("4719field") == "4719field"
    assert parse_plaintext("2group") == "2group"
    assert parse_plaintext("7scalar") == "7scalar"
    assert parse_plaintext("aleo1qyqsqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq5g5x67") \
        == "aleo1qyqsqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq5g5x67"


def test_parse_struct():
    text = "{ tick: 4055i32, sqrt_price: 22526123159817891330747538u128, pool: 4719field }"
    assert parse_plaintext(text) == {
        "tick": 4055,
        "sqrt_price": 22526123159817891330747538,
        "pool": "4719field",
    }


def test_parse_nested_struct_and_array():
    text = "{ inner: { a: 1u8, flag: true }, xs: [1u8, 2u8] }"
    assert parse_plaintext(text) == {"inner": {"a": 1, "flag": True}, "xs": [1, 2]}


def test_parse_record_strips_modes_keeps_nonce():
    text = ("{ owner: aleo1abc.private, amount: 5000000u128.private, "
            "token_id: 99field.private, _nonce: 123group.public }")
    assert parse_plaintext(text) == {
        "owner": "aleo1abc",
        "amount": 5000000,
        "token_id": "99field",
        "_nonce": "123group",
    }


def test_fmt_int_ranges():
    assert fmt_int(5, "u128") == "5u128"
    assert fmt_int(-1, "i32") == "-1i32"
    with pytest.raises(ValueError):
        fmt_int(-1, "u64")            # negative unsigned
    with pytest.raises(ValueError):
        fmt_int(2**32, "u32")         # overflow
    with pytest.raises(ValueError):
        fmt_int(2**31, "i32")         # signed overflow
    with pytest.raises(ValueError):
        fmt_int(True, "u8")           # bool is not an int here


def test_parse_rejects_malformed_and_absent():
    with pytest.raises(ValueError):
        parse_plaintext("{ a: 1u8 2u8 }")      # missing comma between members
    with pytest.raises(ValueError):
        parse_plaintext("[1u8 2u8]")            # missing comma between elements
    with pytest.raises(ValueError):
        parse_plaintext("null")                 # absent mapping entry
    with pytest.raises(ValueError):
        parse_plaintext("")
    with pytest.raises(TypeError):
        parse_plaintext(None)                   # absent value from the node


def test_fmt_fieldlike_and_address():
    assert fmt_fieldlike(123, "field") == "123field"
    assert fmt_fieldlike("123field", "field") == "123field"
    assert fmt_fieldlike("-1field", "field") == "-1field"   # mod-p negative
    assert fmt_fieldlike(-1, "field") == "-1field"
    with pytest.raises(ValueError):
        fmt_fieldlike("123group", "field")   # wrong suffix
    assert fmt_bool(True) == "true"
    assert fmt_bool(False) == "false"
    assert fmt_address("aleo1abc") == "aleo1abc"
    with pytest.raises(ValueError):
        fmt_address("0xdeadbeef")
