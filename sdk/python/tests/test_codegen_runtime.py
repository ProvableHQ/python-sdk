"""Tests for aleo.codegen.runtime — plaintext parsing and literal formatting."""
from aleo.codegen.runtime import parse_plaintext


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
