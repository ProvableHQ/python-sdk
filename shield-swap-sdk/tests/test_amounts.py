"""Exact conversion inside swap preparation; integer inputs retain base units."""
from decimal import Decimal, localcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from aleo_shield_swap import ShieldSwap
from aleo_shield_swap._core import _amount_to_base_units
from .test_swap import _swap_call_on

TOKENS = [SimpleNamespace(id="1field", address="a.aleo", decimals=6),
          SimpleNamespace(id="2field", address="b.aleo", decimals=8)]


@pytest.mark.parametrize("amount", ["1.5", Decimal("1.5")])
def test_swap_converts_input_and_quote_internally(stub_aleo, amount):
    dex = ShieldSwap(stub_aleo)
    dex.api.get_tokens = Mock(return_value=TOKENS)
    handle = _swap_call_on(dex, amount_in=amount, expected_out="0.0005").transact()
    assert stub_aleo.last_call[1][5] == "1500000u128"
    assert stub_aleo.last_call[1][6] == "49750u128"
    assert handle.amount_in == 1500000


def test_integer_amounts_do_not_need_metadata():
    assert _amount_to_base_units(1500000, "unknown", []) == 1500000


def test_decimal_conversion_ignores_context_precision():
    with localcontext() as context:
        context.prec = 2
        assert _amount_to_base_units("123456789.123456", "1field", TOKENS) == 123456789123456


@pytest.mark.parametrize("amount", ["0.0000001", "NaN", "Infinity", "-1", "garbage", str(2**128)])
def test_invalid_decimal_amounts_are_rejected(amount):
    with pytest.raises(ValueError):
        _amount_to_base_units(amount, "1field", TOKENS)


@pytest.mark.parametrize("amount", [1.5, True])
def test_float_and_boolean_amounts_are_rejected(amount):
    with pytest.raises(TypeError):
        _amount_to_base_units(amount, "1field", TOKENS)


def test_unknown_token_is_not_assigned_default_decimals():
    with pytest.raises(ValueError, match="metadata"):
        _amount_to_base_units("1.5", "unknown", TOKENS)


def test_trailing_zeroes_are_exact():
    assert _amount_to_base_units("1.5000000", "1field", TOKENS) == 1500000
