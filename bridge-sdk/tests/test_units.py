from decimal import Decimal

import pytest

from aleo_bridge.errors import InvalidAmountError
from aleo_bridge.units import format_decimal_amount, parse_decimal_amount, resolve_amount


@pytest.mark.parametrize("amount,decimals,expected", [
    ("100", 6, 100_000_000), ("0.5", 6, 500_000), ("1.5", 6, 1_500_000), ("0.01", 8, 1_000_000),
    ("1.000000000000000001", 18, 10**18 + 1), ("42", 0, 42), ("0.123456", 6, 123_456),
    (Decimal("0.5"), 6, 500_000), (7, 6, 7_000_000), (" 2 ", 6, 2_000_000),
])
def test_parse_decimal_amount(amount, decimals, expected):
    assert parse_decimal_amount(amount, decimals) == expected


@pytest.mark.parametrize("amount", ["0.1234567", "1e6", "-1", "1.", "", ".5", "1,5", "abc"])
def test_parse_decimal_amount_rejects(amount):
    with pytest.raises(InvalidAmountError):
        parse_decimal_amount(amount, 6)


def test_parse_decimal_amount_rejects_unicode_digits():
    # Python's \d matches any Unicode decimal digit, not just ASCII 0-9; the regex must be [0-9] only.
    with pytest.raises(InvalidAmountError):
        parse_decimal_amount("٥٦", 6)   # ARABIC-INDIC DIGIT FIVE/SIX — category Nd, not ASCII


def test_parse_decimal_amount_rejects_bad_types_and_decimals():
    with pytest.raises(InvalidAmountError):
        parse_decimal_amount(1.5, 6)  # type: ignore[arg-type]
    with pytest.raises(InvalidAmountError):
        parse_decimal_amount(True, 6)  # type: ignore[arg-type]
    with pytest.raises(InvalidAmountError):
        parse_decimal_amount("1", -1)


def test_format_decimal_amount():
    assert format_decimal_amount(2_000_001, 6) == "2.000001"
    assert format_decimal_amount(1_500_000, 6) == "1.5"
    assert format_decimal_amount(100_000_000, 6) == "100"
    assert format_decimal_amount(1, 18) == "0.000000000000000001"
    assert format_decimal_amount(0, 6) == "0"
    assert format_decimal_amount(42, 0) == "42"
    with pytest.raises(InvalidAmountError):
        format_decimal_amount(-1, 6)


def test_resolve_amount_exactly_one():
    assert resolve_amount(amount="0.001", amount_atomic=None, decimals=8) == 100_000
    assert resolve_amount(amount=None, amount_atomic=100_000, decimals=8) == 100_000
    with pytest.raises(InvalidAmountError, match="exactly one"):
        resolve_amount(amount=None, amount_atomic=None, decimals=8)
    with pytest.raises(InvalidAmountError, match="exactly one"):
        resolve_amount(amount="1", amount_atomic=1, decimals=8)
    with pytest.raises(InvalidAmountError):
        resolve_amount(amount=None, amount_atomic=-5, decimals=8)
    with pytest.raises(InvalidAmountError):
        resolve_amount(amount=None, amount_atomic="5", decimals=8)  # type: ignore[arg-type]
