"""Exact decimal ↔ atomic conversion (port of veil ``utils/units.ts``). No floats anywhere."""
from __future__ import annotations

import re
from decimal import Decimal

from .errors import InvalidAmountError

_DECIMAL_RE = re.compile(r"^([0-9]+)(?:\.([0-9]+))?$")


def _check_decimals(decimals: int) -> None:
    if isinstance(decimals, bool) or not isinstance(decimals, int) or decimals < 0:
        raise InvalidAmountError(f"Asset decimals must be a non-negative integer, got {decimals!r}")


def parse_decimal_amount(amount: "str | Decimal | int", decimals: int) -> int:
    """``"0.5"`` with 6 decimals → ``500000``. Strict: digits with an optional fraction only.

    Rejects exponents, signs, a trailing dot, the empty string, floats, and any fraction longer
    than *decimals* (that precision cannot exist on chain).
    """
    _check_decimals(decimals)
    if isinstance(amount, bool) or not isinstance(amount, (str, int, Decimal)):
        raise InvalidAmountError(f"Amount must be a decimal string, int or Decimal, got {type(amount).__name__}")
    text = format(amount, "f") if isinstance(amount, Decimal) else str(amount).strip()
    match = _DECIMAL_RE.match(text)
    if not match:
        raise InvalidAmountError(f'Invalid decimal amount "{amount}" — use digits with an optional fraction, e.g. "0.5"')
    whole, frac = match.group(1), match.group(2) or ""
    if len(frac) > decimals:
        raise InvalidAmountError(
            f'Amount "{amount}" has {len(frac)} fractional digits but the asset supports {decimals}')
    return int(whole + frac.ljust(decimals, "0"))


def format_decimal_amount(atomic: int, decimals: int) -> str:
    """``2000001`` with 6 decimals → ``"2.000001"``; trailing fractional zeros are stripped."""
    _check_decimals(decimals)
    if isinstance(atomic, bool) or not isinstance(atomic, int) or atomic < 0:
        raise InvalidAmountError(f"Atomic amount must be a non-negative int, got {atomic!r}")
    if decimals == 0:
        return str(atomic)
    digits = str(atomic).rjust(decimals + 1, "0")
    whole, fraction = digits[:-decimals], digits[-decimals:].rstrip("0")
    return f"{whole}.{fraction}" if fraction else whole


def resolve_amount(*, amount: "str | Decimal | int | None", amount_atomic: "int | None", decimals: int) -> int:
    """Exactly one of *amount* (human) / *amount_atomic* (int) → atomic int."""
    if (amount is None) == (amount_atomic is None):
        raise InvalidAmountError("Pass exactly one of amount= (decimal string) or amount_atomic= (int)")
    if amount_atomic is not None:
        if isinstance(amount_atomic, bool) or not isinstance(amount_atomic, int) or amount_atomic < 0:
            raise InvalidAmountError(f"amount_atomic must be a non-negative int, got {amount_atomic!r}")
        return amount_atomic
    return parse_decimal_amount(amount, decimals)  # type: ignore[arg-type]


__all__ = ["parse_decimal_amount", "format_decimal_amount", "resolve_amount"]
