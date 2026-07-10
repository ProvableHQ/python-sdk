"""Shared pure helpers for the Aleo facade (no side-effects, no imports of heavy modules).

This module is intentionally lightweight so it can be imported without pulling
in the compiled extension modules or the network client.
"""
from __future__ import annotations

_MICROCREDITS_PER_CREDIT: int = 1_000_000


def credits_to_microcredits(credits: float | int) -> int:
    """Convert a credits amount (float or int) to integer microcredits.

    Examples
    --------
    >>> credits_to_microcredits(1)
    1000000
    >>> credits_to_microcredits(1.5)
    1500000
    """
    return int(credits * _MICROCREDITS_PER_CREDIT)


def microcredits_to_credits(microcredits: int) -> float:
    """Convert an integer microcredits amount to a credits float.

    Examples
    --------
    >>> microcredits_to_credits(1_000_000)
    1.0
    >>> microcredits_to_credits(1_500_000)
    1.5
    """
    return microcredits / _MICROCREDITS_PER_CREDIT
