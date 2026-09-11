"""Bit-exact mirrors of shield_swap.aleo's position view helpers.

These reproduce the contract's own arithmetic so a caller can value a position
without a transaction.  Each function names the Leo helper it mirrors; the
vectors in ``tests/test_position_math.py`` are ported from the contract's
``tests/test_amm_helpers.leo`` rather than derived here, so a divergence in
either direction shows up as a test failure instead of a wrong balance.

Fee growth is 256-bit and **modular by design**: an ``outside`` counter may
exceed the global one, and the difference wraps at 2^256.  Every subtraction
here goes through :func:`u256_wrapping_sub` for that reason — a plain ``-``
would raise where the contract silently wraps.
"""
from __future__ import annotations

from typing import Any

U256_MOD = 1 << 256
U128_MAX = (1 << 128) - 1
Q128 = 1 << 128


def u256_wrapping_sub(a: int, b: int) -> int:
    """``a - b`` modulo 2^256 — mirrors ``u256::u256_sub``.

    Fee-growth accounting relies on this wrapping: a tick's ``outside`` counter
    can legitimately exceed the global counter, and the contract treats the
    negative result as its two's-complement 256-bit value.
    """
    return (a - b) % U256_MOD


def mul_div(a: int, b: int, denom: int, round_up: bool) -> int:
    """``a * b / denom``, floored or ceiled — mirrors ``view_mul_div``.

    Args:
        a: First factor.
        b: Second factor.
        denom: Divisor.
        round_up: Round a non-zero remainder up rather than down.

    Raises:
        ZeroDivisionError: If *denom* is zero, as the contract's divide would
            abort.
    """
    quotient, remainder = divmod(a * b, denom)
    return quotient + 1 if (round_up and remainder) else quotient


def amount0_delta(sqrt_a: int, sqrt_b: int, liquidity: int,
                  round_up: bool = False) -> int:
    """Token0 backing *liquidity* between two Q128.128 sqrt prices.

    Mirrors ``amt0_div_f``.  Argument order does not matter — the bounds are
    sorted internally.

    Raises:
        ValueError: If the result exceeds ``u128``, matching the contract's
            ``assert(r.hi == 0)``.
    """
    lower, upper = (sqrt_a, sqrt_b) if sqrt_a < sqrt_b else (sqrt_b, sqrt_a)
    diff = u256_wrapping_sub(upper, lower)
    scaled = mul_div(liquidity * Q128, diff, upper, round_up)
    result = mul_div(scaled, 1, lower, round_up)
    if result > U128_MAX:
        raise ValueError(f"amount0 {result} exceeds u128")
    return result


def amount1_delta(sqrt_a: int, sqrt_b: int, liquidity: int,
                  round_up: bool = False) -> int:
    """Token1 backing *liquidity* between two Q128.128 sqrt prices.

    Mirrors ``amt1_shift_f``: ``liquidity * (upper - lower) >> 128``, plus one
    when rounding up and the shift discarded a remainder.

    Raises:
        ValueError: If the intermediate product exceeds 256 bits, matching the
            contract's ``assert(hi256 == 0)``.
    """
    lower, upper = (sqrt_a, sqrt_b) if sqrt_a < sqrt_b else (sqrt_b, sqrt_a)
    product = liquidity * u256_wrapping_sub(upper, lower)
    if product >> 256:
        raise ValueError("amount1 intermediate exceeds 256 bits")
    return (product >> 128) + (1 if (round_up and product & U128_MAX) else 0)


def amounts_for_liquidity(sqrt_current: int, sqrt_a: int, sqrt_b: int,
                          liquidity: int, round_up: bool = False
                          ) -> tuple[int, int]:
    """The token amounts *liquidity* currently holds — ``view_amounts_for_liquidity``.

    Which side the position holds depends on where the pool price sits relative
    to the range: entirely token0 below the range, entirely token1 above it, and
    a split of both while in range.

    Args:
        sqrt_current: The pool's current Q128.128 sqrt price.
        sqrt_a: One range bound's sqrt price.
        sqrt_b: The other bound's sqrt price.
        liquidity: The position's liquidity.
        round_up: Round each amount up rather than down.

    Returns:
        ``(amount0, amount1)`` in raw base units.
    """
    lower, upper = (sqrt_a, sqrt_b) if sqrt_a < sqrt_b else (sqrt_b, sqrt_a)
    below = not lower < sqrt_current           # price at or below the range
    inside = not below and sqrt_current < upper
    above = not below and not sqrt_current < upper

    if below:
        return amount0_delta(lower, upper, liquidity, round_up), 0
    if inside:
        return (amount0_delta(sqrt_current, upper, liquidity, round_up),
                amount1_delta(lower, sqrt_current, liquidity, round_up))
    if above:
        return 0, amount1_delta(lower, upper, liquidity, round_up)
    return 0, 0


def liquidity_for_amount0(lower: int, upper: int, amount0: int) -> int:
    """Liquidity that *amount0* of token0 backs between two sorted sqrt prices.

    The inverse of :func:`amount0_delta` for a range the price sits below or
    inside (pass the current price as *lower* when in range).  Scales down by
    2^128 first, matching the forward direction's chained mul-divs rather than
    multiplying out to 2^256 and dividing back — so the round trip floors
    rather than reproducing the input exactly.

    Args:
        lower: The lower Q128.128 sqrt price (must be below *upper*).
        upper: The upper Q128.128 sqrt price.
        amount0: Raw base units of token0.

    Returns:
        The liquidity, floored.
    """
    intermediate = mul_div(lower, upper, Q128, False)
    return mul_div(amount0, intermediate, upper - lower, False)


def liquidity_for_amount1(lower: int, upper: int, amount1: int) -> int:
    """Liquidity that *amount1* of token1 backs between two sorted sqrt prices.

    The inverse of :func:`amount1_delta` for a range the price sits above or
    inside (pass the current price as *upper* when in range).

    Args:
        lower: The lower Q128.128 sqrt price (must be below *upper*).
        upper: The upper Q128.128 sqrt price.
        amount1: Raw base units of token1.

    Returns:
        The liquidity, floored.
    """
    return mul_div(amount1, Q128, upper - lower, False)


def liquidity_for_amount(sqrt_current: int, sqrt_a: int, sqrt_b: int, *,
                         side: int, amount: int) -> int:
    """Liquidity that *amount* of one token supports in a range — the inverse
    of :func:`amounts_for_liquidity` for a single side.  Floors.

    Args:
        sqrt_current: The pool's current Q128.128 sqrt price.
        sqrt_a: One range bound's sqrt price (order does not matter).
        sqrt_b: The other bound's sqrt price.
        side: ``0`` or ``1`` — which token *amount* is denominated in.
        amount: Raw base units of that token.

    Returns:
        The liquidity, or 0 when the price sits on the side of the range where
        that token backs nothing (token0 above the range, token1 below it).
    """
    lower, upper = (sqrt_a, sqrt_b) if sqrt_a < sqrt_b else (sqrt_b, sqrt_a)
    if side == 0:
        if sqrt_current <= lower:
            return liquidity_for_amount0(lower, upper, amount)
        if sqrt_current < upper:
            return liquidity_for_amount0(sqrt_current, upper, amount)
        return 0
    if sqrt_current >= upper:
        return liquidity_for_amount1(lower, upper, amount)
    if sqrt_current > lower:
        return liquidity_for_amount1(lower, sqrt_current, amount)
    return 0


def liquidity_for_amounts(sqrt_current: int, sqrt_a: int, sqrt_b: int,
                          amount0: int, amount1: int) -> int:
    """The largest liquidity both *amount0* and *amount1* can fund in a range.

    Below the range only token0 counts, above it only token1; in range both
    are consumed and the shorter side caps the position.  Floors, so a mint
    sized from :func:`amounts_for_liquidity` of the result never exceeds the
    amounts given.
    """
    lower, upper = (sqrt_a, sqrt_b) if sqrt_a < sqrt_b else (sqrt_b, sqrt_a)
    if sqrt_current <= lower:
        return liquidity_for_amount0(lower, upper, amount0)
    if sqrt_current < upper:
        return min(liquidity_for_amount0(sqrt_current, upper, amount0),
                   liquidity_for_amount1(lower, sqrt_current, amount1))
    return liquidity_for_amount1(lower, upper, amount1)


def fee_growth_inside(lower_outside: tuple[int, int], lower_tick: int,
                      upper_outside: tuple[int, int], upper_tick: int,
                      tick_current: int,
                      fee_growth_global: tuple[int, int]) -> tuple[int, int]:
    """Fee growth accrued inside a range — ``get_fee_growth_inside``.

    Args:
        lower_outside: The lower tick's ``(fee_growth_outside0, outside1)``.
        lower_tick: The lower tick index.
        upper_outside: The upper tick's ``(outside0, outside1)``.
        upper_tick: The upper tick index.
        tick_current: The pool's current tick.
        fee_growth_global: The pool's ``(global0, global1)``.

    Returns:
        ``(inside0, inside1)`` as 256-bit modular values — subtract two of these
        with :func:`u256_wrapping_sub`, never ``-``.
    """
    global0, global1 = fee_growth_global
    if tick_current >= lower_tick:
        below0, below1 = lower_outside
    else:
        below0 = u256_wrapping_sub(global0, lower_outside[0])
        below1 = u256_wrapping_sub(global1, lower_outside[1])

    if tick_current < upper_tick:
        above0, above1 = upper_outside
    else:
        above0 = u256_wrapping_sub(global0, upper_outside[0])
        above1 = u256_wrapping_sub(global1, upper_outside[1])

    return (u256_wrapping_sub(u256_wrapping_sub(global0, below0), above0),
            u256_wrapping_sub(u256_wrapping_sub(global1, below1), above1))


def fee_owed(growth_now: int, growth_last: int, liquidity: int) -> int:
    """Fees a position has accrued — ``fee_owed``.

    ``floor((growth_now - growth_last) * liquidity / 2^128)`` over the modular
    delta, so a wrapped accumulator settles correctly.

    Raises:
        ValueError: If the result exceeds ``u128``, matching the contract's
            overflow assert.  At ``liquidity == 1`` the contract notes this
            cannot detect a spurious modular underflow, so a nonsensical
            ``growth_last`` yields a large-but-valid figure rather than an error.
    """
    delta = u256_wrapping_sub(growth_now, growth_last)
    whole = (delta >> 128) * liquidity
    frac_hi = ((delta & U128_MAX) * liquidity) >> 128
    if whole >> 128 or frac_hi > U128_MAX - (whole & U128_MAX):
        raise ValueError("fee_owed exceeds u128")
    return (whole & U128_MAX) + frac_hi


def u256_of(value: Any) -> int:
    """A generated ``U256`` struct (or plain int) as a Python integer."""
    if isinstance(value, int):
        return value
    return (int(value.hi) << 128) | int(value.lo)
