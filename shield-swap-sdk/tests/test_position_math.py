"""Position view math — vectors ported from amm-v3 tests/test_amm_helpers.leo.

Expected values are the contract's own assertions, transcribed rather than
recomputed here, so a divergence in either implementation fails this file.
"""
from __future__ import annotations

import pytest

from aleo_shield_swap.position_math import (
    Q128,
    U128_MAX,
    amount0_delta,
    amount1_delta,
    amounts_for_liquidity,
    fee_growth_inside,
    fee_owed,
    u256_of,
    u256_wrapping_sub,
)


def _u256(hi: int, lo: int) -> int:
    """The Leo vectors are written as ``U256 { hi, lo }``."""
    return (hi << 128) | lo


# ── fee_owed — t_fee_owed ────────────────────────────────────────────────────

def test_fee_owed_vectors():
    two128 = _u256(1, 0)
    # delta = 2^128, liquidity 1000 -> 1000 owed
    assert fee_owed(two128, 0, 1000) == 1000
    # delta = 2*2^128 - 2^128 = 2^128, liquidity 5000 -> 5000
    assert fee_owed(_u256(2, 0), two128, 5000) == 5000
    # wrapped accumulator (now < last): modular delta = 2^128, liq 1000 -> 1000
    assert fee_owed(_u256(0, 5), _u256(U128_MAX, 5), 1000) == 1000
    # zero liquidity owes nothing
    assert fee_owed(0, 0, 0) == 0
    # delta = 3*2^128, liquidity 7 -> 21
    assert fee_owed(_u256(3, 0), 0, 7) == 21


def test_fee_owed_overflow_raises():
    with pytest.raises(ValueError, match="exceeds u128"):
        fee_owed(_u256(U128_MAX, 0), 0, U128_MAX)


# ── fee_growth_inside — t_fee_growth_inside ─────────────────────────────────

LOWER_OUTSIDE, LOWER_TICK = (10, 20), -100
UPPER_OUTSIDE, UPPER_TICK = (5, 8), 100
GLOBAL = (1000, 2000)


def _inside(tick_current: int) -> tuple[int, int]:
    return fee_growth_inside(LOWER_OUTSIDE, LOWER_TICK,
                             UPPER_OUTSIDE, UPPER_TICK,
                             tick_current, GLOBAL)


def test_fee_growth_inside_in_range():
    assert _inside(0) == (985, 1972)


def test_fee_growth_inside_below_range():
    assert _inside(-200) == (5, 12)


def test_fee_growth_inside_above_range_wraps():
    # above the range the accounting wraps at 2^256 — modular by design
    assert _inside(150) == (
        _u256(U128_MAX, 340282366920938463463374607431768211451),
        _u256(U128_MAX, 340282366920938463463374607431768211444),
    )


def test_fee_growth_inside_exactly_at_lower_is_in_range():
    # tick_current >= lower takes the in-range arm
    assert _inside(-100) == (985, 1972)


def test_fee_growth_inside_wrapped_outsides():
    # t_fee_growth_inside_wrapped: outside counters exceed the global
    inside0, _ = fee_growth_inside((3000, 0), -100, (2500, 0), 100, 0, (2000, 0))
    # 2000 - 3000 - 2500 modulo 2^256
    assert inside0 == (2000 - 3000 - 2500) % (1 << 256)


# ── u256_wrapping_sub ───────────────────────────────────────────────────────

def test_u256_wrapping_sub_wraps_not_raises():
    assert u256_wrapping_sub(5, 7) == (1 << 256) - 2
    assert u256_wrapping_sub(7, 5) == 2
    assert u256_wrapping_sub(0, 0) == 0


# ── amounts_for_liquidity ───────────────────────────────────────────────────

def test_amounts_below_range_is_all_token0():
    lo, hi = 2 * Q128, 4 * Q128
    a0, a1 = amounts_for_liquidity(Q128, lo, hi, 10**6)
    assert a1 == 0 and a0 == amount0_delta(lo, hi, 10**6)


def test_amounts_above_range_is_all_token1():
    lo, hi = 2 * Q128, 4 * Q128
    a0, a1 = amounts_for_liquidity(8 * Q128, lo, hi, 10**6)
    assert a0 == 0 and a1 == amount1_delta(lo, hi, 10**6)


def test_amounts_in_range_holds_both():
    lo, hi, cur = 2 * Q128, 4 * Q128, 3 * Q128
    a0, a1 = amounts_for_liquidity(cur, lo, hi, 10**6)
    assert a0 > 0 and a1 > 0
    assert a0 == amount0_delta(cur, hi, 10**6)
    assert a1 == amount1_delta(lo, cur, 10**6)


def test_amounts_bound_order_does_not_matter():
    lo, hi, cur = 2 * Q128, 4 * Q128, 3 * Q128
    assert (amounts_for_liquidity(cur, lo, hi, 10**6)
            == amounts_for_liquidity(cur, hi, lo, 10**6))


def test_amounts_at_lower_bound_is_below_arm():
    # below is `not lower < sqrt_current`, so price == lower counts as below
    lo, hi = 2 * Q128, 4 * Q128
    a0, a1 = amounts_for_liquidity(lo, lo, hi, 10**6)
    assert a1 == 0 and a0 == amount0_delta(lo, hi, 10**6)


def test_amounts_at_upper_bound_is_above_arm():
    lo, hi = 2 * Q128, 4 * Q128
    a0, a1 = amounts_for_liquidity(hi, lo, hi, 10**6)
    assert a0 == 0 and a1 == amount1_delta(lo, hi, 10**6)


def test_zero_liquidity_holds_nothing():
    assert amounts_for_liquidity(3 * Q128, 2 * Q128, 4 * Q128, 0) == (0, 0)


def test_round_up_never_below_round_down():
    lo, hi, cur = 2 * Q128 + 7, 4 * Q128 + 13, 3 * Q128 + 5
    down = amounts_for_liquidity(cur, lo, hi, 12345, round_up=False)
    up = amounts_for_liquidity(cur, lo, hi, 12345, round_up=True)
    assert up[0] >= down[0] and up[1] >= down[1]


# ── u256_of ─────────────────────────────────────────────────────────────────

def test_u256_of_accepts_struct_or_int():
    class _S:
        hi, lo = 3, 5
    assert u256_of(_S()) == _u256(3, 5)
    assert u256_of(42) == 42


def test_zero_sqrt_price_raises_rather_than_silently_returning():
    """A zero bound divides by zero in the contract too — surface it."""
    with pytest.raises(ZeroDivisionError):
        amount0_delta(0, 4 * Q128, 10**6)


# ── liquidity_for_amount(s) — inverse of amounts_for_liquidity ──────────────

def _vectors():
    import json
    from pathlib import Path
    return json.loads((Path(__file__).parent / "fixtures"
                       / "q128_oracle_vectors.json").read_text())


def test_liquidity_for_amount_side0_below_range_prices_over_full_width():
    from aleo_shield_swap.position_math import liquidity_for_amount
    from aleo_shield_swap.tick_math import get_sqrt_price_at_tick_x128 as sp
    lower, upper = sp(-600), sp(600)
    amount0, _ = amounts_for_liquidity(lower - 1, lower, upper, 10**12)
    liq = liquidity_for_amount(lower - 1, lower, upper, side=0, amount=amount0)
    assert 0 < liq <= 10**12 and 10**12 - liq <= 2       # floors, near-exact


def test_liquidity_for_amount_side_not_backing_the_range_is_zero():
    from aleo_shield_swap.position_math import liquidity_for_amount
    from aleo_shield_swap.tick_math import get_sqrt_price_at_tick_x128 as sp
    lower, upper = sp(-600), sp(600)
    assert liquidity_for_amount(upper + 1, lower, upper, side=0, amount=10**9) == 0
    assert liquidity_for_amount(lower - 1, lower, upper, side=1, amount=10**9) == 0


def test_liquidity_for_amounts_in_range_is_capped_by_the_shorter_side():
    from aleo_shield_swap.position_math import liquidity_for_amount, liquidity_for_amounts
    from aleo_shield_swap.tick_math import get_sqrt_price_at_tick_x128 as sp
    lower, upper, cur = sp(-600), sp(600), sp(0)
    a0, a1 = amounts_for_liquidity(cur, lower, upper, 10**15)
    from0 = liquidity_for_amount(cur, lower, upper, side=0, amount=a0)
    from1 = liquidity_for_amount(cur, lower, upper, side=1, amount=a1 * 3)
    assert liquidity_for_amounts(cur, lower, upper, a0, a1 * 3) == min(from0, from1)
    assert liquidity_for_amounts(cur, lower, upper, a0, a1 * 3) == from0


def test_liquidity_for_amounts_bound_order_does_not_matter():
    from aleo_shield_swap.position_math import liquidity_for_amounts
    from aleo_shield_swap.tick_math import get_sqrt_price_at_tick_x128 as sp
    lower, upper, cur = sp(-600), sp(600), sp(0)
    assert (liquidity_for_amounts(cur, lower, upper, 10**9, 10**9)
            == liquidity_for_amounts(cur, upper, lower, 10**9, 10**9))


def test_liquidity_roundtrip_is_tight_but_not_exact():
    """amounts_for_liquidity(L) -> liquidity_for_amounts lands within a
    hair of L.  It is NOT exact: the token0 inverse scales ``lower*upper`` down
    by 2^128 first (as the TS oracle does), so it can land a few units either
    side — which is exactly why the rebalance planner re-derives the deposit
    from the solved liquidity and clamps, rather than trusting the inverse."""
    from aleo_shield_swap.position_math import liquidity_for_amounts
    for v in _vectors()["amounts"]:
        lower, upper = int(v["sqrtA"]), int(v["sqrtB"])
        liquidity = int(v["liquidity"])
        for cur in (lower - 1, (lower + upper) // 2, upper):
            a0, a1 = amounts_for_liquidity(cur, lower, upper, liquidity, round_up=True)
            back = liquidity_for_amounts(cur, lower, upper, a0, a1)
            assert abs(back - liquidity) <= liquidity // 10**9 + 2, (v, cur, back)


# ── veil q128 oracle vectors (differentially tested against amm-v3's Python oracles)

def test_oracle_amount_vectors():
    for v in _vectors()["amounts"]:
        lower, upper = int(v["sqrtA"]), int(v["sqrtB"])
        a0 = amount0_delta(lower, upper, int(v["liquidity"]), v["roundUp"])
        a1 = amount1_delta(lower, upper, int(v["liquidity"]), v["roundUp"])
        assert (a0, a1) == (int(v["amount0"]), int(v["amount1"])), v


def test_oracle_mul_div_vectors():
    from aleo_shield_swap.position_math import mul_div
    for v in _vectors()["mulDiv"]:
        assert mul_div(int(v["a"]), int(v["b"]), int(v["d"]), v["roundUp"]) == int(v["result"]), v


def test_oracle_sqrt_price_at_tick_vectors():
    from aleo_shield_swap.tick_math import get_sqrt_price_at_tick_x128
    for v in _vectors()["sqrtPriceAtTickX128"]:
        assert get_sqrt_price_at_tick_x128(v["tick"]) == int(v["sqrtPriceX128"]), v
