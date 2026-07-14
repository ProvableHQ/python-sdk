import pytest

from aleo_shield_swap.tick_math import (
    MAX_SQRT_PRICE,
    MAX_TICK,
    MIN_SQRT_PRICE,
    MIN_TICK,
    Q64,
    dust_scale,
    get_sqrt_price_at_tick,
    round_tick_to_spacing,
)


def test_constants():
    assert Q64 == 9223372036854775808 == 2**63
    assert (MIN_TICK, MAX_TICK) == (-400000, 400000)
    assert MIN_SQRT_PRICE == 19029805711
    assert MAX_SQRT_PRICE == 4470386772317930780047134862


def test_sqrt_price_boundaries():
    assert get_sqrt_price_at_tick(0) == Q64
    assert get_sqrt_price_at_tick(MIN_TICK) == MIN_SQRT_PRICE
    assert get_sqrt_price_at_tick(MAX_TICK) == MAX_SQRT_PRICE
    with pytest.raises(ValueError):
        get_sqrt_price_at_tick(MAX_TICK + 1)
    with pytest.raises(ValueError):
        get_sqrt_price_at_tick(MIN_TICK - 1)


def test_sqrt_price_symmetry():
    # positive ticks invert the negative-tick table entry
    neg = get_sqrt_price_at_tick(-600)
    pos = get_sqrt_price_at_tick(600)
    assert pos == (Q64 * Q64) // neg
    # monotonic: higher tick, higher price
    assert get_sqrt_price_at_tick(600) > get_sqrt_price_at_tick(0) > neg


def test_rounding_and_dust():
    assert round_tick_to_spacing(-62215, 200) == -62400
    assert round_tick_to_spacing(199, 200) == 0
    assert round_tick_to_spacing(-1, 200) == -200
    assert dust_scale(18) == 10**9
    assert dust_scale(9) == 1
    assert dust_scale(6) == 1
