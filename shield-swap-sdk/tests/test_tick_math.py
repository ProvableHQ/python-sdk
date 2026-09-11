import pytest

from aleo_shield_swap.tick_math import (
    MAX_SQRT_RATIO_X128,
    MAX_TICK,
    MIN_SQRT_RATIO_X128,
    MIN_TICK,
    Q128,
    get_sqrt_price_at_tick_x128,
    int_to_u256_plaintext,
    round_tick_to_spacing,
    u256_to_int,
)

# Vectors from the reference implementation (amm-v3 ts-tests/src/utils/math.ts
# @ development c32a5e3), which mirrors the contract's *_X128 constants.
VECTORS = {
    0: Q128,
    1: 340299380613952818054172298683778356828,
    -1: 340265354078544963557816517032075149313,
    100: 341987953891916247014855103371247308527,
    -100: 338585286176225270960996223397573581044,
    MIN_TICK: 702075911466779181339691826087,
    MAX_TICK: 164928161394119051704885410204944470744913033840,
}


def test_sqrt_price_x128_vectors():
    for tick, expected in VECTORS.items():
        assert get_sqrt_price_at_tick_x128(tick) == expected, tick


def test_bounds_are_the_tick_extremes():
    assert (MIN_TICK, MAX_TICK) == (-400000, 400000)
    assert get_sqrt_price_at_tick_x128(MIN_TICK) == MIN_SQRT_RATIO_X128
    assert get_sqrt_price_at_tick_x128(MAX_TICK) == MAX_SQRT_RATIO_X128


def test_out_of_range_tick_rejects():
    with pytest.raises(ValueError):
        get_sqrt_price_at_tick_x128(MAX_TICK + 1)
    with pytest.raises(ValueError):
        get_sqrt_price_at_tick_x128(MIN_TICK - 1)


def test_monotonic_around_zero():
    assert (get_sqrt_price_at_tick_x128(600)
            > get_sqrt_price_at_tick_x128(0)
            > get_sqrt_price_at_tick_x128(-600))


def test_round_tick_to_spacing():
    assert round_tick_to_spacing(4055, 60) == 4020
    assert round_tick_to_spacing(-62215, 200) == -62400
    assert round_tick_to_spacing(199, 200) == 0
    assert round_tick_to_spacing(-1, 200) == -200


def test_u256_helpers():
    class _U:  # duck-typed like the generated struct
        hi, lo = 1, 5
    assert u256_to_int(_U()) == (1 << 128) + 5
    assert u256_to_int({"hi": 0, "lo": 7}) == 7
    assert u256_to_int(9) == 9
    assert int_to_u256_plaintext((1 << 128) + 5) == "{ hi: 1u128, lo: 5u128 }"
    with pytest.raises(ValueError):
        int_to_u256_plaintext(1 << 256)
