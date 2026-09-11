"""Q128.128 fixed-point tick math, mirroring the on-chain *_X128 table in
shield_swap.aleo (amm-v3 src/main.leo @ development).  The magic constants
ARE the contract's — do not "improve" them; a one-off value produces prices
the finalize asserts against.  Amounts are raw native units; the AMM no
longer scales by token decimals.
"""
from __future__ import annotations

from typing import Any

Q128 = 1 << 128
MIN_TICK = -400000
MAX_TICK = 400000
# Sentinel ticks anchoring the contract's initialized-tick linked list.
MIN_TICK_SENTINEL = -400001
MAX_TICK_SENTINEL = 400001
# sqrt price at MIN_TICK / MAX_TICK — the bounds the swap finalize accepts.
MIN_SQRT_RATIO_X128 = 702075911466779181339691826087
MAX_SQRT_RATIO_X128 = 484680305 * Q128 + 8756686347225649145659787327114459760

_TWO_256_MINUS_1 = (1 << 256) - 1

# round(2^128 / sqrt(1.0001^n)) for each power-of-two tick component.
_MAGIC_X128 = {
    1: 340265354078544963557816517032075149313,
    2: 340248342086729790484326174814286782778,
    3: 340231330945450418515964920540021147199,
    4: 340214320654664324051920982716015181260,
    8: 340146287995602323631171512101879684304,
    16: 340010263488231146823593991679159461444,
    32: 339738377640345403697157401104375502016,
    64: 339195258003219555707034227454543997025,
    128: 338111622100601834656805679988414885971,
    256: 335954724994790223023589805789778977700,
    512: 331682121138379247127172139078559817300,
    1024: 323299236684853023288211250268160618739,
    2048: 307163716377032989948697243942600083929,
    4096: 277268403626896220162999269216087595045,
    8192: 225923453940442621947126027127485391333,
    16384: 149997214084966997727330242082538205943,
    32768: 66119101136024775622716233608466517926,
    65536: 12847376061809297530290974190478138313,
    131072: 485053260817066172746253684029974020,
    262144: 691415978906521570653435304214168,
}
_BITS = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
         16384, 32768, 65536, 131072, 262144)


def get_sqrt_price_at_tick_x128(tick: int) -> int:
    """Q128.128 sqrt price at *tick* as the full 256-bit integer.

    Exact mirror of the contract's ``get_sqrt_price_at_tick_x128``: apply the
    *_X128 magic constants via ``(ratio * MAGIC) >> 128``, then invert
    ``(2^256 - 1) // lo`` for positive ticks.
    """
    if tick < MIN_TICK or tick > MAX_TICK:
        raise ValueError(f"Tick {tick} out of range [{MIN_TICK}, {MAX_TICK}]")
    abs_tick = abs(tick)
    low_bits = abs_tick & 0x3
    ratio = Q128 if low_bits == 0 else _MAGIC_X128[low_bits]
    for bit in _BITS:
        if abs_tick & bit:
            ratio = (ratio * _MAGIC_X128[bit]) >> 128
    if tick > 0:
        ratio = _TWO_256_MINUS_1 // (ratio & (Q128 - 1))
    return ratio


def round_tick_to_spacing(tick: int, spacing: int) -> int:
    """Largest spacing-aligned tick <= *tick* (mint bounds MUST be aligned)."""
    return (tick // spacing) * spacing


def u256_to_int(v: Any) -> int:
    """Full integer value of a wire U256: generated struct, decoded dict,
    or already-an-int."""
    if isinstance(v, int):
        return v
    if isinstance(v, dict):
        return (int(v["hi"]) << 128) | int(v["lo"])
    return (int(v.hi) << 128) | int(v.lo)


def int_to_u256_plaintext(value: int) -> str:
    """A 256-bit integer as the contract's ``{ hi, lo }`` struct literal."""
    if not 0 <= value < (1 << 256):
        raise ValueError(f"{value} out of range for U256")
    return f"{{ hi: {value >> 128}u128, lo: {value & (Q128 - 1)}u128 }}"
