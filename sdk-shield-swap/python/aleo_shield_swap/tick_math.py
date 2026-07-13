"""Q64 fixed-point tick math, ported from the TS SDK (utils/tick-math.ts),
which mirrors the on-chain tick_math table in shield_swap.  The magic
constants ARE the contract's — do not "improve" them; a one-off value
produces prices the finalize asserts against.
"""
from __future__ import annotations

Q64 = 9223372036854775808          # 2**63 — the contract's sqrt-price scale
MIN_TICK = -400000
MAX_TICK = 400000
# sqrt price at MIN_TICK / MAX_TICK — the bounds the swap finalize accepts.
MIN_SQRT_PRICE = 19029805711
MAX_SQRT_PRICE = 4470386772317930780047134862

# sqrt(1.0001^-bit) in Q64 for each power-of-two tick component.
_MAGIC = {
    1: 9222910902837697536, 2: 9222449791875588096, 3: 9221988703967300608,
    4: 9221527639111677952, 8: 9219683610192801792, 16: 9215996658532725760,
    32: 9208627177859081216, 64: 9193905890596798464, 128: 9164533880601766912,
    256: 9106071067403056128, 512: 8990261907820801024, 1024: 8763043369415622656,
    2048: 8325689215117605888, 4096: 7515375139346884608, 8192: 6123667489441693696,
    16384: 4065682634442729984, 32768: 1792161827361994496, 65536: 348228825923923264,
    131072: 13147394978735516, 262144: 18740867660568, 524288: 38079361,
}
_BITS = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
         16384, 32768, 65536, 131072, 262144, 524288)


def get_sqrt_price_at_tick(tick: int) -> int:
    """Q64 sqrt price at *tick*, matching the contract's table."""
    if tick < MIN_TICK or tick > MAX_TICK:
        raise ValueError(f"Tick {tick} out of range [{MIN_TICK}, {MAX_TICK}]")
    if tick == 0:
        return Q64
    abs_tick = abs(tick)
    low_bits = abs_tick & 0x3
    ratio = Q64 if low_bits == 0 else _MAGIC[low_bits]
    for bit in _BITS:
        if abs_tick & bit:
            ratio = (ratio * _MAGIC[bit]) >> 63
    # The table encodes negative ticks; invert for positive ones.
    if tick < 0:
        return ratio
    return (Q64 * Q64) // ratio


def round_tick_to_spacing(tick: int, spacing: int) -> int:
    """Largest spacing-aligned tick <= *tick* (mint bounds MUST be aligned)."""
    return (tick // spacing) * spacing


def dust_scale(decimals: int) -> int:
    """Divisor the contract normalizes a token's raw amounts by.

    shield_swap accounts in 9-decimal-normalized units: raw inputs must
    satisfy ``raw % dust_scale(decimals) == 0`` or the contract rejects them.
    """
    return 10 ** (decimals - 9) if decimals > 9 else 1
