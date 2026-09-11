"""Trader-workflow invariants on live pool data — veil's
``traders.integration.test.ts`` for the helpers the Python SDK has (spot
price, LP range selection, market scan).  Read-only."""
from __future__ import annotations

from decimal import Decimal

import pytest

from aleo_shield_swap.tick_math import get_sqrt_price_at_tick_x128, u256_to_int

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def deep_pool(live_dex_module):
    """The first listed pool with live liquidity (a thin pool is unusable for
    range selection)."""
    for entry in live_dex_module.api.get_pools():
        slot = live_dex_module.get_slot(entry.key)
        if slot.liquidity > 0:
            return entry, slot
    pytest.skip("need at least one live pool with liquidity")


def test_spot_price_is_coherent(deep_pool):
    entry, slot = deep_pool
    d0 = entry.token0_info.decimals if entry.token0_info else 6
    d1 = entry.token1_info.decimals if entry.token1_info else 6
    price = slot.price(d0, d1)
    inverse = Decimal(1) / price
    assert price > 0
    # Reciprocity: token0-per-token1 is 1 / token1-per-token0.
    assert abs(price * inverse - 1) < Decimal("1e-6")
    # And the human price is the raw sqrt price squared, decimal-adjusted.
    raw = Decimal(u256_to_int(slot.sqrt_price)) / Decimal(1 << 128)
    assert abs(price - raw * raw * Decimal(10) ** (d0 - d1)) < Decimal("1e-18")


def test_in_range_lp_position_brackets_the_live_price(deep_pool):
    _, slot = deep_pool
    lower, upper = slot.tick_range(width=5)              # ±5 spacings
    assert lower % slot.tick_spacing == 0 and upper % slot.tick_spacing == 0
    assert lower < upper
    price = u256_to_int(slot.sqrt_price)
    assert get_sqrt_price_at_tick_x128(lower) <= price
    assert get_sqrt_price_at_tick_x128(upper) > price


def test_market_scan_every_listed_pool_reads_live_state(live_dex_module):
    entries = live_dex_module.api.get_pools()[:10]
    assert entries
    with_liquidity = 0
    for entry in entries:
        slot = live_dex_module.get_slot(entry.key)
        assert u256_to_int(slot.sqrt_price) > 0
        assert isinstance(slot.liquidity, int)
        with_liquidity += slot.liquidity > 0
    assert with_liquidity > 0
