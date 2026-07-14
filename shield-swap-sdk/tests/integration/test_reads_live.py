"""Read tier: EVERY read action against the live DEX API + testnet chain.

No credentials, no spending — run with ``-m live``.  Assertions target
invariants and shapes, not exact live figures (testnet state varies).
"""
from __future__ import annotations

import pytest

from aleo_shield_swap.errors import (
    DexApiError,
    PoolNotFoundError,
    SwapOutputNotFinalizedError,
)
from .conftest import skip_if_access_gated
from aleo_shield_swap.tick_math import (
    MAX_SQRT_PRICE,
    MAX_TICK,
    MIN_SQRT_PRICE,
    MIN_TICK,
)

pytestmark = pytest.mark.live

BURN_ADDRESS = "aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3ljyzc"


@pytest.fixture(scope="module")
def pools(live_dex_module):
    entries = live_dex_module.api.get_pools()
    assert entries, "no pools on the DEX API"
    return entries


@pytest.fixture(scope="module")
def pool(pools):
    return pools[0]


# ── dex.api (off-chain service) ──────────────────────────────────────────────

def test_api_get_pools_shapes(pools):
    for entry in pools:
        assert entry.key.endswith("field")
        assert entry.token0.endswith("field") and entry.token1.endswith("field")
        assert isinstance(entry.enabled, bool)
        if entry.token0_info is not None:
            assert entry.token0_info.decimals >= 0
            assert entry.token0_info.wrapper_program.endswith(".aleo")


def test_api_get_tokens(live_dex_module):
    tokens = live_dex_module.api.get_tokens()
    assert tokens, "no tokens in the registry"
    for tok in tokens:
        assert tok.address.endswith("field")
        assert tok.decimals >= 0
        assert tok.symbol


def test_api_get_route_quotes_both_directions(live_dex_module, pool):
    scale = 10 ** (pool.token0_info.decimals if pool.token0_info else 6)
    fwd = skip_if_access_gated(lambda: live_dex_module.api.get_route(
        token_in=pool.token0, token_out=pool.token1, amount_in=scale))
    assert fwd.token_in == pool.token0 and fwd.token_out == pool.token1
    assert fwd.hops, "route has no hops"
    rev = live_dex_module.api.get_route(
        token_in=pool.token1, token_out=pool.token0)
    assert rev.token_in == pool.token1


def test_api_get_ohlcv(live_dex_module, pool):
    candles = skip_if_access_gated(lambda: live_dex_module.api.get_ohlcv(
        pool.key, granularity="1d",
        from_ts="2026-01-01T00:00:00", to_ts="2026-12-31T00:00:00"))
    for candle in candles:                     # may be empty on a quiet pool
        assert float(candle.h) >= float(candle.l)


def test_api_get_public_balances_shape(live_dex_module):
    # Any address is valid to query; the burn address just returns few/none.
    balances = skip_if_access_gated(
        lambda: live_dex_module.api.get_public_balances(BURN_ADDRESS))
    for bal in balances:
        assert int(bal.balance) >= 0
        assert bal.token_id.endswith("field")


def test_api_get_swap_unknown_id_raises(live_dex_module):
    with pytest.raises(DexApiError):
        live_dex_module.api.get_swap("0field")


# ── Chain reads (node, via the facade) ───────────────────────────────────────

def test_get_pool_matches_api(live_dex_module, pool):
    chain_pool = live_dex_module.get_pool(pool.key)
    assert chain_pool.token0 == pool.token0
    assert chain_pool.token1 == pool.token1
    assert chain_pool.scale0 >= 1 and chain_pool.scale1 >= 1


def test_get_slot_invariants(live_dex_module, pool):
    slot = live_dex_module.get_slot(pool.key)
    assert MIN_SQRT_PRICE <= slot.sqrt_price <= MAX_SQRT_PRICE
    assert MIN_TICK <= slot.tick <= MAX_TICK
    assert slot.tick_spacing > 0
    assert slot.next_init_below <= slot.tick <= slot.next_init_above
    d0 = pool.token0_info.decimals if pool.token0_info else 9
    d1 = pool.token1_info.decimals if pool.token1_info else 9
    assert slot.price(d0, d1) > 0


def test_is_pool_initialized(live_dex_module, pool):
    assert live_dex_module.is_pool_initialized(pool.key) is True
    assert live_dex_module.is_pool_initialized("1field") is False


def test_missing_pool_raises(live_dex_module):
    with pytest.raises(PoolNotFoundError):
        live_dex_module.get_pool("1field")


def test_get_swap_output_absent_raises(live_dex_module):
    with pytest.raises(SwapOutputNotFinalizedError):
        live_dex_module.get_swap_output("1field")


# ── Local derivations vs live chain (the strongest cheap invariants) ─────────

def test_local_pool_key_matches_indexer(live_dex_module, pools):
    """Local BHP256 derivation must reproduce the indexer's pool keys."""
    matched = 0
    for entry in pools:
        chain_pool = live_dex_module.get_pool(entry.key)
        derived = live_dex_module.derive_pool_key(
            chain_pool.token0, chain_pool.token1, chain_pool.fee)
        assert derived == entry.key
        matched += 1
    assert matched > 0


def test_local_tick_key_locates_initialized_tick(live_dex_module, pool):
    """A tick the slot names as initialized must be readable via the locally
    derived tick key (i32 struct-hash parity with the contract)."""
    slot = live_dex_module.get_slot(pool.key)
    for tick in (slot.next_init_below, slot.next_init_above):
        if MIN_TICK < tick < MAX_TICK:      # sentinels bound the list
            key = live_dex_module.derive_tick_key(pool.key, tick)
            raw = live_dex_module._mapping_value("ticks", key)
            assert raw is not None, f"ticks[{tick}] unreachable via derived key"
            return
    pytest.skip("no non-sentinel initialized tick to probe")
