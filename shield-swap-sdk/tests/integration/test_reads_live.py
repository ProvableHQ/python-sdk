"""Read tier: EVERY read action against the live DEX API + testnet chain.

No credentials, no spending — run with ``-m live``.  Assertions target
invariants and shapes, not exact live figures (testnet state varies).
"""
from __future__ import annotations

import time

import pytest

from aleo_shield_swap.errors import (
    PoolNotFoundError,
    SwapOutputNotFinalizedError,
)
from .conftest import ENDPOINT, skip_if_access_gated
from aleo_shield_swap.tick_math import (
    MAX_SQRT_RATIO_X128,
    MAX_TICK,
    MIN_SQRT_RATIO_X128,
    MIN_TICK,
    u256_to_int,
)

pytestmark = pytest.mark.live



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
            assert entry.token0_info.amm_token_program.endswith(".aleo")


def test_api_get_tokens(live_dex_module):
    tokens = live_dex_module.api.get_tokens()
    assert tokens, "no tokens in the registry"
    for tok in tokens:
        assert tok.address.endswith("field")
        assert tok.decimals >= 0
        assert tok.symbol


def test_api_get_route_quotes_both_directions(live_dex_module, pool):
    # amount_in is a CANONICAL decimal amount — "1" means one whole token, not
    # 10**decimals base units. Passing base units quotes a trade 10**decimals
    # too large and returns a price from deep in the book.
    fwd = skip_if_access_gated(lambda: live_dex_module.api.get_route(
        token_in=pool.token0, token_out=pool.token1, amount_in="1"))
    assert fwd.token_in == pool.token0 and fwd.token_out == pool.token1
    assert fwd.hops, "route has no hops"
    rev = live_dex_module.api.get_route(
        token_in=pool.token1, token_out=pool.token0)
    assert rev.token_in == pool.token1


def test_api_get_ohlcv(live_dex_module, pool):
    # unix seconds, not ISO-8601: the API's from/to are int64 and reject a
    # timestamp string with 400.
    now = int(time.time())
    candles = skip_if_access_gated(lambda: live_dex_module.api.get_ohlcv(
        pool.key, granularity="1d",
        from_ts=now - 30 * 86_400, to_ts=now))
    for candle in candles:                     # may be empty on a quiet pool
        assert float(candle.h) >= float(candle.l)


def test_public_balances_are_chain_reads_for_any_address(live_dex_module):
    # Any (valid) address is queryable — a fresh key holds nothing, and an
    # absent mapping entry reads as 0 rather than an error.  Not the all-zero
    # burn address: the node rejects it as a mapping key with a 404.
    import aleo
    nobody = str(aleo.testnet.PrivateKey.random().address)
    programs = sorted({t.amm_token_program for t in live_dex_module.api.get_tokens()
                       if t.amm_token_program})
    assert programs
    balances = live_dex_module.get_public_balances(programs, address=nobody)
    assert set(balances) == set(programs)
    assert all(v == 0 for v in balances.values())


def test_swap_output_unknown_id_is_not_finalized(live_dex_module):
    # Swap detail is a chain read now (the API's /swaps routes are retired).
    with pytest.raises(SwapOutputNotFinalizedError):
        live_dex_module.get_swap_output("0field")


# ── Chain reads (node, via the facade) ───────────────────────────────────────

def test_get_pool_matches_api(live_dex_module, pool):
    chain_pool = live_dex_module.get_pool(pool.key)
    assert chain_pool.token0 == pool.token0
    assert chain_pool.token1 == pool.token1
    # New stack: raw native amounts — the scale fields are gone.
    assert not hasattr(chain_pool, "scale0")
    assert isinstance(chain_pool.enabled, bool)


def test_get_slot_invariants(live_dex_module, pool):
    slot = live_dex_module.get_slot(pool.key)
    sqrt_price = u256_to_int(slot.raw.sqrt_price)
    assert MIN_SQRT_RATIO_X128 <= sqrt_price <= MAX_SQRT_RATIO_X128
    assert MIN_TICK <= slot.tick <= MAX_TICK
    assert slot.tick_spacing > 0
    assert slot.next_init_below <= slot.tick <= slot.next_init_above
    d0 = pool.token0_info.decimals if pool.token0_info else 9
    d1 = pool.token1_info.decimals if pool.token1_info else 9
    assert slot.price(d0, d1) > 0


def test_registry_agrees_with_chain_on_wrappedness(live_dex_module):
    """Staging registry rows vs the chain's from_wrapper_token_id mapping:
    a token is wrapped exactly when its underlying token id differs from
    its own address."""
    checked = 0
    for tok in live_dex_module.api.get_tokens():
        if tok.underlying_token_id is None:
            continue
        registry_wrapped = tok.underlying_token_id != tok.address
        assert live_dex_module._is_wrapped(tok.address) == registry_wrapped, tok.symbol
        if registry_wrapped:
            assert tok.underlying_program != tok.amm_token_program
        checked += 1
    assert checked > 0, "registry exposed no underlying_token_id rows"


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


# ── veil reads.integration parity ────────────────────────────────────────────

class _VectorAccount:
    """The blinding test vectors' account (tests/test_blinding.py)."""

    class _VK:
        def to_scalar(self):
            return ("3349263049717637823474981214792818709117236390684139545647480917"
                    "22770623877scalar")

    view_key = _VK()
    address = "aleo1rhgdu77hgyqd3xjj8ucu3jj9r2krwz6mnzyd80gncr5fxcwlh5rsvzp9px"


def test_validation_reads_agree_with_the_api_fee_tier_registry(live_dex_module):
    """The API's fee-tier registry and the chain's fee_tiers /
    fee_to_tick_spacing / tick_spacings mappings describe the same table; an
    unregistered fee reads False rather than erroring."""
    dex = live_dex_module
    tiers = skip_if_access_gated(lambda: dex.api.get_fee_tiers())
    assert tiers
    fee = tiers[0].fee_tier
    assert dex._mapping_value("fee_tiers", f"{fee}u16") == "true"
    spacing = dex._mapping_value("fee_to_tick_spacing", f"{fee}u16")
    assert spacing and spacing.endswith("u32")
    assert dex._mapping_value("tick_spacings", spacing) == "true"
    assert dex._mapping_value("fee_tiers", "65535u16") in (None, "false")


def test_absence_reads_as_false_not_error(live_dex_module):
    dex = live_dex_module
    assert dex.is_pool_initialized("3" * 75 + "field") is False
    # A never-used address reads as absent.  (The zero address itself is
    # rejected by the node's mapping endpoint, so probe a fresh key instead.)
    import aleo as _aleo
    fresh = str(_aleo.testnet.PrivateKey.random().address)
    assert dex._mapping_value("used_blinded_addresses", fresh) is None
    assert dex._mapping_value("frozen_position", "444444444444444444field") is None
    assert dex._position_state("5" * 75 + "field") is None
    assert dex.get_swap_execution("2" * 75 + "field") is None


def test_tick_math_brackets_the_live_sqrt_price(live_dex_module, pool):
    """The Q128.128 table agrees with the chain: the live price sits inside
    its active tick's bracket."""
    from aleo_shield_swap.tick_math import get_sqrt_price_at_tick_x128, u256_to_int
    slot = live_dex_module.get_slot(pool.key)
    price = u256_to_int(slot.sqrt_price)
    assert get_sqrt_price_at_tick_x128(slot.tick) <= price
    assert price < get_sqrt_price_at_tick_x128(slot.tick + 1)


def test_fresh_blinded_identity_is_unused_on_chain(live_dex_module):
    from aleo_shield_swap.derivations import next_blinded_identity
    identity = next_blinded_identity(live_dex_module._aleo, _VectorAccount(),
                                     program=live_dex_module.program)
    assert identity.counter >= 0
    assert identity.blinding_factor.endswith("field")
    assert identity.blinded_address.startswith("aleo1")
    assert live_dex_module._mapping_value("used_blinded_addresses",
                                          identity.blinded_address) is None


def test_target_program_exposes_the_expected_mappings(live_dex_module):
    import requests
    res = requests.get(f"{ENDPOINT}/v2/testnet/program/{live_dex_module.program}/mappings",
                       timeout=30)
    res.raise_for_status()
    names = set(res.json())
    expected = {"pools", "slots", "swap_outputs", "used_blinded_addresses", "positions",
                "ticks", "global_paused", "token_allowed", "token_paused", "pair_paused",
                "frozen_position", "pool_creation_is_open", "from_wrapper_token_id",
                "to_wrapper_token_id",
                # 2026-09 additions
                "swap_execution_headers", "swap_execution_hops", "pool_creators"}
    assert expected <= names, sorted(expected - names)


def test_trade_controls_on_a_live_pool(live_dex_module, pool):
    """The gates the swap finalize checks: the pool is enabled, its tokens are
    allowed (create_pool hard-requires it), and the pause switches read as
    booleans (absent == not paused)."""
    dex = live_dex_module
    onchain = dex.get_pool(pool.key)
    assert onchain.enabled is True
    assert dex._mapping_value("token_allowed", str(onchain.token0)) == "true"
    assert dex._mapping_value("token_allowed", str(onchain.token1)) == "true"
    for mapping, key in (("global_paused", "true"),
                         ("token_paused", str(onchain.token0)),
                         ("token_paused", str(onchain.token1)),
                         ("pool_creation_is_open", "true")):
        assert dex._mapping_value(mapping, key) in (None, "true", "false"), (mapping, key)


def test_get_tick_via_slot_neighbours(live_dex_module, pool):
    from aleo_shield_swap.tick_math import MAX_TICK_SENTINEL
    dex = live_dex_module
    slot = dex.get_slot(pool.key)
    target = slot.next_init_above
    tick = dex._tick_info(pool.key, target)
    if tick is not None:                       # the sentinel itself has no entry
        assert tick.tick == target
        assert tick.liquidity_gross >= 0
        assert tick.prev < target < tick.next
    assert dex._tick_info(pool.key, MAX_TICK_SENTINEL + 1) is None


def test_next_blinded_identity_collides_where_reserved_counters_do_not(live_dex_module, tmp_path):
    """Why the journal exists (veil's blindedIdentityStore.e2e): two
    unguarded derivations from identical chain state return the SAME
    identity — the collision — while two reservations from a journal hand
    out different counters, both unused on chain."""
    from aleo_shield_swap.derivations import blinded_identity_at, next_blinded_identity
    from aleo_shield_swap.journal import Journal
    dex = live_dex_module
    acct = _VectorAccount()
    first = next_blinded_identity(dex._aleo, acct, program=dex.program)
    second = next_blinded_identity(dex._aleo, acct, program=dex.program)
    assert (first.counter, first.blinded_address) == (second.counter, second.blinded_address)

    journal = Journal(tmp_path / "j.jsonl")
    c0, c1 = journal.reserve_counters(2)
    assert c0 != c1
    a = blinded_identity_at(dex._aleo, acct, dex.program, c0)
    b = blinded_identity_at(dex._aleo, acct, dex.program, c1)
    assert a.blinded_address != b.blinded_address
