"""Read tier: live chain + live DEX API, no credentials, no spending."""
import pytest

from aleo_shield_swap.tick_math import MAX_TICK, MIN_TICK

pytestmark = pytest.mark.live


def test_pools_and_slots_live(live_dex):
    pools = live_dex.api.get_pools()
    assert pools, "no pools on the DEX API"
    pool = pools[0]
    assert pool.key.endswith("field")

    slot = live_dex.get_slot(pool.key)
    assert slot.sqrt_price > 0
    assert MIN_TICK <= slot.tick <= MAX_TICK


def test_local_pool_key_matches_indexer(live_dex):
    """The strongest cheap invariant: local derivation == indexer's key."""
    for entry in live_dex.api.get_pools():
        fee_pips = int(float(entry.pool.fee) * 100)   # API serves fee as bps string
        derived = live_dex.derive_pool_key(entry.token0, entry.token1, fee_pips)
        if derived == entry.key:
            return
    pytest.fail("no pool's key matched the local derivation — check fee units")
