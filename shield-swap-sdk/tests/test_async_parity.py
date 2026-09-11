"""Sync/async surface parity.

``sdk/AGENTS.md``: "Every HTTP client ships both … never duplicate
orchestration."  A read method that exists on one client and not the other is a
gap, so this fails rather than letting the surfaces drift silently.
"""
from __future__ import annotations

from aleo_shield_swap.async_client import AsyncShieldSwap
from aleo_shield_swap.client import ShieldSwap

#: Sync-only by decision, not oversight — each needs a journal or the LP router
#: surface the async client does not carry yet.
SYNC_ONLY = {
    "from_profile", "onboard", "status",            # profile/journal bound
    "swap_many", "collect_all",                     # require a journal
    "create_pool", "mint", "increase_liquidity",    # LP surface
    "decrease_liquidity", "collect", "burn",
    "rebalance_position",
    "get_positions", "find_tick_predecessor",
}


def test_every_public_read_exists_on_both_clients():
    sync = {n for n in vars(ShieldSwap) if not n.startswith("_")}
    asyn = {n for n in vars(AsyncShieldSwap) if not n.startswith("_")}
    missing = sync - asyn - SYNC_ONLY
    assert not missing, f"present on ShieldSwap but not AsyncShieldSwap: {sorted(missing)}"


def test_owned_position_views_are_on_both():
    for name in ("get_owned_positions", "get_owned_position"):
        assert hasattr(ShieldSwap, name)
        assert hasattr(AsyncShieldSwap, name)


def test_sync_only_list_has_no_stale_entries():
    """Keep SYNC_ONLY honest: an entry that async has gained should be removed."""
    asyn = {n for n in vars(AsyncShieldSwap) if not n.startswith("_")}
    stale = SYNC_ONLY & asyn
    assert not stale, f"async now has these — drop from SYNC_ONLY: {sorted(stale)}"


def test_shared_write_verbs_agree_on_parameter_defaults():
    """Name parity is not enough: the async swap once defaulted
    deadline_offset_blocks to 100 (~5 min) while the sync one used 10,000
    (~8 h), so a delegated proof could outlive its own deadline.  Every
    parameter both clients share must carry the same default."""
    import inspect
    for name in ("swap", "claim_swap_output"):
        sync_params = inspect.signature(getattr(ShieldSwap, name)).parameters
        async_params = inspect.signature(getattr(AsyncShieldSwap, name)).parameters
        for pname, p in sync_params.items():
            if pname in async_params and p.default is not inspect.Parameter.empty:
                assert async_params[pname].default == p.default, (
                    f"{name}({pname}=): sync {p.default!r} vs async {async_params[pname].default!r}")
