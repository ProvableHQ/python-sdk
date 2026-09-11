"""The async clients against live testnet — the same reads the sync suites
prove, through ``AsyncApiClient`` and ``AsyncShieldSwap`` over ``AsyncAleo``
(httpx, the ``[async]`` extra).  Read-only; the account tier needs the e2e
key and self-provisioned scanner credentials."""
from __future__ import annotations

import os

import aleo as aleo_pkg
import pytest
from aleo import AsyncAleo, HTTPProvider

from aleo_shield_swap.api import AsyncApiClient, api_url_for
from aleo_shield_swap.async_client import AsyncShieldSwap
from aleo_shield_swap.errors import DexApiError, NotAuthenticatedError, SwapOutputNotFinalizedError

from .conftest import ENDPOINT, PRIVATE_KEY, account_tier, dps_credentials

pytestmark = pytest.mark.live


async def _fresh_api() -> AsyncApiClient:
    pk = aleo_pkg.testnet.PrivateKey.random()
    api = AsyncApiClient(api_url_for("testnet"))
    await api.authenticate(str(pk.address), lambda m: str(pk.sign(m.encode())))
    return api


def _async_dex(*, with_account: bool = False) -> AsyncShieldSwap:
    key, cid = dps_credentials() if (with_account and PRIVATE_KEY) else (
        os.environ.get("ALEO_E2E_API_KEY"), os.environ.get("ALEO_E2E_CONSUMER_ID"))
    a = AsyncAleo(HTTPProvider(ENDPOINT, network="testnet", api_key=key, consumer_id=cid))
    if with_account and PRIVATE_KEY:
        a.default_account = a.account.from_private_key(PRIVATE_KEY)
    return AsyncShieldSwap(a)


async def _scanner_retry(call, attempts: int = 3):
    """The hosted scanner is a remote service and AsyncRecordScanner rides
    httpx's 5 s default read timeout — a loaded scan of a few hundred
    records can trip it.  Retry the read a couple of times before failing."""
    import asyncio
    import httpx
    for attempt in range(attempts):
        try:
            return await call()
        except httpx.ReadTimeout:
            if attempt == attempts - 1:
                raise
            await asyncio.sleep(3)


async def test_async_api_public_gated_and_session_surface():
    api = AsyncApiClient(api_url_for("testnet"))
    pools = await api.get_pools()
    assert pools and (await api.get_tokens())
    with pytest.raises(NotAuthenticatedError):
        await api.get_fee_tiers()                      # gated, no credential
    cfg = await api.get_compliance()
    assert isinstance(cfg.global_paused, bool)
    stats = await api.get_pool_stats_batch([p.key for p in pools[:2]])
    assert stats and set(stats) <= {p.key for p in pools[:2]}

    authed = await _fresh_api()
    assert authed.is_authenticated
    assert (await authed.referral_status()).has_access is True
    assert await authed.get_fee_tiers()
    route = await authed.get_route(token_in=pools[0].token0, token_out=pools[0].token1,
                                   amount_in="0.01", pool_key=pools[0].key)
    assert route.hops
    me = await authed.get_session()
    refreshed = await authed.refresh_session()
    assert refreshed.address == me.address and authed._csrf == refreshed.csrf_token
    assert sum(s.current for s in await authed.list_sessions()) == 1
    out = await authed.logout()
    assert out.ended is True and not authed.is_authenticated
    with pytest.raises(NotAuthenticatedError):
        await authed.get_fee_tiers()                   # cookies cleared with the session


async def test_async_chain_reads_agree_with_the_sync_client(live_dex_module):
    adex = _async_dex()
    entry = live_dex_module.api.get_pools()[0]
    sync_pool = live_dex_module.get_pool(entry.key)
    pool = await adex.get_pool(entry.key)
    assert (pool.token0, pool.token1, pool.fee) == (sync_pool.token0, sync_pool.token1, sync_pool.fee)
    # Derivations are pure and local, so they stay synchronous on the async client.
    assert adex.derive_pool_key(pool.token0, pool.token1, pool.fee) == entry.key
    assert await adex.is_pool_initialized(entry.key) is True
    slot = await adex.get_slot(entry.key)
    assert slot.liquidity >= 0 and isinstance(slot.tick, int)
    creator = await adex.get_pool_creator(entry.key)
    assert creator is None or creator.startswith("aleo1")
    with pytest.raises(SwapOutputNotFinalizedError):
        await adex.get_swap_output("0field")
    assert await adex.get_swap_execution("0field") is None
    nobody = str(aleo_pkg.testnet.PrivateKey.random().address)
    programs = sorted({t.amm_token_program for t in live_dex_module.api.get_tokens()
                       if t.amm_token_program})
    balances = await adex.get_public_balances(programs, address=nobody)
    assert set(balances) == set(programs) and all(v == 0 for v in balances.values())


@account_tier
async def test_async_record_reads_for_the_e2e_account(account_dex):
    adex = _async_dex(with_account=True)
    acct = adex._aleo.default_account
    await adex._aleo.records.register(acct)
    tokens = await adex.api.get_tokens()
    programs = sorted({t.underlying_program or t.amm_token_program for t in tokens
                       if t.underlying_program or t.amm_token_program})
    # Bracket the async read with sync reads: when nothing moved in the
    # window (no concurrent write tier on this account) the two clients
    # must agree exactly; when something did, only the shape is comparable.
    sync_before = account_dex.get_private_balances(programs)
    private = await _scanner_retry(lambda: adex.get_private_balances(programs, account=acct))
    sync_after = account_dex.get_private_balances(programs)
    assert set(private) == set(programs) and all(v >= 0 for v in private.values())
    if sync_before == sync_after:
        assert private == sync_after
    balances = await _scanner_retry(adex.get_balances)
    assert balances and all(e["total"] == e["public"] + e["private"] for e in balances.values())
    owned = await _scanner_retry(adex.get_owned_positions)
    assert isinstance(owned, list)
    if owned:
        one = await adex.get_owned_position(owned[0].position_token_id)
        assert one is not None and one.pool_key == owned[0].pool_key
