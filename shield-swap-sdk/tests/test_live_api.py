"""Live smoke tests against the deployed amm-api — auth handshake included.

Opt-in: set ``SHIELD_SWAP_LIVE=1`` to run (they hit the network). The funded
half additionally needs ``ALEO_E2E_PRIVATE_KEY`` — an account that has
redeemed an invite code (``/access/status`` → ``has_access: true``).

Verified live invariants:
  * challenge/verify works for ANY account — signature only, no funds;
  * gated endpoints 401 without a token and 403 without redeemed access;
  * with the e2e account: route quoting, OHLCV, and balances succeed.
"""
from __future__ import annotations

import os
import time

import pytest

import aleo
from aleo_shield_swap import ApiClient
from aleo_shield_swap.errors import DexApiError

pytestmark = pytest.mark.skipif(
    os.environ.get("SHIELD_SWAP_LIVE") != "1",
    reason="live API tests — set SHIELD_SWAP_LIVE=1 to run",
)


def _authed_client(pk) -> ApiClient:
    api = ApiClient()
    api.authenticate(str(pk.address), lambda msg: str(pk.sign(msg.encode())))
    return api


@pytest.fixture(scope="module")
def funded_api() -> ApiClient:
    key = os.environ.get("ALEO_E2E_PRIVATE_KEY")
    if not key:
        pytest.skip("ALEO_E2E_PRIVATE_KEY not set")
    pk = aleo.testnet.PrivateKey.from_string(key)
    api = _authed_client(pk)
    status = api._get("/access/status")["data"]
    if not status.get("has_access"):
        pytest.skip("e2e account has not redeemed an invite code")
    return api


def test_authenticate_needs_no_funds():
    pk = aleo.testnet.PrivateKey.random()
    api = _authed_client(pk)
    assert api._token and api._token.count(".") == 2  # JWT shape


def test_gated_endpoint_rejects_missing_token():
    api = ApiClient()
    pools = api.get_pools()
    with pytest.raises(DexApiError) as exc:
        api.get_route(token_in=pools[0].token0, token_out=pools[0].token1,
                      amount_in=1_000)
    assert exc.value.status == 401


def test_gated_endpoint_rejects_unredeemed_account():
    pk = aleo.testnet.PrivateKey.random()
    api = _authed_client(pk)
    assert api._get("/access/status")["data"]["has_access"] is False
    pools = api.get_pools()
    with pytest.raises(DexApiError) as exc:
        api.get_route(token_in=pools[0].token0, token_out=pools[0].token1,
                      amount_in=1_000)
    assert exc.value.status == 403


def test_funded_route_quote(funded_api):
    pools = funded_api.get_pools()
    assert pools
    # Quote against the first pool that has an executable route in either
    # direction — pools can legitimately lack one-directional liquidity.
    for p in pools:
        for tin, tout in ((p.token1, p.token0), (p.token0, p.token1)):
            try:
                route = funded_api.get_route(token_in=tin, token_out=tout,
                                             amount_in=1_000)
            except DexApiError as e:
                if e.status == 404:
                    continue
                raise
            assert route.hops
            assert route.token_in == tin
            return
    pytest.fail("no pool had an executable route in either direction")


def test_funded_ohlcv(funded_api):
    pool = funded_api.get_pools()[0]
    now = int(time.time())
    candles = funded_api.get_ohlcv(pool.key, granularity="1h",
                                   from_ts=now - 86_400, to_ts=now)
    assert isinstance(candles, list)  # may be empty on a quiet pool


def test_funded_balances(funded_api):
    key = os.environ["ALEO_E2E_PRIVATE_KEY"]
    addr = str(aleo.testnet.PrivateKey.from_string(key).address)
    balances = funded_api.get_public_balances(addr)
    assert isinstance(balances, list)
