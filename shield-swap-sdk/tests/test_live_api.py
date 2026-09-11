"""Live smoke tests against the deployed amm-api — auth handshake included.

Opt-in: set ``SHIELD_SWAP_LIVE=1`` to run (they hit the network). The funded
half additionally needs ``ALEO_E2E_PRIVATE_KEY`` — the funded e2e account.

Verified live invariants:
  * challenge/verify works for ANY account — signature only, no funds;
  * gated endpoints 401 without a session; authentication alone grants
    access (no invite gate — a fresh key has ``has_access: true``);
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
def e2e_address() -> str:
    key = os.environ.get("ALEO_E2E_PRIVATE_KEY")
    if not key:
        pytest.skip("ALEO_E2E_PRIVATE_KEY not set")
    return str(aleo.testnet.PrivateKey.from_string(key).address)


@pytest.fixture(scope="module")
def funded_api() -> ApiClient:
    key = os.environ.get("ALEO_E2E_PRIVATE_KEY")
    if not key:
        pytest.skip("ALEO_E2E_PRIVATE_KEY not set")
    pk = aleo.testnet.PrivateKey.from_string(key)
    api = _authed_client(pk)
    assert api.referral_status().has_access    # authentication is the gate
    return api


def test_authenticate_needs_no_funds():
    pk = aleo.testnet.PrivateKey.random()
    api = _authed_client(pk)
    # The session rides as an httpOnly cookie + CSRF token (legacy
    # deployments returned a body JWT) — either way the credential is held
    # and the gated status probe answers.
    assert api.is_authenticated
    assert api.referral_status().has_access is True


def test_gated_endpoint_rejects_missing_token():
    api = ApiClient()
    pools = api.get_pools()
    with pytest.raises(DexApiError) as exc:
        api.get_route(token_in=pools[0].token0, token_out=pools[0].token1,
                      amount_in=1_000)
    assert exc.value.status == 401


def test_authentication_alone_grants_access():
    # No invite gate: a never-redeemed key has access the moment it
    # authenticates, owns a referral code to share, and can mint a durable
    # API token.
    pk = aleo.testnet.PrivateKey.random()
    api = _authed_client(pk)
    status = api.referral_status()
    assert status.has_access is True and status.referred_by is None
    assert api.my_referral_code()               # issued on first login
    assert api.create_api_token("probe").token


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


def test_funded_fee_tiers(funded_api):
    tiers = funded_api.get_fee_tiers()
    assert tiers and all(isinstance(t.fee_tier, int) for t in tiers)


def test_referral_status_typed(funded_api):
    assert funded_api.referral_status().has_access is True


def test_airdrop_request_is_rate_limit_tolerant(funded_api, e2e_address):
    from aleo_shield_swap.errors import AirdropRateLimitedError
    try:
        start = funded_api.request_airdrop(e2e_address)
        job = funded_api.get_airdrop_job(start.job_id)
        assert job.status in ("running", "complete")
    except AirdropRateLimitedError:
        pass                             # claimed recently — contract confirmed
