"""Write tier: the full private-swap lifecycle on the REAL testnet.

Spends real testnet funds (DPS fee master pays proving fees). Mirrors the
main SDK's live-e2e conventions: pytest.mark.live + credential skips.
"""
from __future__ import annotations

import os
import time

import pytest

from aleo_shield_swap.errors import SwapOutputNotFinalizedError
from aleo_shield_swap.types import SwapHandle

from .conftest import ENDPOINT, write_tier

pytestmark = [pytest.mark.live, pytest.mark.slow]


def _with_retry(fn, attempts=3, delay=5.0):
    last = None
    for _ in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - live services flake
            last = exc
            time.sleep(delay)
    raise last


@write_tier
def test_private_swap_roundtrip():
    from aleo import Aleo, HTTPProvider

    from aleo_shield_swap import ShieldSwap

    provider = HTTPProvider(ENDPOINT, network="testnet",
                            api_key=os.environ["ALEO_E2E_API_KEY"])
    aleo = Aleo(provider)
    aleo.network_client.consumer_id = os.environ["ALEO_E2E_CONSUMER_ID"]
    acct = aleo.account.from_private_key(os.environ["ALEO_E2E_PRIVATE_KEY"])
    aleo.default_account = acct
    aleo.records.register(acct)
    dex = ShieldSwap(aleo)

    # Pick a pool where the account holds a private balance of one side.
    pools = dex.api.get_pools()
    assert pools
    pool = pools[0]
    token_in = pool.token0
    program_in = pool.token0_info.wrapper_program

    balances = dex.get_private_balances([program_in], account=acct)
    amount_in = min(balances[program_in], 10 ** max(pool.token0_info.decimals - 2, 0))
    if amount_in == 0:
        pytest.skip(f"account holds no private {program_in} records to swap")

    route = dex.api.get_route(token_in=token_in, token_out=pool.token1,
                              amount_in=amount_in)
    expected = (int(float(route.estimated_amount_out) * 10 ** pool.token1_info.decimals)
                if route.estimated_amount_out else None)

    handle = _with_retry(lambda: dex.swap(
        pool_key=pool.key, token_in_id=token_in, amount_in=amount_in,
        expected_out=expected, slippage_bps=100,
        token_in_program=program_in).delegate(acct))
    assert isinstance(handle, SwapHandle)
    assert handle.swap_id and handle.blinded_address
    assert SwapHandle.from_json(handle.to_json()) == handle

    # Poll until the request finalizes (~a few blocks).
    deadline = time.time() + 300
    while True:
        try:
            dex.get_swap_output(handle.swap_id)
            break
        except SwapOutputNotFinalizedError:
            if time.time() > deadline:
                pytest.fail("swap request did not finalize within 5 minutes")
            time.sleep(10)

    result = _with_retry(lambda: dex.claim_swap_output(handle).delegate(acct))
    assert result.amount_out > 0
