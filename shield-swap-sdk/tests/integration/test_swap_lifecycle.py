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

    # consumer_id must go on the PROVIDER, not just the network client: the
    # record scanner is built lazily from provider config, so setting it only
    # on network_client leaves the scanner with a key and no consumer — it
    # cannot mint a JWT and every record read answers Unauthorized.
    provider = HTTPProvider(ENDPOINT, network="testnet",
                            api_key=os.environ["ALEO_E2E_API_KEY"],
                            consumer_id=os.environ["ALEO_E2E_CONSUMER_ID"])
    aleo = Aleo(provider)
    acct = aleo.account.from_private_key(os.environ["ALEO_E2E_PRIVATE_KEY"])
    aleo.default_account = acct
    aleo.records.register(acct)
    dex = ShieldSwap(aleo)
    # get_route is auth-gated; without a session it answers 401 well before
    # anything is proved.
    dex.api.authenticate(str(acct.address),
                         lambda msg: str(aleo.account.sign(msg.encode(), acct)))

    # Pick a pool where the account holds a private balance of one side.
    pools = dex.api.get_pools()
    assert pools
    pool = pools[0]
    token_in = pool.token0
    # Records that FUND a token live in the underlying program (wrapped
    # assets) or the ARC-20 itself (plain).
    program_in = (pool.token0_info.underlying_program
                  or pool.token0_info.amm_token_program)

    balances = dex.get_private_balances([program_in], account=acct)
    amount_in = min(balances[program_in], 10 ** max(pool.token0_info.decimals - 2, 0))
    if amount_in == 0:
        pytest.skip(f"account holds no private {program_in} records to swap")

    # Use the SDK's own quote conversion rather than reimplementing it: the
    # route endpoint takes a CANONICAL decimal amount, so passing raw base
    # units quotes a trade 10**decimals too large and yields an
    # amount_out_min the pool cannot pay — proved, broadcast, rejected.
    expected = dex._quote_expected_out(
        token_in_id=token_in, token_out_id=pool.token1, amount_in=amount_in)

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


@write_tier
def test_wrapped_flow_roundtrip():
    """Cutover check (guide §7): a wrapped-side swap auto-routes through
    shield_swap_router, funded with UNDERLYING records; the claim of a
    wrapped output routes even when the swap started on the core."""
    from aleo import Aleo, HTTPProvider

    from aleo_shield_swap import ShieldSwap

    # consumer_id must go on the PROVIDER, not just the network client: the
    # record scanner is built lazily from provider config, so setting it only
    # on network_client leaves the scanner with a key and no consumer — it
    # cannot mint a JWT and every record read answers Unauthorized.
    provider = HTTPProvider(ENDPOINT, network="testnet",
                            api_key=os.environ["ALEO_E2E_API_KEY"],
                            consumer_id=os.environ["ALEO_E2E_CONSUMER_ID"])
    aleo = Aleo(provider)
    acct = aleo.account.from_private_key(os.environ["ALEO_E2E_PRIVATE_KEY"])
    aleo.default_account = acct
    aleo.records.register(acct)
    dex = ShieldSwap(aleo)
    # get_route is auth-gated; without a session it answers 401 well before
    # anything is proved.
    dex.api.authenticate(str(acct.address),
                         lambda msg: str(aleo.account.sign(msg.encode(), acct)))

    tokens = {t.address: t for t in dex.api.get_tokens()}
    case = None
    for pool in dex.api.get_pools():
        for token_in, token_out in ((pool.token0, pool.token1),
                                    (pool.token1, pool.token0)):
            tok = tokens.get(token_in)
            if (tok and tok.underlying_token_id
                    and tok.underlying_token_id != tok.address):
                case = (pool, token_in, tok)
                break
        if case:
            break
    if case is None:
        pytest.skip("no pool with a wrapped side on the new core yet")
    pool, token_in, tok = case

    # Registry and chain must agree before we spend anything on it.
    assert dex._is_wrapped(token_in), f"{tok.symbol}: registry/chain disagree"
    program_in = tok.underlying_program          # UNDERLYING records fund it

    balances = dex.get_private_balances([program_in], account=acct)
    amount_in = min(balances.get(program_in, 0),
                    10 ** max(tok.decimals - 2, 0))
    if amount_in == 0:
        pytest.skip(f"no private {program_in} records to fund the wrapped swap")

    handle = _with_retry(lambda: dex.swap(
        pool_key=pool.key, token_in_id=token_in, amount_in=amount_in,
        slippage_bps=500, token_in_program=program_in).delegate(acct))
    assert handle.swap_id and handle.blinded_address

    deadline = time.time() + 300
    while True:
        try:
            dex.get_swap_output(handle.swap_id)
            break
        except SwapOutputNotFinalizedError:
            if time.time() > deadline:
                pytest.fail("routed swap request did not finalize within 5 minutes")
            time.sleep(10)

    # Claim dispatches by the finalized SwapOutput's token shapes — the
    # wrapped refund side unwraps to the signer in the same transaction.
    result = _with_retry(lambda: dex.claim_swap_output(handle).delegate(acct))
    assert result.amount_out > 0
