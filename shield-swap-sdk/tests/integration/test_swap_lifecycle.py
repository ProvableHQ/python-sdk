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

from .conftest import ENDPOINT, dps_credentials, poll_until, write_tier

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


def _largest_record_amount(aleo, acct, program: str) -> int:
    """The biggest single unspent token record in *program*.

    A swap is funded by ONE record, so sizing from the summed private balance
    over-asks once holdings fragment across many small records (the account
    accumulates dust from earlier lifecycle runs).  0 when there are none.
    """
    from aleo_shield_swap._core import parse_token_record_info, record_plaintext
    amounts = []
    for rec in aleo.record_provider.find(acct, program=program, unspent=True):
        info = parse_token_record_info(record_plaintext(rec) or "")
        if info is not None:
            amounts.append(info["amount"])
    return max(amounts, default=0)


@write_tier
def test_private_swap_roundtrip():
    from aleo import Aleo, HTTPProvider

    from aleo_shield_swap import ShieldSwap

    # consumer_id must go on the PROVIDER, not just the network client: the
    # record scanner is built lazily from provider config, so setting it only
    # on network_client leaves the scanner with a key and no consumer — it
    # cannot mint a JWT and every record read answers Unauthorized.
    key, cid = dps_credentials()
    provider = HTTPProvider(ENDPOINT, network="testnet", api_key=key, consumer_id=cid)
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

    amount_in = min(_largest_record_amount(aleo, acct, program_in),
                    10 ** max(pool.token0_info.decimals - 2, 0))
    if amount_in == 0:
        pytest.skip(f"account holds no private {program_in} records to swap")

    # Use the SDK's own quote conversion rather than reimplementing it: the
    # route endpoint takes a CANONICAL decimal amount, so passing raw base
    # units quotes a trade 10**decimals too large and yields an
    # amount_out_min the pool cannot pay — proved, broadcast, rejected.
    expected = dex._quote_expected_out(
        token_in_id=token_in, token_out_id=pool.token1, amount_in=amount_in,
        pool_key=pool.key)

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

    # The claim consumed swap_outputs[swap_id]; a second claim is the
    # documented non-retryable absence (e2e.test.ts).
    assert poll_until(lambda: _absent(dex, handle), 24, 5)
    with pytest.raises(SwapOutputNotFinalizedError):
        dex.claim_swap_output(handle)

    # The fill receipt survives the claim: header + one hop for a single-pool
    # swap, lp_fee derived from the gross fee.
    receipt = dex.get_swap_execution(handle.swap_id)
    assert receipt is not None and receipt.swap_id == handle.swap_id
    assert len(receipt.hops) == 1
    hop = receipt.hops[0]
    assert hop.pool == pool.key and hop.amount_in == amount_in
    assert hop.amount_out == result.amount_out
    assert hop.lp_fee == hop.fee_paid - hop.protocol_fee >= 0
    assert receipt.executed_height > 0


def _absent(dex, handle) -> bool:
    try:
        dex.get_swap_output(handle.swap_id)
        return False
    except SwapOutputNotFinalizedError:
        return True


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
    key, cid = dps_credentials()
    provider = HTTPProvider(ENDPOINT, network="testnet", api_key=key, consumer_id=cid)
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

    amount_in = min(_largest_record_amount(aleo, acct, program_in),
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


@write_tier
def test_concurrent_swaps_reserve_distinct_identities(tmp_path, monkeypatch):
    """veil's ``runs two swaps concurrently`` + the blinded-identity store:
    two swaps in one batch reserve distinct counters from the journal (an
    unguarded pair would derive the same blinded address and the second
    would revert at finalize), and collect_all claims both."""
    from aleo_shield_swap import ShieldSwap

    key, cid = dps_credentials()
    monkeypatch.setenv("ALEO_E2E_API_KEY", key)
    monkeypatch.setenv("ALEO_E2E_CONSUMER_ID", cid)
    monkeypatch.setenv("SHIELD_SWAP_PRIVATE_KEY", os.environ["ALEO_E2E_PRIVATE_KEY"])
    dex = ShieldSwap.from_profile(tmp_path / "home")
    report = dex.onboard()                     # imports the funded account; no-op stages skip
    assert report.funded

    held = {tid: v for tid, v in dex.status().balances.items() if v.get("private", 0) > 0}
    pools = dex.api.get_pools()
    pool = next(p for p in pools if p.token0 in held or p.token1 in held)
    token_in = pool.token0 if pool.token0 in held else pool.token1
    dec = (pool.token0_info.decimals if token_in == pool.token0 else pool.token1_info.decimals)
    amount_in = min(held[token_in]["private"] // 1000, 10 ** max(dec - 2, 0))
    assert amount_in > 0

    batch = dex.swap_many(pool_key=pool.key, token_in_id=token_in,
                          amount_in=amount_in, count=2, slippage_bps=500)
    assert len(batch.handles) == 2, f"swap failures: {batch.failures}"
    assert len({h.blinded_address for h in batch.handles}) == 2
    assert len({h.swap_id for h in batch.handles}) == 2

    deadline = time.time() + 900
    claimed = 0
    while time.time() < deadline and claimed < 2:
        claimed += len(dex.collect_all().claimed)
        if claimed < 2:
            time.sleep(15)
    assert claimed == 2, "swap outputs never became claimable"
    assert dex.status().pending_claim_ids == []
    # A fresh journal for an account with swap history is seeded past the
    # counters already used on chain, so the two swaps sit at the end of the
    # cursor, not at 0 and 1 (which the chain would have rejected).
    events = dex.journal.events()
    swapped = sorted(e["counter"] for e in events if e["type"] == "swap")
    assert len(swapped) == 2 and swapped[1] == swapped[0] + 1
    assert dex.journal.counter_cursor() == swapped[1] + 1
    skipped = [e for e in events if e["type"] == "counters_skipped"]
    assert (swapped[0] == 0) == (not skipped)          # seeded iff history exists
