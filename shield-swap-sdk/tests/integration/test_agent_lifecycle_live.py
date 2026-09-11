"""Live proof of the full agent lifecycle — fresh profile to collected swap.

Opt in: python -m pytest tests/integration/test_agent_lifecycle_live.py -m live
Env:    SHIELD_SWAP_REFERRAL_CODE   optional — a referral code to credit; when
                                    set, the referral stage must run and
                                    record the referrer.  Access itself needs
                                    nothing beyond authentication.
Provable + DEX API credentials self-provision during onboarding.

One ordered test: onboarding a fresh account is rate-limited and slow, so
each phase asserts and feeds the next rather than re-onboarding.  Covers
all four stress-test flows: startup, discovery+swaps, liquidity
(mint/resize/collect), and collection.
"""
from __future__ import annotations

import os
import time

import pytest

pytestmark = pytest.mark.live


def test_full_lifecycle_from_fresh_profile(tmp_path, monkeypatch):
    # Prove the participant path: credentials must SELF-provision and the
    # profile key must be genuinely fresh (a developer shell may carry
    # SHIELD_SWAP_PRIVATE_KEY, which would import a funded, already-redeemed
    # account and void every "fresh account" assertion below).
    monkeypatch.delenv("ALEO_E2E_API_KEY", raising=False)
    monkeypatch.delenv("ALEO_E2E_CONSUMER_ID", raising=False)
    monkeypatch.delenv("SHIELD_SWAP_PRIVATE_KEY", raising=False)
    monkeypatch.delenv("SHIELD_SWAP_PRIVATE_KEY_FILE", raising=False)
    from aleo_shield_swap import ShieldSwap

    # ── Startup: fresh key material, full registration, airdrop ────────────
    # No code is required: authentication alone grants access.  A referral
    # code, when the environment offers one, is optional attribution.
    referral_code = os.environ.get("SHIELD_SWAP_REFERRAL_CODE")
    dex = ShieldSwap.from_profile(tmp_path / "home")
    report = dex.onboard(referral_code=referral_code)
    assert report.funded, f"onboard did not fund: {report.outcomes}"
    ran = {o.name for o in report.outcomes if o.action == "ran"}
    assert "authenticate" in ran and "credentials" in ran   # genuinely fresh
    assert ("referral" in ran) == bool(referral_code)

    # Idempotence: a second onboard is a no-op, even if the code is repeated.
    again = dex.onboard(referral_code=referral_code)
    assert all(o.action == "skipped" for o in again.outcomes)

    # ── Discovery: pools, balances, positions ──────────────────────────────
    st = dex.status()
    assert st.authenticated and st.has_access
    if referral_code:
        assert dex.api.referral_status().referred_by
    assert dex.api.my_referral_code()          # every account gets one
    held = {tid for tid, v in st.balances.items() if v.get("private", 0) > 0}
    assert held, "airdrop records not visible in private balances"
    pools = dex.api.get_pools()
    assert pools, "no pools available to trade"

    # Pick a pool whose tokens we actually hold (the conversation pattern).
    pool = next(p for p in pools if p.token0 in held or p.token1 in held)
    token_in = pool.token0 if pool.token0 in held else pool.token1
    # Raw native units (the AMM no longer scales): ~1e-5 of a token.
    d0 = pool.token0_info.decimals if pool.token0_info else 9
    d1 = pool.token1_info.decimals if pool.token1_info else 9
    dec_in = d0 if token_in == pool.token0 else d1

    # ── Swaps: concurrent counters, journaled handles ───────────────────────
    batch = dex.swap_many(pool_key=pool.key, token_in_id=token_in,
                          amount_in=10 ** max(dec_in - 5, 1), count=2)
    assert len(batch.handles) == 2, f"swap failures: {batch.failures}"
    assert len({h.blinded_address for h in batch.handles}) == 2

    # ── Collection: poll until both claims land ────────────────────────────
    deadline = time.monotonic() + 900
    claimed_total = 0
    while time.monotonic() < deadline and claimed_total < 2:
        result = dex.collect_all()
        claimed_total += len(result.claimed)
        if claimed_total < 2:
            time.sleep(15)
    assert claimed_total == 2, "swap outputs never became claimable"

    # ── Liquidity: mint, resize, collect the owed earnings ─────────────────
    lo, hi = dex.get_slot(pool.key).tick_range(width=4)
    # ~1e-7 of each token, raw native units.
    amt0, amt1 = 10 ** max(d0 - 7, 1), 10 ** max(d1 - 7, 1)

    # Right after claims, the scanner can still serve just-spent records; a
    # mint built on one is silently dropped.  Model the careful client:
    # verify the drop, let the scanner refresh, re-select records, retry.
    # "Confirmed" is not "landed": such a mint confirms as REJECTED with no
    # exception raised, so the positions mapping is the only truth.  An
    # attempt counts once its position is readable on chain; a dropped one
    # is retired from the journal so it leaves no phantom position behind.
    minted = None
    for attempt in range(3):
        try:
            candidate = dex.mint(pool_key=pool.key, tick_lower=lo, tick_upper=hi,
                                 amount0_desired=amt0,
                                 amount1_desired=amt1).delegate()
        except Exception:
            candidate = None
        if candidate is not None and candidate.position_token_id:
            landed_by = time.monotonic() + 120
            while time.monotonic() < landed_by:
                if dex._position_state(candidate.position_token_id) is not None:
                    minted = candidate
                    break
                time.sleep(10)
            if minted is not None:
                break
            dex.journal.record_position_burned(candidate.position_token_id,
                                               candidate.transaction_id)
        if attempt == 2:
            pytest.fail("mint never landed on chain in 3 attempts (stale records?)")
        time.sleep(60)               # scanner catches up; records re-scan
    assert minted.position_token_id, "mint returned no position id"
    assert any(v.position_token_id == minted.position_token_id
               for v in dex.get_positions())

    pos = dex._position_state(minted.position_token_id)
    assert pos is not None and pos.liquidity > 0

    # The freshly minted PositionNFT record must reach the scanner before a
    # resize can spend it — poll instead of failing on the immediate read.
    from aleo_shield_swap._core import find_position_plaintext
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        records = dex._aleo.record_provider.find(
            dex._aleo.default_account, program=dex.program, unspent=True)
        if find_position_plaintext(records, pool.key):
            break
        time.sleep(15)

    dex.decrease_liquidity(pool_key=pool.key,
                           liquidity_to_remove=pos.liquidity // 2).delegate()

    # The re-issued position record can lag the scanner — collect_all is
    # designed to be re-run until the owed amounts drain.
    deadline = time.monotonic() + 600
    fees = []
    while time.monotonic() < deadline and not fees:
        try:
            fees = dex.collect_all().fees
        except Exception:
            pass                     # stale record / transient — retry
        if not fees:
            time.sleep(20)
    assert fees, "LP earnings never became collectable"

    # ── Resumability: a brand-new client sees a clean, consistent state ────
    fresh = ShieldSwap.from_profile(tmp_path / "home")
    st2 = fresh.status()
    assert st2.pending_claim_ids == []
    assert st2.counter_cursor == 2
    assert len(st2.open_positions) == 1        # the minted position, journaled
