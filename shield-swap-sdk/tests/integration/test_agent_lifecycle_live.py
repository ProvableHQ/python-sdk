"""Live proof of the full agent lifecycle — fresh profile to collected swap.

Opt in: python -m pytest tests/integration/test_agent_lifecycle_live.py -m live
Env:    ALEO_E2E_PRIVATE_KEY     mints a fresh invite code (or set
                                 SHIELD_SWAP_INVITE_CODE explicitly)
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

_HAS_CODE_SOURCE = bool(os.environ.get("SHIELD_SWAP_INVITE_CODE")
                        or os.environ.get("ALEO_E2E_PRIVATE_KEY"))


def _invite_code() -> str:
    explicit = os.environ.get("SHIELD_SWAP_INVITE_CODE")
    if explicit:
        return explicit
    import aleo

    from aleo_shield_swap import ApiClient

    pk = aleo.testnet.PrivateKey.from_string(os.environ["ALEO_E2E_PRIVATE_KEY"])
    api = ApiClient()
    api.authenticate(str(pk.address), lambda m: str(pk.sign(m.encode())))
    return api._post("/access/generate", {"count": 1})["data"]["codes"][0]


@pytest.mark.skipif(not _HAS_CODE_SOURCE,
                    reason="no invite code source (SHIELD_SWAP_INVITE_CODE "
                           "or ALEO_E2E_PRIVATE_KEY)")
def test_full_lifecycle_from_fresh_profile(tmp_path, monkeypatch):
    # Prove the participant path: credentials must SELF-provision.
    monkeypatch.delenv("ALEO_E2E_API_KEY", raising=False)
    monkeypatch.delenv("ALEO_E2E_CONSUMER_ID", raising=False)
    from aleo_shield_swap import ShieldSwap

    # ── Startup: fresh key material, full registration, airdrop ────────────
    dex = ShieldSwap.from_profile(tmp_path / "home")
    report = dex.onboard(invite_code=_invite_code())
    assert report.funded, f"onboard did not fund: {report.outcomes}"
    ran = {o.name for o in report.outcomes if o.action == "ran"}
    assert "authenticate" in ran and "redeem" in ran   # genuinely fresh

    # Idempotence: a second onboard is a no-op.
    again = dex.onboard()
    assert all(o.action == "skipped" for o in again.outcomes)

    # ── Discovery: pools, balances, positions ──────────────────────────────
    st = dex.status()
    assert st.authenticated and st.has_access
    held = {tid for tid, v in st.balances.items() if v.get("private", 0) > 0}
    assert held, "airdrop records not visible in private balances"
    pools = dex.api.get_pools()
    assert pools, "no pools available to trade"

    # Pick a pool whose tokens we actually hold (the conversation pattern).
    pool = next(p for p in pools if p.token0 in held or p.token1 in held)
    token_in = pool.token0 if pool.token0 in held else pool.token1
    state = dex.get_pool(pool.key)
    scale_in = state.scale0 if token_in == pool.token0 else state.scale1

    # ── Swaps: concurrent counters, journaled handles ───────────────────────
    batch = dex.swap_many(pool_key=pool.key, token_in_id=token_in,
                          amount_in=10**4 * int(scale_in), count=2)
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
    scale0, scale1 = int(state.scale0), int(state.scale1)
    minted = dex.mint(pool_key=pool.key, tick_lower=lo, tick_upper=hi,
                      amount0_desired=100 * scale0,
                      amount1_desired=100 * scale1).delegate()
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
