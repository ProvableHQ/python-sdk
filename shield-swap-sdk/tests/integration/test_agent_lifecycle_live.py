"""Live proof of the full agent lifecycle — fresh profile to collected swap.

Opt in: python -m pytest tests/integration/test_agent_lifecycle_live.py -m live
Env:    ALEO_E2E_PRIVATE_KEY     mints a fresh invite code (or set
                                 SHIELD_SWAP_INVITE_CODE explicitly)
        ALEO_E2E_API_KEY /
        ALEO_E2E_CONSUMER_ID     delegated-proving + scanner credentials
        ALEO_E2E_ENDPOINT        API origin (default https://api.provable.com)

One ordered test: onboarding a fresh account is rate-limited and slow, so
each phase asserts and feeds the next rather than re-onboarding.
"""
from __future__ import annotations

import json
import os
import time

import pytest

pytestmark = pytest.mark.live

_HAS_CODE_SOURCE = bool(os.environ.get("SHIELD_SWAP_INVITE_CODE")
                        or os.environ.get("ALEO_E2E_PRIVATE_KEY"))
_HAS_DPS = all(os.environ.get(k)
               for k in ("ALEO_E2E_API_KEY", "ALEO_E2E_CONSUMER_ID"))


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
@pytest.mark.skipif(not _HAS_DPS,
                    reason="delegated-proving env missing "
                           "(ALEO_E2E_API_KEY/ALEO_E2E_CONSUMER_ID)")
def test_full_lifecycle_from_fresh_profile(tmp_path):
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

    # ── Swaps: concurrent counters, journaled handles ───────────────────────
    batch = dex.swap_many(pool_key=pool.key, token_in_id=token_in,
                          amount_in=10**4, count=2)
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

    # ── Resumability: a brand-new client sees a clean, consistent state ────
    fresh = ShieldSwap.from_profile(tmp_path / "home")
    st2 = fresh.status()
    assert st2.pending_claim_ids == []
    assert st2.counter_cursor == 2


@pytest.mark.skipif(not _HAS_CODE_SOURCE,
                    reason="no invite code source")
def test_live_registration_without_dps_creds(tmp_path, monkeypatch):
    """The registration half of the lifecycle, provable without DPS creds:
    fresh account authenticates, redeems a minted invite, and the
    credentials stage fails with the instructive error (not a crash)."""
    from aleo_shield_swap import ShieldSwap
    from aleo_shield_swap.errors import CredentialsMissingError

    monkeypatch.delenv("ALEO_E2E_API_KEY", raising=False)
    monkeypatch.delenv("ALEO_E2E_CONSUMER_ID", raising=False)

    dex = ShieldSwap.from_profile(tmp_path / "home")
    with pytest.raises(CredentialsMissingError, match="ALEO_E2E_API_KEY"):
        dex.onboard(invite_code=_invite_code())

    st = dex.status()
    assert st.authenticated and st.has_access     # auth + redeem DID land
    stages = {e["name"]: e["action"] for e in dex.journal.events()
              if e["type"] == "stage"}
    assert stages["authenticate"] == "ran" and stages["redeem"] == "ran"
    # And the fresh JWT is persisted for the next session.
    assert json.loads((dex.profile.home / "credentials.json").read_text())["jwt"]
