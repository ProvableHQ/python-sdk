"""Live tiers, matching the main SDK's convention (test_live_e2e.py):
every module here sets ``pytestmark = pytest.mark.live`` — excluded from the
default run (pytest.ini addopts); opt in with ``-m live``.  The write tier
additionally skips without funded credentials."""
from __future__ import annotations

import os

import pytest

ENDPOINT = os.environ.get("ALEO_E2E_ENDPOINT", "https://api.provable.com")  # origin, no /v2
WRITE = all(os.environ.get(k) for k in
            ("ALEO_E2E_PRIVATE_KEY", "ALEO_E2E_API_KEY", "ALEO_E2E_CONSUMER_ID"))

write_tier = pytest.mark.skipif(not WRITE, reason="write-tier env vars not set")


def _make_live_dex():
    from aleo import Aleo, HTTPProvider

    from aleo_shield_swap import ShieldSwap

    # Pass the DPS credentials the write tier gates on: without them the
    # hosted record scanner answers Unauthorized, so every record read — and
    # therefore every write test — fails before it reaches the chain.
    aleo = Aleo(HTTPProvider(
        ENDPOINT, network="testnet",
        api_key=os.environ.get("ALEO_E2E_API_KEY"),
        consumer_id=os.environ.get("ALEO_E2E_CONSUMER_ID"),
    ))
    dex = ShieldSwap(aleo)
    # Some API endpoints are auth-gated (signature challenge/verify);
    # authentication alone grants access. Prefer the e2e account (it is the
    # funded one); fall back to a throwaway signature so the session layer
    # is still exercised. Gated tests skip on 401.
    pk = os.environ.get("ALEO_E2E_PRIVATE_KEY")
    acct = aleo.account.from_private_key(pk) if pk else aleo.account.create()
    try:
        dex.api.authenticate(
            str(acct.address),
            lambda msg: str(aleo.account.sign(msg.encode(), acct)),
        )
    except Exception:
        pass                     # auth endpoint down — gated tests will skip
    if pk and os.environ.get("ALEO_E2E_API_KEY"):
        try:
            aleo.default_account = acct
            aleo.records.register(acct)   # scanning needs a registration
        except Exception:
            pass                 # scanner down — record reads will surface it
    return dex


def skip_if_access_gated(call):
    """Run *call*; skip the test when the API says the account lacks access."""
    from aleo_shield_swap.errors import DexApiError

    try:
        return call()
    except DexApiError as exc:
        if exc.status in (401, 403):
            pytest.skip(f"DEX API access-gated for this account: {exc.body[:80]}")
        raise


@pytest.fixture
def live_dex():
    return _make_live_dex()


@pytest.fixture(scope="module")
def live_dex_module():
    """Module-scoped live client — read tests share pools/token fetches."""
    return _make_live_dex()
