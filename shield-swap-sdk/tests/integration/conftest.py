"""Live tiers, matching the main SDK's convention (test_live_e2e.py):
every module here sets ``pytestmark = pytest.mark.live`` — excluded from the
default run (pytest.ini addopts); opt in with ``-m live``.

Tiers (mirroring veil's ``test/integration`` gates):

* read-only live      — network only; no account needed
* live + account      — ``ALEO_E2E_PRIVATE_KEY``; record scans need Provable
                        API credentials, which are imported from
                        ``ALEO_E2E_API_KEY``/``ALEO_E2E_CONSUMER_ID`` when set
                        and otherwise SELF-PROVISIONED once per session via the
                        keyless ``POST /consumers`` (what onboarding does)
* write (``slow``)    — the account's tier, and it spends testnet funds
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable, Optional

import pytest

ENDPOINT = os.environ.get("ALEO_E2E_ENDPOINT", "https://api.provable.com")  # origin, no /v2
PRIVATE_KEY = os.environ.get("ALEO_E2E_PRIVATE_KEY")

account_tier = pytest.mark.skipif(not PRIVATE_KEY, reason="ALEO_E2E_PRIVATE_KEY not set")
write_tier = account_tier

_CREDS: Optional[tuple[str, str]] = None


def dps_credentials() -> tuple[str, str]:
    """``(api_key, consumer_id)`` for the scanner + delegated proving.

    Env first; otherwise provisioned once per session and cached (a fresh
    consumer per run is fine — the key/id pair is what matters, not the name).
    """
    global _CREDS
    if _CREDS is None:
        key = os.environ.get("ALEO_E2E_API_KEY")
        cid = os.environ.get("ALEO_E2E_CONSUMER_ID")
        if not (key and cid):
            from aleo_shield_swap.lifecycle import provision_provable_credentials
            key, cid = provision_provable_credentials(
                ENDPOINT, f"shield-swap-itest-{int(time.time())}")
        _CREDS = (key, cid)
    return _CREDS


def _make_live_dex(*, with_account: bool = False):
    from aleo import Aleo, HTTPProvider

    from aleo_shield_swap import ShieldSwap

    # Credentials go on the PROVIDER: the record scanner is built lazily from
    # provider config, so a key without a consumer cannot mint a JWT and every
    # record read answers Unauthorized.
    key, cid = dps_credentials() if (with_account and PRIVATE_KEY) else (
        os.environ.get("ALEO_E2E_API_KEY"), os.environ.get("ALEO_E2E_CONSUMER_ID"))
    aleo = Aleo(HTTPProvider(ENDPOINT, network="testnet", api_key=key, consumer_id=cid))
    dex = ShieldSwap(aleo)
    # Authentication alone grants API access (no invite gate).  Prefer the e2e
    # account (the funded one); fall back to a throwaway signature so the
    # session layer is still exercised.
    acct = aleo.account.from_private_key(PRIVATE_KEY) if PRIVATE_KEY else aleo.account.create()
    try:
        dex.api.authenticate(str(acct.address),
                             lambda msg: str(aleo.account.sign(msg.encode(), acct)))
    except Exception:
        pass                     # auth endpoint down — gated tests will surface it
    if with_account and PRIVATE_KEY:
        aleo.default_account = acct
        aleo.records.register(acct)      # scanning needs a registration
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


def poll_until(predicate: Callable[[], Any], tries: int, seconds: float) -> bool:
    """veil's ``pollUntil``: call *predicate* up to *tries* times *seconds*
    apart; True on the first truthy result, False when exhausted."""
    for i in range(tries):
        if predicate():
            return True
        if i < tries - 1:
            time.sleep(seconds)
    return False


def with_retry(fn, attempts=3, delay=5.0):
    last = None
    for _ in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - live services flake
            last = exc
            time.sleep(delay)
    raise last  # type: ignore[misc]


@pytest.fixture
def live_dex():
    return _make_live_dex()


@pytest.fixture(scope="module")
def live_dex_module():
    """Module-scoped live client — read tests share pools/token fetches."""
    return _make_live_dex()


@pytest.fixture(scope="module")
def account_dex():
    """Module-scoped client bound to the e2e account with a registered
    scanner and delegated-proving credentials (account/write tiers)."""
    if not PRIVATE_KEY:
        pytest.skip("ALEO_E2E_PRIVATE_KEY not set")
    return _make_live_dex(with_account=True)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Expose each phase's report on the item so order-dependent lifecycles
    can abort after the first failure instead of burning fees on doomed
    follow-up writes."""
    outcome = yield
    setattr(item, f"rep_{call.when}", outcome.get_result())
