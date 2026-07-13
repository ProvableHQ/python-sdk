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


@pytest.fixture
def live_dex():
    from aleo import Aleo, HTTPProvider

    from aleo_shield_swap import ShieldSwap

    aleo = Aleo(HTTPProvider(ENDPOINT, network="testnet"))
    return ShieldSwap(aleo)
