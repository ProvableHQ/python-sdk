"""Read-only mainnet check: every pause switch on every active route resolves.

Gated by BRIDGE_LIVE_READS=1. Keyless facade; no Ethereum or Solana connection is configured, so the
registry's public read endpoints and the default Solana RPC are the ones exercised. Which routes are
paused is printed, not asserted: operators flip those switches (on 2026-10-30 every Hyperlane warp
program except USDT was paused). A transient public-RPC failure after the retry budget is a skip, not
a failure in this SDK.
"""
import os
import re

import pytest

from aleo_bridge import Bridge
from aleo_bridge.registry import DEFAULT_REGISTRY

pytestmark = [pytest.mark.live, pytest.mark.slow,
              pytest.mark.skipif(os.environ.get("BRIDGE_LIVE_READS") != "1", reason="set BRIDGE_LIVE_READS=1")]

TRANSIENT = re.compile(r"HTTP status (429|5\d\d)|ConnectionError|Timeout|request failed")
ACTIVE_MAINNET = [r.id for r in DEFAULT_REGISTRY.routes(environment="mainnet") if r.active]


@pytest.fixture(scope="module")
def keyless_bridge() -> Bridge:
    from aleo import Aleo, HTTPProvider
    return Bridge(Aleo(HTTPProvider("https://edge.provable.com/api", network="mainnet")), ethereum=None, solana=None)


@pytest.mark.parametrize("route_id", ACTIVE_MAINNET)
def test_every_switch_on_every_active_route_resolves(keyless_bridge, route_id):
    status = keyless_bridge.pause_status(route_id)
    for s in status.switches:
        if s.state == "unknown" and TRANSIENT.search(s.detail or ""):
            pytest.skip(f"public RPC unavailable: {s.detail}")
    assert status.switches, "a route with no switches would never be gated"
    assert not status.unknown, status.reason
    print(route_id, "PAUSED" if status.paused else "open", status.reason)
