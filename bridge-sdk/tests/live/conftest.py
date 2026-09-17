import os

import pytest

from aleo_bridge import Bridge


@pytest.fixture(scope="session")
def live_bridge() -> Bridge:
    if os.environ.get("BRIDGE_LIVE_READS") != "1":
        pytest.skip("set BRIDGE_LIVE_READS=1 (and BRIDGE_PRIVATE_KEY) to run read-only live checks")
    if not os.environ.get("BRIDGE_PRIVATE_KEY"):
        pytest.skip("BRIDGE_PRIVATE_KEY is required for Bridge.from_env()")
    bridge = Bridge.from_env()
    if bridge.environment != "mainnet":
        pytest.skip("live read checks target the mainnet routes; unset ALEO_NETWORK or set it to mainnet")
    return bridge
