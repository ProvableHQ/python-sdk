"""Known-answer test: the Sealevel ``HyperlaneToken`` account's ``remote_routers`` map decodes from live mainnet dumps."""
import base64
import json
from pathlib import Path

import pytest

from aleo_bridge._sealevel import decode_hyperlane_token_remote_routers
from aleo_bridge.errors import ConfigurationError
from aleo_bridge.registry import DEFAULT_REGISTRY

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("name", ["sol", "bat"])
def test_decodes_remote_routers_and_finds_the_aleo_entry(name):
    fx = json.loads((FIXTURES / f"sealevel-token-pda-{name}.json").read_text())
    routers = decode_hyperlane_token_remote_routers(base64.b64decode(fx["data_base64"]))
    aleo_domain = DEFAULT_REGISTRY.chain("aleo").protocol_domains["hyperlane"]
    assert aleo_domain in routers and len(routers[aleo_domain]) == 32 and routers[aleo_domain] != bytes(32)
    assert all(len(v) == 32 for v in routers.values())
    # The entry is the Aleo warp program's address as bytes32. Pinning its exact value is left to the
    # builder if the aleo facade exposes a program-address helper; presence and shape are the KAT here.


def test_short_or_corrupt_data_is_a_configuration_error():
    with pytest.raises(ConfigurationError, match="malformed Sealevel token account"):
        decode_hyperlane_token_remote_routers(b"\x01" + b"\x00" * 10)
    with pytest.raises(ConfigurationError, match="malformed Sealevel token account"):
        decode_hyperlane_token_remote_routers(b"\x00" + b"\x00" * 200)       # not initialized
