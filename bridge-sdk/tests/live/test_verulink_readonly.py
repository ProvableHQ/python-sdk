"""Read deployment, pause state, fee rates and limits without accounts or writes."""
import os

import pytest

from aleo import Aleo, HTTPProvider
from aleo_bridge import Bridge, Ethereum

pytestmark = [pytest.mark.live, pytest.mark.skipif(
    os.environ.get("BRIDGE_VERULINK_READ_ONLY") != "1",
    reason="set BRIDGE_VERULINK_READ_ONLY=1 for public deployment reads")]


@pytest.mark.parametrize("source,destination,recipient", [
    ("aleo", "bsc", "0x734C0a5AB55885974cEDb9D6ff71d8E8448c7375"),
    ("bsc", "aleo", "aleo1k5a999p502ty77ur8yryj7qmj2rtxyl8jylpnkcnfuvgyt0vxvyssyn69g"),
])
def test_public_verulink_quote(source, destination, recipient):
    aleo = Aleo(HTTPProvider(os.environ.get("ALEO_RPC_URL", "https://edge.provable.com/api"), network="mainnet"))
    bsc = Ethereum(os.environ.get("BSC_RPC_URL", "https://bsc-dataseed.bnbchain.org"))
    bridge = Bridge(aleo, evm={"bsc": bsc})
    quote = bridge.quote(source_chain=source, source_asset="aleo", destination_chain=destination,
                         amount="20", recipient=recipient)
    assert quote.amount_atomic == 20_000_000
    assert quote.amount_out and 0 <= quote.fee_atomic < quote.amount_atomic
    assert quote.plan.amount_out == quote.amount_out
    assert aleo.default_account is None and bsc.address is None
    print(source, "->", destination, "rate:", quote.rate, "received:", quote.amount_out)
