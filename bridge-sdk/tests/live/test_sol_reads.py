"""Read-only mainnet checks for the Solana side. Gate: BRIDGE_LIVE_READS=1.

No key, no funds: decodes the live inner IGP account and quotes leg 11 (SOL -> Aleo, 1 lamport) for a
pinned sender. Fund-moving legs 11-12 run from scripts/rehearse.py (plan 4).

A public RPC's rate limiting (HTTP 429) or a transient 5xx is not a bug in this SDK, so those are
skips, not failures (mirrors tests/live/test_eth_reads.py's ``_run`` helper).
"""
import os
import re

import pytest

pytest.importorskip("solders")

from aleo import Aleo, HTTPProvider

from aleo_bridge import Bridge, Solana
from aleo_bridge import _sealevel as sl
from aleo_bridge.errors import BridgeError
from tests.fakes.sealevel_fixtures import ALEO_MAINNET_DOMAIN, DESTINATION_GAS_AMOUNT, TRANSFER, igp_account_data

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("BRIDGE_LIVE_READS") != "1", reason="set BRIDGE_LIVE_READS=1 to hit mainnet RPCs"),
]

ALEO_ENDPOINT = os.environ.get("ALEO_ENDPOINT", "https://edge.provable.com/api")
SENDER = os.environ.get("BRIDGE_LIVE_SOLANA_SENDER", TRANSFER["senderAddress"])
RECIPIENT = os.environ.get("BRIDGE_LIVE_ALEO_RECIPIENT", TRANSFER["recipientAleoAddress"])

_HTTP_STATUS_RE = re.compile(r"HTTP status (\d+)")


def _run(fn):
    """Run *fn*; a 429/5xx (or an unreachable public RPC) is an environment condition, not a test failure."""
    try:
        return fn()
    except BridgeError as exc:
        message = str(exc)
        match = _HTTP_STATUS_RE.search(message)
        if match and (int(match.group(1)) == 429 or int(match.group(1)) >= 500):
            pytest.skip(f"public RPC rate-limited or unavailable ({message})")
        if "request failed:" in message:
            pytest.skip(f"public RPC unreachable: {message}")
        raise


@pytest.fixture(scope="module")
def bridge() -> Bridge:
    aleo = Aleo(HTTPProvider(ALEO_ENDPOINT, network="mainnet"))
    return Bridge(aleo, solana=Solana(os.environ.get("SOLANA_RPC_URL")))


def test_live_igp_account_decodes_with_the_fixture_shape(bridge):
    metadata = _run(lambda: bridge.sol.metadata())
    live = _run(lambda: bridge.sol._account_data(metadata.igp_account))
    assert live is not None
    account = sl.decode_igp_account(live)
    recorded = sl.decode_igp_account(igp_account_data())
    assert account.bump == recorded.bump and account.beneficiary == recorded.beneficiary
    assert account.owner == recorded.owner
    oracle = account.gas_oracles[ALEO_MAINNET_DOMAIN]
    assert oracle.token_decimals == recorded.gas_oracles[ALEO_MAINNET_DOMAIN].token_decimals == 6
    assert oracle.token_exchange_rate > 0 and oracle.gas_price > 0
    lamports = sl.quote_igp_lamports(live, ALEO_MAINNET_DOMAIN, DESTINATION_GAS_AMOUNT)
    assert 0 < lamports < 1_000_000_000          # sanity: below 1 SOL; the recorded value was 2_900_000


def test_live_leg_11_quote_for_a_pinned_sender(bridge, record_property):
    quote = _run(lambda: bridge.sol.quote_transfer_remote(RECIPIENT, amount_atomic=1, sender=SENDER))
    assert quote.plan.route_id == sl.SOLANA_ROUTE_ID and quote.plan.sender == SENDER
    assert quote.igp_lamports > 0 and quote.network_fee_lamports > 0
    client = bridge.solana.client
    rents = [int(_run(lambda n=n: client.get_minimum_balance_for_rent_exemption(n)).value) for n in (141, 194, 0)]
    assert quote.rent_lamports == sum(rents)
    assert quote.total_lamports == 1 + quote.igp_lamports + quote.network_fee_lamports + quote.rent_lamports
    record_property("leg_11_quote", {
        "igp_lamports": quote.igp_lamports, "network_fee_lamports": quote.network_fee_lamports,
        "rent_lamports": quote.rent_lamports, "total_lamports": quote.total_lamports})
    print(f"leg 11 quote: igp={quote.igp_lamports} fee={quote.network_fee_lamports} "
          f"rent={quote.rent_lamports} total={quote.total_lamports}")
