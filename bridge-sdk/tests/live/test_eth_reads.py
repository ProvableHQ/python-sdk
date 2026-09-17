"""Read-only mainnet checks: quotes for all three Hyperlane routes, the xReserve deposit quote,
Mailbox.delivered, and a pinned-address balance read.

Gated by BRIDGE_LIVE_READS=1 and ETHEREUM_RPC_URL. Nothing here signs or needs a key — the
``Ethereum`` connection is built with no ``private_key``/``signer``, and the ``Bridge`` is built
over a real (keyless) ``aleo.Aleo`` facade so only the ``bridge.eth`` surface is exercised.

A public RPC's rate limiting (HTTP 429) or a transient 5xx is not a bug in this SDK, so those are
skips, not failures.
"""
import os

import pytest
import requests

from aleo_bridge.errors import InsufficientBalanceError
from aleo_bridge.eth import Ethereum

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("BRIDGE_LIVE_READS") != "1" or not os.environ.get("ETHEREUM_RPC_URL"),
        reason="set BRIDGE_LIVE_READS=1 and ETHEREUM_RPC_URL"),
]

ALEO_RECIPIENT = "aleo1kypwp5m7qtk9mwazgcpg0tq8aal23mnrvwfvug65qgcg9xvsrqgspyjm6n"
# The xReserve contract custodies deposited USDC, so its balance exceeds the 2 USDC minimum and its
# allowance reads are meaningful. Any funded mainnet address works; keep it read-only.
PINNED_SENDER = "0x8888888199b2Df864bf678259607d6D5EBb4e3Ce"
KNOWN_MESSAGE_ID = "0xc7c2c763ef846ff1583d9222d8ecbfc56da2e0cdcc9a63bc4bde51467644794d"  # delivered on Aleo


def _run(fn):
    """Run *fn*; a 429/5xx from the public RPC is an environment condition, not a test failure."""
    try:
        return fn()
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status == 429 or (status is not None and status >= 500):
            pytest.skip(f"public RPC rate-limited (HTTP {status})")
        raise
    except requests.exceptions.ConnectionError as exc:
        pytest.skip(f"public RPC unreachable: {exc}")
    except requests.exceptions.Timeout:
        pytest.skip("public RPC timed out")


@pytest.fixture(scope="module")
def eth():
    from aleo import Aleo, HTTPProvider

    from aleo_bridge import Bridge

    aleo = Aleo(HTTPProvider("https://edge.provable.com/api", network="mainnet"))  # no default_account: read-only
    bridge = Bridge(aleo, ethereum=Ethereum(os.environ["ETHEREUM_RPC_URL"]))
    return bridge.eth


def test_connection_is_mainnet_and_read_only(eth):
    chain_id = _run(lambda: eth.conn.chain_id)
    assert chain_id == 1 and not eth.conn.can_sign and eth.chain.id == "ethereum"


@pytest.mark.parametrize("asset,amount_atomic,router_type", [("eth", 1, "native"), ("wbtc", 1, "collateral"), ("usdt", 1, "collateral")])
def test_hyperlane_quotes_at_minimum_amounts(eth, asset, amount_atomic, router_type, record_property):
    q = _run(lambda: eth.quote_transfer_remote(asset, ALEO_RECIPIENT, amount_atomic=amount_atomic, sender=PINNED_SENDER))
    record_property(f"hyperlane_quote_{asset}", {
        "native_fee_atomic": q.native_fee_atomic, "native_value_atomic": q.native_value_atomic,
        "approval_required": q.approval_required})
    assert q.native_fee_atomic > 0 and q.plan.amount_atomic == amount_atomic
    if router_type == "native":
        assert q.native_value_atomic == amount_atomic + q.native_fee_atomic and q.approval_required is None
    else:
        assert q.native_value_atomic == q.native_fee_atomic and isinstance(q.approval_required, bool)
    assert q.fees[0].asset_id == "ethereum/eth" and q.fees[0].estimated


def test_xreserve_deposit_quote_read_only(eth, record_property):
    try:
        q = _run(lambda: eth.quote_deposit_usdc(ALEO_RECIPIENT, amount="2", sender=PINNED_SENDER))
    except InsufficientBalanceError as exc:
        # An accepted outcome (controller ruling): PINNED_SENDER's on-chain USDC balance may have
        # moved since this test was written. The exception itself proves the balance-check mechanics
        # ran (the SDK read PINNED_SENDER's USDC balance before building a quote).
        record_property("xreserve_quote_outcome", f"InsufficientBalanceError: {exc}")
        return
    record_property("xreserve_quote_outcome", {
        "route_id": q.plan.route_id, "max_fee_atomic": q.max_fee_atomic, "balance_atomic": q.balance_atomic})
    assert q.plan.route_id == "xreserve:ethereum/usdc->aleo/usdcx" and q.max_fee_atomic == 100_000
    assert q.hook_data == bytes(65) and len(q.remote_recipient_bytes32) == 32 and q.balance_atomic >= 2_000_000


def test_mailbox_delivered_reads(eth, record_property):
    delivered = _run(lambda: eth.is_delivered(KNOWN_MESSAGE_ID))
    record_property("mailbox_delivered_known_id", delivered)
    assert isinstance(delivered, bool)
    assert _run(lambda: eth.is_delivered("0x" + "00" * 32)) is False


def test_balance_pinned_sender(eth, record_property):
    value = _run(lambda: eth.balance("ethereum/eth", address=PINNED_SENDER))
    record_property("eth_balance_pinned_sender", value)
    assert isinstance(value, int) and value >= 0


def test_chain_status_read_only(eth):
    status = _run(lambda: eth.chain_status())
    assert status.chain_id == "ethereum" and status.address is None and status.balances == {}
