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



# --- BAT / USDG / ZEC (veil PR #169 arc22-hyperlane.live.test.ts, read-only parts) -----------------

from tests.fakes.sealevel_fixtures import SPL_TRANSFER  # noqa: E402

SPL_SENDER = SPL_TRANSFER["senderAddress"]
SPL_ROUTES = ["hyperlane:solana/bat->aleo/bat", "hyperlane:solana/usdg->aleo/usdg", "hyperlane:solana/zec->aleo/zec"]


@pytest.mark.parametrize("route_id", SPL_ROUTES)
def test_live_spl_quote_excludes_the_token_amount_from_the_sol_total(bridge, route_id, record_property):
    route = bridge.registry.route(route_id)
    source = bridge.registry.asset(route.source_asset_id)
    quote = _run(lambda: bridge.sol.quote_transfer_remote(RECIPIENT, amount="0.0001", sender=SPL_SENDER, asset=source.id))
    assert quote.plan.route_id == route_id and quote.plan.amount_atomic == 10 ** (source.decimals - 4)
    assert quote.igp_lamports > 0 and quote.network_fee_lamports > 0 and quote.rent_lamports > 0
    assert quote.total_lamports == quote.igp_lamports + quote.network_fee_lamports + quote.rent_lamports
    assert all(f.asset_id == "solana/sol" for f in quote.fees)
    record_property(f"spl_quote_{source.key}", {"igp_lamports": quote.igp_lamports, "network_fee_lamports": quote.network_fee_lamports,
                                                "rent_lamports": quote.rent_lamports, "total_lamports": quote.total_lamports})


@pytest.mark.parametrize("route_id", SPL_ROUTES)
def test_live_spl_collateral_program_mint_and_escrow(bridge, route_id):
    """The warp program is executable, its token PDA is owned by it, the mint and escrow are owned by the
    pinned token program, the mint carries the registry decimals and the escrow holds that mint."""
    route = bridge.registry.route(route_id)
    source = bridge.registry.asset(route.source_asset_id)
    metadata = sl.solana_route_metadata(route)
    client = bridge.solana.client

    def info(address: str):
        value = _run(lambda: client.get_account_info(bridge.sol._pubkey(address), commitment="confirmed", encoding="base64")).value
        assert value is not None, f"{address} does not exist on mainnet"
        data = value.data
        return str(value.owner), (__import__("base64").b64decode(data[0]) if isinstance(data, (list, tuple)) else bytes(data))

    # the SDK's AccountInfo carries owner + data, not the executable flag: a deployed program is
    # the account owned by the BPF upgradeable loader
    program_owner, _ = info(metadata.warp_program_address)
    assert program_owner == "BPFLoaderUpgradeab1e11111111111111111111111"
    token_owner, _ = info(metadata.token_pda)
    assert token_owner == metadata.warp_program_address
    mint_owner, mint_data = info(metadata.collateral_mint_address)
    assert mint_owner == metadata.spl_token_program_address and metadata.collateral_mint_address == source.locator.value
    assert len(mint_data) >= 82 and mint_data[44] == source.decimals and mint_data[45] == 1      # initialized mint
    escrow_owner, escrow_data = info(metadata.escrow_pda)
    assert escrow_owner == metadata.spl_token_program_address
    assert len(escrow_data) >= 165 and sl.b58encode(escrow_data[:32]) == metadata.collateral_mint_address
    # the recorded sender's ZEC token account is readable through the SDK's balance reader
    assert bridge.sol.balance(source.id, address=SPL_SENDER) >= 0


def test_live_historical_zec_deposit_recovers_to_done_without_a_signer(bridge, record_property):
    """veil 'recovers the observed ZEC deposit and verifies canonical Aleo delivery without a signer'."""
    from aleo_bridge.errors import PollingTimeoutError

    checkpoint = {
        "version": 1,
        "route": {"id": "hyperlane:solana/zec->aleo/zec", "registryVersion": bridge.registry.version},
        "intent": {"source": {"chain": "solana", "asset": "zec"}, "destination": {"chain": "aleo", "asset": "zec"},
                   "bridgeProtocol": "hyperlane", "amount": "0.0001", "sender": SPL_SENDER,
                   "recipient": SPL_TRANSFER["recipientAleoAddress"]},
        "source": {"transactionId": SPL_TRANSFER["signature"]},
    }
    progress = _run(lambda: bridge.recover(checkpoint))
    if progress.next == "wait":
        try:
            progress = _run(lambda: bridge.wait(progress, timeout_seconds=45, poll_seconds=2))
        except PollingTimeoutError:
            pytest.skip("the public Solana RPC did not serve the historical transaction within 45 s")
    record_property("historical_zec_recovery", {"next": progress.next, "status": progress.receipt.status.value,
                                                 "message_id": progress.receipt.protocol_state.get("messageId")})
    assert progress.next == "done" and progress.receipt.status.value == "COMPLETED"
    assert progress.receipt.source_tx_id == SPL_TRANSFER["signature"]
    assert progress.receipt.protocol_state.get("messageId") == SPL_TRANSFER["source"].rsplit("/", 1)[-1]
