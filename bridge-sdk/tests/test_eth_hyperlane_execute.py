import dataclasses

import pytest
from eth_account import Account
from eth_utils import keccak
from web3 import Web3

from aleo_bridge.errors import (BridgeError, ConfigurationError, InvalidRecipientError,
                                RegistryVersionMismatchError, RouteUnavailableError)
from aleo_bridge.eth import Ethereum
from aleo_bridge.types import DispatchReceipt, Status
from tests.fakes.fake_web3 import ZERO_ADDRESS, dispatch_id_log, event_log, fake_web3, make_bridge

KEY = "0x" + "11" * 32
ACCT = Account.from_key(KEY)
ALEO = "aleo1kypwp5m7qtk9mwazgcpg0tq8aal23mnrvwfvug65qgcg9xvsrqgspyjm6n"
OTHER_ALEO = "aleo1rhgdu77hgyqd3xjj8ucu3jj9r2krwz6mnzyd80gncr5fxcwlh5rsvzp9px"
OTHER = "0x0000000000000000000000000000000000000009"
ALEO_BYTES32 = "0xb102e0d37e02ec5dbba2460287ac07ef7ea8ee636392ce235402308299901811"
ETH_ROUTER = "0x38D447694f5c1f773ae3132cf93bF30B7Ec1Fa5A"
WBTC, WBTC_ROUTER = "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599", "0x20CDC85778b732073F7EecEF3DF25c0d310f8772"
USDT, USDT_ROUTER = "0xdAC17F958D2ee523a2206206994597C13D831ec7", "0x3C2064D78e4578E8F936E3db42aEF044E33FBF31"
MAILBOX = "0xc005dc82818d67AF737725bD4bf75435d065D239"
MESSAGE_ID = bytes.fromhex("ab" * 32)
APPROVE = keccak(text="approve(address,uint256)")[:4].hex()
TRANSFER_REMOTE = keccak(text="transferRemote(uint32,bytes32,uint256)")[:4].hex()


def setup(router, *, with_dispatch_log=True, **config):
    w3 = fake_web3(**config)
    if with_dispatch_log:
        w3.provider.receipt_logs = lambda tx: (
            [dispatch_id_log(MAILBOX, MESSAGE_ID, tx_hash=tx["hash"])] if tx["to"] == Web3.to_checksum_address(router) else [])
    bridge = make_bridge(ethereum=Ethereum(w3=w3, private_key=KEY))
    return bridge.eth, w3


def test_native_eth_dispatch_is_one_transaction_with_value():
    eth, w3 = setup(ETH_ROUTER, quotes={ETH_ROUTER: [(ZERO_ADDRESS, 69_000_000_000_101)]})
    result = eth.transfer_remote("eth", ALEO, amount_atomic=100).send(poll_seconds=0.001)
    assert isinstance(result, DispatchReceipt)
    assert len(w3.provider.sent) == 1 and w3.provider.sent[0]["value"] == 0x3EC1507D5065
    assert w3.provider.sent[0]["to"] == Web3.to_checksum_address(ETH_ROUTER)
    assert w3.provider.sent[0]["data"][2:10] == TRANSFER_REMOTE
    receipt = result.receipt
    assert receipt.status == Status.DELIVERY_PENDING and receipt.protocol == "hyperlane"
    assert result.message_id == Web3.to_hex(MESSAGE_ID) and receipt.id == result.message_id
    assert receipt.source_tx_id == w3.provider.hash_at(1) == result.transaction_id and result.amount_atomic == 100
    assert receipt.protocol_state == {
        "routeId": "hyperlane:ethereum/eth->aleo/eth", "approvalTxIds": [], "sourceSender": ACCT.address,
        "recipientBytes32": ALEO_BYTES32, "destinationDomain": 1634493807,
        "nativeValueAtomic": "69000000000101", "amountAtomic": "100", "messageId": Web3.to_hex(MESSAGE_ID),
        "sourceNonce": "0",
    }


def test_every_broadcast_records_the_nonce_it_went_out_at():
    """Recovery can only tell a pending transaction from a replaced one if the nonce is saved."""
    eth, w3 = setup(WBTC_ROUTER, quotes={WBTC_ROUTER: [(ZERO_ADDRESS, 50_000), (WBTC, 100_000)]})
    w3.provider.nonce_latest = w3.provider.nonce_pending = 83
    seen = []
    result = eth.transfer_remote("ethereum/wbtc", ALEO, amount="0.001").send(poll_seconds=0.001,
                                                                            on_checkpoint=seen.append)
    assert [t["nonce"] for t in w3.provider.sent] == [83, 84]          # the approval, then the dispatch
    assert result.receipt.protocol_state["sourceNonce"] == "84"
    assert [cp.source.get("sourceNonce") for cp in seen] == ["83", "84", "84"]


def test_wbtc_approves_exact_token_amount_then_dispatches_with_fee_value():
    eth, w3 = setup(WBTC_ROUTER, quotes={WBTC_ROUTER: [(ZERO_ADDRESS, 50_000), (WBTC, 100_000)]})
    result = eth.transfer_remote("ethereum/wbtc", ALEO, amount="0.001").send(poll_seconds=0.001)
    sent = w3.provider.sent
    assert [t["to"] for t in sent] == [Web3.to_checksum_address(WBTC), Web3.to_checksum_address(WBTC_ROUTER)]
    assert sent[0]["data"][2:].lower() == APPROVE + WBTC_ROUTER[2:].lower().rjust(64, "0") + format(100_000, "064x")
    assert sent[0]["value"] == 0 and sent[1]["value"] == 0xC350
    assert result.receipt.protocol_state["approvalTxIds"] == [w3.provider.hash_at(1)] and result.receipt.source_tx_id == w3.provider.hash_at(2)


def test_wbtc_sufficient_allowance_skips_approval():
    eth, w3 = setup(WBTC_ROUTER, quotes={WBTC_ROUTER: [(ZERO_ADDRESS, 50_000), (WBTC, 100_000)]},
                    allowances={(WBTC, ACCT.address, WBTC_ROUTER): 100_000})
    result = eth.transfer_remote("wbtc", ALEO, amount_atomic=100_000).send(poll_seconds=0.001)
    assert len(w3.provider.sent) == 1 and result.receipt.protocol_state["approvalTxIds"] == []


def test_usdt_resets_non_zero_allowance_first():
    eth, w3 = setup(USDT_ROUTER, quotes={USDT_ROUTER: [(ZERO_ADDRESS, 50_000), (USDT, 1_000_000)]},
                    allowances={(USDT, ACCT.address, USDT_ROUTER): 1})
    result = eth.transfer_remote("usdt", ALEO, amount="1").send(poll_seconds=0.001)
    sent = w3.provider.sent
    assert len(sent) == 3
    assert sent[0]["data"][2:].lower() == APPROVE + USDT_ROUTER[2:].lower().rjust(64, "0") + "0" * 64
    assert sent[1]["data"][2:].lower() == APPROVE + USDT_ROUTER[2:].lower().rjust(64, "0") + format(1_000_000, "064x")
    assert sent[2]["data"][2:10] == TRANSFER_REMOTE
    assert result.receipt.protocol_state["approvalTxIds"] == [w3.provider.hash_at(1), w3.provider.hash_at(2)]


def test_usdt_zero_allowance_needs_no_reset():
    eth, w3 = setup(USDT_ROUTER, quotes={USDT_ROUTER: [(ZERO_ADDRESS, 50_000), (USDT, 1_000_000)]})
    eth.transfer_remote("usdt", ALEO, amount="1").send(poll_seconds=0.001)
    assert len(w3.provider.sent) == 2


def test_approval_timeout_is_pending_and_checkpointed_before_polling():
    eth, w3 = setup(WBTC_ROUTER, quotes={WBTC_ROUTER: [(ZERO_ADDRESS, 50_000), (WBTC, 100_000)]})
    w3.provider.pending_nth.add(1)
    seen = []
    result = eth.transfer_remote("wbtc", ALEO, amount_atomic=100_000).send(
        timeout_seconds=0.01, poll_seconds=0.001, on_checkpoint=seen.append)
    assert result.receipt.status == Status.SOURCE_APPROVAL_PENDING and result.receipt.source_tx_id is None
    assert result.receipt.id == w3.provider.hash_at(1) and result.message_id is None and len(w3.provider.sent) == 1
    assert [cp.source for cp in seen] == [{"approvalTransactionIds": [w3.provider.hash_at(1)], "sourceNonce": "0"}]
    assert seen[0].intent == {"source": {"chain": "ethereum", "asset": "wbtc"}, "destination": {"chain": "aleo", "asset": "wbtc"},
                              "bridgeProtocol": "hyperlane", "amount": "0.001", "recipient": ALEO,
                              "sender": ACCT.address, "mintMode": "public"}


def test_dispatch_timeout_is_source_confirming_with_hash():
    eth, w3 = setup(ETH_ROUTER, quotes={ETH_ROUTER: [(ZERO_ADDRESS, 1_000)]})
    w3.provider.pending_nth.add(1)
    seen = []
    result = eth.transfer_remote("eth", ALEO, amount_atomic=100).send(
        timeout_seconds=0.01, poll_seconds=0.001, on_checkpoint=seen.append)
    assert result.receipt.status == Status.SOURCE_CONFIRMING and result.receipt.source_tx_id == w3.provider.hash_at(1)
    assert result.receipt.id == w3.provider.hash_at(1) and "messageId" not in result.receipt.protocol_state
    assert [cp.source for cp in seen] == [{"transactionId": w3.provider.hash_at(1), "sourceNonce": "0"}]


def test_dispatch_id_survives_unrelated_log_before_it():
    """A log from an unrelated event (different address/topic, e.g. an ERC-20 Transfer) preceding the
    Mailbox DispatchId log in the receipt must not prevent the message id from being decoded."""
    noise_topic = "0x" + keccak(text="Transfer(address,address,uint256)").hex()

    def logs_with_noise_first(tx):
        noise = event_log(WBTC, [noise_topic], "0x", log_index=1, tx_hash=tx["hash"])
        dispatch = dispatch_id_log(MAILBOX, MESSAGE_ID, tx_hash=tx["hash"], log_index=2)
        return [noise, dispatch] if tx["to"] == Web3.to_checksum_address(ETH_ROUTER) else []

    eth, w3 = setup(ETH_ROUTER, quotes={ETH_ROUTER: [(ZERO_ADDRESS, 1_000)]})
    w3.provider.receipt_logs = logs_with_noise_first
    result = eth.transfer_remote("eth", ALEO, amount_atomic=100).send(poll_seconds=0.001)
    assert result.message_id == Web3.to_hex(MESSAGE_ID)


def test_first_dispatch_id_wins_when_receipt_has_two():
    """veil's ``messageIdFromReceipt`` returns the FIRST matching DispatchId event, not the last."""
    first_id, second_id = bytes.fromhex("11" * 32), bytes.fromhex("22" * 32)

    def two_dispatch_logs(tx):
        return [dispatch_id_log(MAILBOX, first_id, tx_hash=tx["hash"], log_index=1),
                dispatch_id_log(MAILBOX, second_id, tx_hash=tx["hash"], log_index=2)]

    eth, w3 = setup(ETH_ROUTER, quotes={ETH_ROUTER: [(ZERO_ADDRESS, 1_000)]})
    w3.provider.receipt_logs = two_dispatch_logs
    result = eth.transfer_remote("eth", ALEO, amount_atomic=100).send(poll_seconds=0.001)
    assert result.message_id == Web3.to_hex(first_id)


def test_dispatch_id_from_a_foreign_address_is_ignored():
    """``process_receipt`` decodes by topic only — a ``DispatchId`` emitted by anything other than the
    route's Mailbox is another protocol's event and must read as "no message id"."""
    eth, w3 = setup(ETH_ROUTER, quotes={ETH_ROUTER: [(ZERO_ADDRESS, 1_000)]})
    w3.provider.receipt_logs = lambda tx: [dispatch_id_log(WBTC, MESSAGE_ID, tx_hash=tx["hash"], log_index=1)]
    result = eth.transfer_remote("eth", ALEO, amount_atomic=100).send(poll_seconds=0.001)
    assert result.message_id is None and result.receipt.id == w3.provider.hash_at(1)
    assert result.receipt.status == Status.DELIVERY_PENDING and "messageId" not in result.receipt.protocol_state


def test_mailbox_dispatch_id_wins_over_an_earlier_foreign_one():
    """First match *among the Mailbox's own* events, not first match overall."""
    foreign, mine = bytes.fromhex("11" * 32), bytes.fromhex("22" * 32)

    def logs(tx):
        return [dispatch_id_log(WBTC, foreign, tx_hash=tx["hash"], log_index=1),
                dispatch_id_log(MAILBOX, mine, tx_hash=tx["hash"], log_index=2)]

    eth, w3 = setup(ETH_ROUTER, quotes={ETH_ROUTER: [(ZERO_ADDRESS, 1_000)]})
    w3.provider.receipt_logs = logs
    result = eth.transfer_remote("eth", ALEO, amount_atomic=100).send(poll_seconds=0.001)
    assert result.message_id == Web3.to_hex(mine)


def test_missing_dispatch_id_log_keeps_tx_hash_as_id():
    eth, w3 = setup(ETH_ROUTER, with_dispatch_log=False, quotes={ETH_ROUTER: [(ZERO_ADDRESS, 1_000)]})
    result = eth.transfer_remote("eth", ALEO, amount_atomic=100).send(poll_seconds=0.001)
    assert result.receipt.status == Status.DELIVERY_PENDING and result.message_id is None
    assert result.receipt.id == w3.provider.hash_at(1) and "messageId" not in result.receipt.protocol_state


def test_build_lists_approval_then_dispatch_without_sending():
    eth, w3 = setup(WBTC_ROUTER, quotes={WBTC_ROUTER: [(ZERO_ADDRESS, 50_000), (WBTC, 100_000)]})
    txs = eth.transfer_remote("wbtc", ALEO, amount_atomic=100_000).build()
    assert [t["to"] for t in txs] == [Web3.to_checksum_address(WBTC), Web3.to_checksum_address(WBTC_ROUTER)]
    assert [t["value"] for t in txs] == [0, 50_000] and w3.provider.sent == []


def test_plan_driven_transfer_is_identical_to_the_asset_driven_one():
    eth, w3 = setup(WBTC_ROUTER, quotes={WBTC_ROUTER: [(ZERO_ADDRESS, 50_000), (WBTC, 100_000)]})
    quote = eth.quote_transfer_remote("wbtc", ALEO, amount_atomic=100_000)
    by_asset = eth.transfer_remote("wbtc", ALEO, amount_atomic=100_000).build()
    by_plan = eth.transfer_remote(plan=quote.plan).build()
    assert by_plan == by_asset and len(by_plan) == 2
    assert eth.quote_transfer_remote(plan=quote.plan) == quote          # the quote round-trips through its own plan
    assert w3.provider.sent == []


def test_plan_driven_transfer_rejects_a_tampered_stale_or_foreign_plan():
    """Nothing in a caller-supplied plan is trusted, and every rejection happens before any RPC."""
    eth, w3 = setup(WBTC_ROUTER, quotes={WBTC_ROUTER: [(ZERO_ADDRESS, 50_000), (WBTC, 100_000)]})
    plan = eth.quote_transfer_remote("wbtc", ALEO, amount_atomic=100_000).plan
    w3.provider.methods.clear()
    with pytest.raises(BridgeError, match="plan does not match the requested transfer: amount"):
        eth.transfer_remote(plan=dataclasses.replace(plan, amount_atomic=99_999))
    with pytest.raises(BridgeError, match="plan does not match the requested transfer: recipient"):
        eth.transfer_remote(recipient=OTHER_ALEO, plan=plan)             # explicit argument vs the plan's own value
    with pytest.raises(RegistryVersionMismatchError):
        eth.transfer_remote(plan=dataclasses.replace(plan, registry_version="0000-00-00.stale"))
    with pytest.raises(RouteUnavailableError, match="not a hyperlane one"):
        eth.transfer_remote(plan=dataclasses.replace(plan, route_id="xreserve:ethereum/usdc->aleo/usdcx"))
    with pytest.raises(RouteUnavailableError):
        eth.transfer_remote(plan=dataclasses.replace(plan, route_id="hyperlane:nowhere/nothing->aleo/eth"))
    with pytest.raises(ConfigurationError, match="does not match connected account"):
        eth.transfer_remote(plan=dataclasses.replace(plan, sender=OTHER))
    with pytest.raises(BridgeError, match="checksummed EVM address"):
        eth.transfer_remote(plan=dataclasses.replace(plan, sender=None))
    with pytest.raises(ValueError, match="not both"):
        eth.transfer_remote("wbtc", plan=plan)
    with pytest.raises(ValueError, match="not both"):
        eth.quote_transfer_remote(plan=plan, sender=ACCT.address)
    assert w3.provider.methods == [] and w3.provider.sent == []


def test_read_only_connection_cannot_transfer():
    w3 = fake_web3(quotes={ETH_ROUTER: [(ZERO_ADDRESS, 1_000)]})
    eth = make_bridge(ethereum=Ethereum(w3=w3)).eth
    with pytest.raises(ConfigurationError, match="read-only"):
        eth.transfer_remote("eth", ALEO, amount_atomic=100)


def test_transfer_remote_without_a_plan_or_a_recipient_names_the_missing_recipient():
    """Same guard as the quote path: a missing recipient is an InvalidRecipientError, not a TypeError."""
    eth, w3 = setup(ETH_ROUTER, quotes={ETH_ROUTER: [(ZERO_ADDRESS, 1_000)]})
    with pytest.raises(InvalidRecipientError, match="recipient is required when no plan is given"):
        eth.transfer_remote("eth", amount_atomic=100)
    assert w3.provider.methods == [] and w3.provider.sent == []
