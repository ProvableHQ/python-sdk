import pytest
from eth_account import Account
from eth_utils import keccak
from web3 import Web3

from aleo_bridge.errors import ConfigurationError
from aleo_bridge.eth import Ethereum
from aleo_bridge.types import DispatchReceipt, Status
from tests.fakes.fake_web3 import ZERO_ADDRESS, dispatch_id_log, fake_web3, make_bridge, tx_hash_for

KEY = "0x" + "11" * 32
ACCT = Account.from_key(KEY)
ALEO = "aleo1kypwp5m7qtk9mwazgcpg0tq8aal23mnrvwfvug65qgcg9xvsrqgspyjm6n"
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
    assert receipt.source_tx_id == tx_hash_for(1) == result.transaction_id and result.amount_atomic == 100
    assert receipt.protocol_state == {
        "routeId": "hyperlane:ethereum/eth->aleo/eth", "approvalTxIds": [], "sourceSender": ACCT.address,
        "recipientBytes32": ALEO_BYTES32, "destinationDomain": 1634493807,
        "nativeValueAtomic": "69000000000101", "amountAtomic": "100", "messageId": Web3.to_hex(MESSAGE_ID),
    }


def test_wbtc_approves_exact_token_amount_then_dispatches_with_fee_value():
    eth, w3 = setup(WBTC_ROUTER, quotes={WBTC_ROUTER: [(ZERO_ADDRESS, 50_000), (WBTC, 100_000)]})
    result = eth.transfer_remote("ethereum/wbtc", ALEO, amount="0.001").send(poll_seconds=0.001)
    sent = w3.provider.sent
    assert [t["to"] for t in sent] == [Web3.to_checksum_address(WBTC), Web3.to_checksum_address(WBTC_ROUTER)]
    assert sent[0]["data"][2:].lower() == APPROVE + WBTC_ROUTER[2:].lower().rjust(64, "0") + format(100_000, "064x")
    assert sent[0]["value"] == 0 and sent[1]["value"] == 0xC350
    assert result.receipt.protocol_state["approvalTxIds"] == [tx_hash_for(1)] and result.receipt.source_tx_id == tx_hash_for(2)


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
    assert result.receipt.protocol_state["approvalTxIds"] == [tx_hash_for(1), tx_hash_for(2)]


def test_usdt_zero_allowance_needs_no_reset():
    eth, w3 = setup(USDT_ROUTER, quotes={USDT_ROUTER: [(ZERO_ADDRESS, 50_000), (USDT, 1_000_000)]})
    eth.transfer_remote("usdt", ALEO, amount="1").send(poll_seconds=0.001)
    assert len(w3.provider.sent) == 2


def test_approval_timeout_is_pending_and_checkpointed_before_polling():
    eth, w3 = setup(WBTC_ROUTER, quotes={WBTC_ROUTER: [(ZERO_ADDRESS, 50_000), (WBTC, 100_000)]})
    w3.provider.pending.add(tx_hash_for(1))
    seen = []
    result = eth.transfer_remote("wbtc", ALEO, amount_atomic=100_000).send(
        timeout_seconds=0.01, poll_seconds=0.001, on_checkpoint=seen.append)
    assert result.receipt.status == Status.SOURCE_APPROVAL_PENDING and result.receipt.source_tx_id is None
    assert result.receipt.id == tx_hash_for(1) and result.message_id is None and len(w3.provider.sent) == 1
    assert [cp.source for cp in seen] == [{"approvalTransactionIds": [tx_hash_for(1)]}]
    assert seen[0].intent == {"source": {"chain": "ethereum", "asset": "wbtc"}, "destination": {"chain": "aleo", "asset": "wbtc"},
                              "bridgeProtocol": "hyperlane", "amount": "0.001", "recipient": ALEO,
                              "sender": ACCT.address, "mintMode": "public"}


def test_dispatch_timeout_is_source_confirming_with_hash():
    eth, w3 = setup(ETH_ROUTER, quotes={ETH_ROUTER: [(ZERO_ADDRESS, 1_000)]})
    w3.provider.pending.add(tx_hash_for(1))
    seen = []
    result = eth.transfer_remote("eth", ALEO, amount_atomic=100).send(
        timeout_seconds=0.01, poll_seconds=0.001, on_checkpoint=seen.append)
    assert result.receipt.status == Status.SOURCE_CONFIRMING and result.receipt.source_tx_id == tx_hash_for(1)
    assert result.receipt.id == tx_hash_for(1) and "messageId" not in result.receipt.protocol_state
    assert [cp.source for cp in seen] == [{"transactionId": tx_hash_for(1)}]


def test_missing_dispatch_id_log_keeps_tx_hash_as_id():
    eth, _ = setup(ETH_ROUTER, with_dispatch_log=False, quotes={ETH_ROUTER: [(ZERO_ADDRESS, 1_000)]})
    result = eth.transfer_remote("eth", ALEO, amount_atomic=100).send(poll_seconds=0.001)
    assert result.receipt.status == Status.DELIVERY_PENDING and result.message_id is None
    assert result.receipt.id == tx_hash_for(1) and "messageId" not in result.receipt.protocol_state


def test_build_lists_approval_then_dispatch_without_sending():
    eth, w3 = setup(WBTC_ROUTER, quotes={WBTC_ROUTER: [(ZERO_ADDRESS, 50_000), (WBTC, 100_000)]})
    txs = eth.transfer_remote("wbtc", ALEO, amount_atomic=100_000).build()
    assert [t["to"] for t in txs] == [Web3.to_checksum_address(WBTC), Web3.to_checksum_address(WBTC_ROUTER)]
    assert [t["value"] for t in txs] == [0, 50_000] and w3.provider.sent == []


def test_read_only_connection_cannot_transfer():
    w3 = fake_web3(quotes={ETH_ROUTER: [(ZERO_ADDRESS, 1_000)]})
    eth = make_bridge(ethereum=Ethereum(w3=w3)).eth
    with pytest.raises(ConfigurationError, match="read-only"):
        eth.transfer_remote("eth", ALEO, amount_atomic=100)
