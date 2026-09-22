import pytest
from eth_account import Account
from web3 import Web3

from aleo_bridge import encoding
from aleo_bridge.errors import (BridgeError, ChainMismatchError, CheckpointInvalidError, ConfigurationError,
                                UnsupportedRouteError)
from aleo_bridge.eth import Ethereum, _plan_for
from aleo_bridge.registry import DEFAULT_REGISTRY
from aleo_bridge.types import Receipt, Status
from tests.fakes.fake_web3 import deposited_log, dispatch_id_log, fake_web3, make_bridge

KEY = "0x" + "11" * 32
ACCT = Account.from_key(KEY)
ALEO = "aleo1kypwp5m7qtk9mwazgcpg0tq8aal23mnrvwfvug65qgcg9xvsrqgspyjm6n"
ALEO32 = encoding.aleo_address_to_bytes32(ALEO)
WBTC, WBTC_ROUTER = "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599", "0x20CDC85778b732073F7EecEF3DF25c0d310f8772"
MAILBOX = "0xc005dc82818d67AF737725bD4bf75435d065D239"
SEPOLIA_USDC, SEPOLIA_XRESERVE = "0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238", "0x008888878f94C0d87defdf0B07f46B93C1934442"
REMOTE_TOKEN = bytes.fromhex("b143ed52c774cd1d4a519d0e796f15916be5a9e1d45edcd9852dd23f68f53401")
H1, H2 = "0x" + "11" * 32, "0x" + "22" * 32
MESSAGE_ID = bytes.fromhex("ab" * 32)
DELIVERED_ID = "0xc7c2c763ef846ff1583d9222d8ecbfc56da2e0cdcc9a63bc4bde51467644794d"
WBTC_ROUTE = DEFAULT_REGISTRY.route("hyperlane:ethereum/wbtc->aleo/wbtc")
USDC_ROUTE = DEFAULT_REGISTRY.route("xreserve:sepolia/usdc->aleo-testnet/usdcx")


def hyperlane_state(**overrides):
    state = {"routeId": WBTC_ROUTE.id, "approvalTxIds": [H1], "sourceSender": ACCT.address,
             "recipientBytes32": "0x" + ALEO32.hex(), "destinationDomain": 1634493807,
             "nativeValueAtomic": "50000", "amountAtomic": "100000"}
    state.update(overrides)
    return state


def xreserve_state(**overrides):
    state = {"routeId": USDC_ROUTE.id, "approvalTxIds": [H1], "sourceSender": ACCT.address, "mintMode": "public",
             "intendedRecipient": ALEO, "xReserveContract": SEPOLIA_XRESERVE, "tokenAddress": SEPOLIA_USDC,
             "sourceChainId": 11155111, "remoteDomain": 10002, "remoteRecipientBytes32": "0x" + ALEO32.hex(),
             "hookData": "0x" + "00" * 65, "amountAtomic": "2000000", "maxFeeAtomic": "100000"}
    state.update(overrides)
    return state


def mainnet(*, signed=True, **config):
    w3 = fake_web3(**config)
    conn = Ethereum(w3=w3, private_key=KEY) if signed else Ethereum(w3=w3)
    return make_bridge(ethereum=conn).eth, w3


WBTC_PLAN = _plan_for(DEFAULT_REGISTRY, WBTC_ROUTE, amount_atomic=100_000, recipient=ALEO, sender=ACCT.address)
USDC_PLAN = _plan_for(DEFAULT_REGISTRY, USDC_ROUTE, amount_atomic=2_000_000, recipient=ALEO, sender=ACCT.address)


def test_approval_pending_branch():
    eth, w3 = mainnet()
    receipt = Receipt(id=H1, protocol="hyperlane", status=Status.SOURCE_APPROVAL_PENDING, protocol_state=hyperlane_state())
    assert eth.source_status(WBTC_PLAN, receipt) is receipt                      # no receipt yet → unchanged
    w3.provider.add_receipt(H1)
    advanced = eth.source_status(WBTC_PLAN, receipt)
    assert advanced.status == Status.SOURCE_SUBMISSION_PENDING and advanced.id == H1
    w3.provider.add_receipt(H1, status=0)
    failed = eth.source_status(WBTC_PLAN, receipt)
    assert failed.status == Status.FAILED and failed.protocol_state["sourceError"] == f"EVM approval transaction reverted: {H1}"
    assert "eth_sendRawTransaction" not in w3.provider.methods


def test_hyperlane_source_confirming_branch():
    eth, w3 = mainnet()
    receipt = Receipt(id=H2, protocol="hyperlane", status=Status.SOURCE_CONFIRMING, source_tx_id=H2, protocol_state=hyperlane_state())
    assert eth.source_status(WBTC_PLAN, receipt) is receipt
    w3.provider.add_receipt(H2, logs=[dispatch_id_log(MAILBOX, MESSAGE_ID, tx_hash=H2)], sender=ACCT.address, to=WBTC_ROUTER)
    advanced = eth.source_status(WBTC_PLAN, receipt)
    assert advanced.status == Status.DELIVERY_PENDING and advanced.id == Web3.to_hex(MESSAGE_ID)
    assert advanced.source_tx_id == H2 and advanced.protocol_state["messageId"] == Web3.to_hex(MESSAGE_ID)
    w3.provider.add_receipt(H2)                                                  # confirmed, no DispatchId log
    no_id = eth.source_status(WBTC_PLAN, receipt)
    assert no_id.status == Status.DELIVERY_PENDING and no_id.id == H2 and "messageId" not in no_id.protocol_state
    w3.provider.add_receipt(H2, status=0)
    failed = eth.source_status(WBTC_PLAN, receipt)
    assert failed.status == Status.FAILED and failed.protocol_state["sourceError"] == f"EVM transaction reverted: {H2}"


DROPPED_MESSAGE = ("transaction {h} (nonce 83) was dropped or replaced before it mined; no funds moved by it "
                   "— recover() then resume() re-dispatches")


def test_a_dropped_dispatch_is_expired_not_confirming_forever():
    """Mainnet 2026-09-22: the dispatch was replaced at its own nonce, so it can never mine — but
    ``source_status`` kept answering SOURCE_CONFIRMING (next == "wait") indefinitely."""
    eth, w3 = mainnet()
    receipt = Receipt(id=H2, protocol="hyperlane", status=Status.SOURCE_CONFIRMING, source_tx_id=H2,
                      protocol_state=hyperlane_state(sourceNonce="83"))
    w3.provider.nonce_latest = 83                                    # still the account's next nonce: merely pending
    w3.provider.add_transaction(H2, sender=ACCT.address, to=WBTC_ROUTER)
    assert eth.source_status(WBTC_PLAN, receipt) is receipt
    w3.provider.tx_not_found.add(H2)                                 # gone from every mempool, and the nonce moved on
    assert eth.source_status(WBTC_PLAN, receipt) is receipt          # nonce 83 is still unused: nothing replaced it
    w3.provider.nonce_latest = 84
    expired = eth.source_status(WBTC_PLAN, receipt)
    assert expired.status == Status.EXPIRED and expired.protocol_state["dropped"] is True
    assert expired.protocol_state["sourceError"] == DROPPED_MESSAGE.format(h=H2)
    assert expired.next_action is None and w3.provider.sent == []


def test_a_dropped_xreserve_deposit_is_expired_too():
    w3 = fake_web3(chain_id=11155111)
    eth = make_bridge(environment="testnet", ethereum=Ethereum(w3=w3)).eth
    receipt = Receipt(id=H2, protocol="xreserve", status=Status.SOURCE_CONFIRMING, source_tx_id=H2,
                      protocol_state=xreserve_state(sourceNonce="83"))
    w3.provider.tx_not_found.add(H2)
    w3.provider.nonce_latest = 84
    expired = eth.source_status(USDC_PLAN, receipt)
    assert expired.status == Status.EXPIRED and expired.protocol_state["dropped"] is True
    assert expired.protocol_state["sourceError"] == DROPPED_MESSAGE.format(h=H2)


def test_a_receipt_without_a_source_nonce_still_waits():
    """Checkpoints written before the nonce was recorded must not become EXPIRED on a guess."""
    eth, w3 = mainnet()
    receipt = Receipt(id=H2, protocol="hyperlane", status=Status.SOURCE_CONFIRMING, source_tx_id=H2,
                      protocol_state=hyperlane_state())
    w3.provider.tx_not_found.add(H2)
    w3.provider.nonce_latest = 84
    assert eth.source_status(WBTC_PLAN, receipt) is receipt
    assert "eth_getTransactionByHash" not in w3.provider.methods     # no nonce, no reason to ask


def test_hyperlane_state_must_match_plan():
    eth, _ = mainnet()
    bad = Receipt(id=H2, protocol="hyperlane", status=Status.SOURCE_CONFIRMING, source_tx_id=H2,
                  protocol_state=hyperlane_state(amountAtomic="1"))
    with pytest.raises(CheckpointInvalidError, match="does not match the prepared transfer"):
        eth.source_status(WBTC_PLAN, bad)
    wrong_route = Receipt(id=H2, protocol="hyperlane", status=Status.SOURCE_CONFIRMING, source_tx_id=H2,
                          protocol_state=hyperlane_state(routeId="hyperlane:ethereum/eth->aleo/eth"))
    with pytest.raises(CheckpointInvalidError, match="does not match the prepared route"):
        eth.source_status(WBTC_PLAN, wrong_route)
    missing_hash = Receipt(id=H2, protocol="hyperlane", status=Status.SOURCE_CONFIRMING, protocol_state=hyperlane_state())
    with pytest.raises(CheckpointInvalidError, match="source transaction id"):
        eth.source_status(WBTC_PLAN, missing_hash)
    other = Receipt(id=H2, protocol="hyperlane", status=Status.DELIVERY_PENDING, source_tx_id=H2, protocol_state=hyperlane_state())
    with pytest.raises(BridgeError, match="SOURCE_APPROVAL_PENDING and SOURCE_CONFIRMING"):
        eth.source_status(WBTC_PLAN, other)


def test_xreserve_source_confirming_branch_works_read_only():
    w3 = fake_web3(chain_id=11155111)
    eth = make_bridge(environment="testnet", ethereum=Ethereum(w3=w3)).eth          # no signer: uses sourceSender
    receipt = Receipt(id=H2, protocol="xreserve", status=Status.SOURCE_CONFIRMING, source_tx_id=H2, protocol_state=xreserve_state())
    assert eth.source_status(USDC_PLAN, receipt) is receipt
    log = deposited_log(SEPOLIA_XRESERVE, local_token=SEPOLIA_USDC, depositor=ACCT.address, remote_recipient32=ALEO32,
                        value=2_000_000, remote_domain=10002, remote_token32=REMOTE_TOKEN, max_fee=100_000,
                        hook_data=bytes(65), tx_hash=H2, log_index=3)
    w3.provider.add_receipt(H2, logs=[log], sender=ACCT.address, to=SEPOLIA_XRESERVE)
    advanced = eth.source_status(USDC_PLAN, receipt)
    nonce = encoding.xreserve_deposit_nonce(0, bytes.fromhex(H2[2:]), 3)
    assert advanced.status == Status.ATTESTATION_PENDING and advanced.protocol_state["nonce"] == "0x" + nonce.hex()
    assert advanced.id == advanced.protocol_state["messageHash"] and advanced.protocol_state["depositLogIndex"] == 3
    assert advanced.protocol_state["approvalTxIds"] == [H1] and advanced.source_tx_id == H2
    w3.provider.add_receipt(H2, status=0)
    assert eth.source_status(USDC_PLAN, receipt).status == Status.FAILED


def test_xreserve_state_validation():
    w3 = fake_web3(chain_id=11155111)
    eth = make_bridge(environment="testnet", ethereum=Ethereum(w3=w3, private_key=KEY)).eth
    for bad in (xreserve_state(mintMode="private"), xreserve_state(intendedRecipient="aleo1" + "q" * 58),
                xreserve_state(hookData="0x00"), xreserve_state(amountAtomic="abc")):
        receipt = Receipt(id=H2, protocol="xreserve", status=Status.SOURCE_CONFIRMING, source_tx_id=H2, protocol_state=bad)
        with pytest.raises(CheckpointInvalidError):
            eth.source_status(USDC_PLAN, receipt)
    no_owner = Receipt(id=H2, protocol="xreserve", status=Status.SOURCE_CONFIRMING, source_tx_id=H2,
                       protocol_state=xreserve_state(sourceSender=None))
    plan_without_sender = _plan_for(DEFAULT_REGISTRY, USDC_ROUTE, amount_atomic=2_000_000, recipient=ALEO, sender=None)
    read_only = make_bridge(environment="testnet", ethereum=Ethereum(w3=fake_web3(chain_id=11155111))).eth
    with pytest.raises(ConfigurationError, match="prepared sender"):
        read_only.source_status(plan_without_sender, no_owner)


def test_xreserve_source_confirming_asserts_the_chain_before_reading():
    """Mirrors the Hyperlane branch: a connection pointed at the wrong network must not read
    receipts or logs from it."""
    w3 = fake_web3()                                                                # chain 1, route wants 11155111
    eth = make_bridge(environment="testnet", ethereum=Ethereum(w3=w3)).eth
    receipt = Receipt(id=H2, protocol="xreserve", status=Status.SOURCE_CONFIRMING, source_tx_id=H2,
                      protocol_state=xreserve_state())
    with pytest.raises(ChainMismatchError):
        eth.source_status(USDC_PLAN, receipt)
    assert "eth_getTransactionReceipt" not in w3.provider.methods and "eth_getLogs" not in w3.provider.methods


def test_is_delivered_reads_mailbox():
    eth, w3 = mainnet(delivered={DELIVERED_ID})
    assert eth.is_delivered(DELIVERED_ID) is True
    assert eth.is_delivered(bytes.fromhex(DELIVERED_ID[2:])) is True
    assert eth.is_delivered("0x" + "00" * 32) is False
    assert w3.provider.methods.count("eth_call") == 3
    with pytest.raises(BridgeError, match="32-byte message id"):
        eth.is_delivered("0x1234")
    testnet = make_bridge(environment="testnet", ethereum=Ethereum(w3=fake_web3(chain_id=11155111))).eth
    with pytest.raises(UnsupportedRouteError, match="Mailbox"):
        testnet.is_delivered(DELIVERED_ID)


def test_balance_native_and_erc20():
    eth, _ = mainnet(eth_balances={ACCT.address: 5}, token_balances={(WBTC, ACCT.address): 7})
    assert eth.balance("eth") == 5 and eth.balance("ethereum/wbtc") == 7 and eth.balance("usdt") == 0
    read_only, _ = mainnet(signed=False, eth_balances={ACCT.address: 5})
    with pytest.raises(ConfigurationError, match="address"):
        read_only.balance("eth")
    assert read_only.balance("eth", address=ACCT.address) == 5
