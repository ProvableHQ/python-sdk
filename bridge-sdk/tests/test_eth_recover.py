import dataclasses

import pytest
from eth_account import Account
from web3 import Web3

from aleo_bridge import encoding
from aleo_bridge.checkpoint import create_checkpoint
from aleo_bridge.errors import BridgeError, CheckpointInvalidError, RegistryVersionMismatchError
from aleo_bridge.eth import Ethereum, _plan_for
from aleo_bridge.registry import DEFAULT_REGISTRY
from aleo_bridge.types import Receipt, Status
from tests.fakes.fake_web3 import deposited_log, dispatch_id_log, fake_web3, make_bridge, sent_transfer_remote_log

KEY = "0x" + "11" * 32
ACCT = Account.from_key(KEY)
OTHER = "0x0000000000000000000000000000000000000009"
ALEO = "aleo1kypwp5m7qtk9mwazgcpg0tq8aal23mnrvwfvug65qgcg9xvsrqgspyjm6n"
ALEO32 = encoding.aleo_address_to_bytes32(ALEO)
WBTC_ROUTER = "0x20CDC85778b732073F7EecEF3DF25c0d310f8772"
MAILBOX = "0xc005dc82818d67AF737725bD4bf75435d065D239"
SEPOLIA_USDC, SEPOLIA_XRESERVE = "0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238", "0x008888878f94C0d87defdf0B07f46B93C1934442"
REMOTE_TOKEN = bytes.fromhex("b143ed52c774cd1d4a519d0e796f15916be5a9e1d45edcd9852dd23f68f53401")
APPROVAL, DISPATCH = "0x" + "11" * 32, "0x" + "22" * 32
RECOVERED, RECOVERED_2, EARLIER = "0x" + "44" * 32, "0x" + "55" * 32, "0x" + "66" * 32
RECOVERED_MESSAGE_ID = bytes.fromhex("cd" * 32)
WBTC_ROUTE = DEFAULT_REGISTRY.route("hyperlane:ethereum/wbtc->aleo/wbtc")
USDC_ROUTE = DEFAULT_REGISTRY.route("xreserve:sepolia/usdc->aleo-testnet/usdcx")
WBTC_PLAN = _plan_for(DEFAULT_REGISTRY, WBTC_ROUTE, amount_atomic=100_000, recipient=ALEO, sender=ACCT.address)


def hyperlane_checkpoint(plan=WBTC_PLAN, *, tx_id=None):
    state = {"routeId": WBTC_ROUTE.id, "approvalTxIds": [APPROVAL], "sourceSender": plan.sender,
             "recipientBytes32": "0x" + ALEO32.hex(), "destinationDomain": 1634493807, "nativeValueAtomic": "50000", "amountAtomic": "100000"}
    receipt = (Receipt(id=tx_id, protocol="hyperlane", status=Status.SOURCE_CONFIRMING, source_tx_id=tx_id, protocol_state=state)
               if tx_id else Receipt(id=APPROVAL, protocol="hyperlane", status=Status.SOURCE_APPROVAL_PENDING, protocol_state=state))
    return create_checkpoint(plan, receipt, DEFAULT_REGISTRY)


def xreserve_checkpoint(plan, hook: bytes, *, tx_id=None):
    state = {"routeId": USDC_ROUTE.id, "approvalTxIds": [APPROVAL], "sourceSender": ACCT.address, "mintMode": plan.mint_mode,
             "intendedRecipient": ALEO, "xReserveContract": SEPOLIA_XRESERVE, "tokenAddress": SEPOLIA_USDC, "sourceChainId": 11155111,
             "remoteDomain": 10002, "remoteRecipientBytes32": "0x" + ALEO32.hex(), "hookData": "0x" + hook.hex(),
             "amountAtomic": "2000000", "maxFeeAtomic": "100000"}
    receipt = (Receipt(id=tx_id, protocol="xreserve", status=Status.SOURCE_CONFIRMING, source_tx_id=tx_id, protocol_state=state)
               if tx_id else Receipt(id=APPROVAL, protocol="xreserve", status=Status.SOURCE_APPROVAL_PENDING, protocol_state=state))
    return create_checkpoint(plan, receipt, DEFAULT_REGISTRY)


def mainnet_read_only():
    w3 = fake_web3()
    return make_bridge(ethereum=Ethereum(w3=w3)).eth, w3


def dispatch_history(w3, tx_hash, *, sender=ACCT.address, block_number=0x65, amount=100_000):
    w3.provider.history_logs.append(sent_transfer_remote_log(WBTC_ROUTER, destination=1634493807, recipient32=ALEO32,
                                                             amount=amount, tx_hash=tx_hash, block_number=block_number))
    w3.provider.add_transaction(tx_hash, sender=sender, to=WBTC_ROUTER, block_number=block_number)
    w3.provider.add_receipt(tx_hash, logs=[dispatch_id_log(MAILBOX, RECOVERED_MESSAGE_ID, tx_hash=tx_hash)],
                            sender=sender, to=WBTC_ROUTER, block_number=block_number)


def test_checkpoint_must_match_plan_and_registry():
    eth, _ = mainnet_read_only()
    cp = hyperlane_checkpoint()
    with pytest.raises(CheckpointInvalidError, match="does not match the prepared route"):
        eth.recover_source(WBTC_PLAN, dataclasses.replace(cp, route={**cp.route, "id": "hyperlane:ethereum/eth->aleo/eth"}))
    with pytest.raises(RegistryVersionMismatchError):
        eth.recover_source(WBTC_PLAN, dataclasses.replace(cp, route={**cp.route, "registryVersion": "0000-00-00.stale"}))
    with pytest.raises(CheckpointInvalidError, match="no submitted transaction"):
        eth.recover_source(WBTC_PLAN, dataclasses.replace(cp, source=None))


def test_hyperlane_rejects_checkpoint_with_destination_leg():
    eth, _ = mainnet_read_only()
    cp = hyperlane_checkpoint()
    bad = dataclasses.replace(cp, destination={"transactionId": "0x" + "77" * 32})
    with pytest.raises(CheckpointInvalidError, match="destination"):
        eth.recover_source(WBTC_PLAN, bad)


def test_hyperlane_unmined_approval_stays_approval_pending():
    eth, w3 = mainnet_read_only()
    receipt = eth.recover_source(WBTC_PLAN, hyperlane_checkpoint())
    assert receipt.status == Status.SOURCE_APPROVAL_PENDING and receipt.id == APPROVAL and receipt.source_tx_id is None
    assert receipt.protocol_state["approvalTxIds"] == [APPROVAL] and receipt.protocol_state["sourceSender"] == ACCT.address
    assert "eth_getLogs" not in w3.provider.methods


def test_hyperlane_confirmed_approval_without_dispatch_stops_at_submission_boundary():
    eth, w3 = mainnet_read_only()
    w3.provider.add_receipt(APPROVAL, block_number=0x65)
    receipt = eth.recover_source(WBTC_PLAN, hyperlane_checkpoint())
    assert receipt.status == Status.SOURCE_SUBMISSION_PENDING and receipt.id == APPROVAL
    assert "eth_getLogs" in w3.provider.methods and w3.provider.sent == []


def test_hyperlane_scan_finds_the_dispatch_after_the_approval_block():
    eth, w3 = mainnet_read_only()
    w3.provider.add_receipt(APPROVAL, block_number=0x65)
    dispatch_history(w3, EARLIER, block_number=0x10)             # before the approval: must be ignored by fromBlock
    dispatch_history(w3, RECOVERED, block_number=0x66)
    receipt = eth.recover_source(WBTC_PLAN, hyperlane_checkpoint())
    assert receipt.status == Status.DELIVERY_PENDING and receipt.source_tx_id == RECOVERED
    assert receipt.id == Web3.to_hex(RECOVERED_MESSAGE_ID) == receipt.protocol_state["messageId"]
    assert receipt.protocol_state["approvalTxIds"] == [APPROVAL]


def test_hyperlane_scan_ignores_other_senders_and_amounts():
    eth, w3 = mainnet_read_only()
    w3.provider.add_receipt(APPROVAL, block_number=0x65)
    dispatch_history(w3, RECOVERED, sender=OTHER)
    dispatch_history(w3, RECOVERED_2, amount=99_999)
    assert eth.recover_source(WBTC_PLAN, hyperlane_checkpoint()).status == Status.SOURCE_SUBMISSION_PENDING


def test_hyperlane_multiple_matches_refuse_to_choose():
    eth, w3 = mainnet_read_only()
    w3.provider.add_receipt(APPROVAL, block_number=0x65)
    dispatch_history(w3, RECOVERED)
    dispatch_history(w3, RECOVERED_2)
    with pytest.raises(BridgeError, match="Multiple matching Hyperlane dispatches"):
        eth.recover_source(WBTC_PLAN, hyperlane_checkpoint())


def test_hyperlane_required_scan_needs_sender_and_confirmed_approval():
    eth, w3 = mainnet_read_only()
    with pytest.raises(BridgeError, match="no confirmed approval block"):
        eth.recover_source(WBTC_PLAN, hyperlane_checkpoint(), required=True)
    plan_no_sender = _plan_for(DEFAULT_REGISTRY, WBTC_ROUTE, amount_atomic=100_000, recipient=ALEO, sender=None)
    w3.provider.add_receipt(APPROVAL, block_number=0x65)
    with pytest.raises(BridgeError, match="without the source account"):
        eth.recover_source(plan_no_sender, hyperlane_checkpoint(plan_no_sender), required=True)
    assert eth.recover_source(plan_no_sender, hyperlane_checkpoint(plan_no_sender)).status == Status.SOURCE_SUBMISSION_PENDING


def test_hyperlane_required_scan_with_no_matching_dispatch_is_not_fatal():
    """A completed scan (known sender, confirmed approval) that matches nothing is a valid answer,
    not an inability to scan: required=True still returns SOURCE_SUBMISSION_PENDING, never sends."""
    eth, w3 = mainnet_read_only()
    w3.provider.add_receipt(APPROVAL, block_number=0x65)
    receipt = eth.recover_source(WBTC_PLAN, hyperlane_checkpoint(), required=True)
    assert receipt.status == Status.SOURCE_SUBMISSION_PENDING and receipt.id == APPROVAL
    assert "eth_getLogs" in w3.provider.methods and w3.provider.sent == []


def test_hyperlane_saved_dispatch_is_observed_not_resent():
    eth, w3 = mainnet_read_only()
    cp = hyperlane_checkpoint(tx_id=DISPATCH)
    pending = eth.recover_source(WBTC_PLAN, cp)
    assert pending.status == Status.SOURCE_CONFIRMING and pending.source_tx_id == DISPATCH
    w3.provider.add_receipt(DISPATCH, logs=[dispatch_id_log(MAILBOX, RECOVERED_MESSAGE_ID, tx_hash=DISPATCH)], sender=ACCT.address, to=WBTC_ROUTER)
    done = eth.recover_source(WBTC_PLAN, cp)
    assert done.status == Status.DELIVERY_PENDING and done.id == Web3.to_hex(RECOVERED_MESSAGE_ID) and w3.provider.sent == []


def test_xreserve_scan_recovers_a_confirmed_deposit_from_an_approval_only_checkpoint():
    w3 = fake_web3(chain_id=11155111)
    eth = make_bridge(environment="testnet", ethereum=Ethereum(w3=w3)).eth
    plan = _plan_for(DEFAULT_REGISTRY, USDC_ROUTE, amount_atomic=2_000_000, recipient=ALEO, sender=ACCT.address)
    hook = bytes(65)
    cp = xreserve_checkpoint(plan, hook)
    assert cp.source["hookData"] == "0x" + hook.hex()
    assert eth.recover_source(plan, cp).status == Status.SOURCE_APPROVAL_PENDING
    w3.provider.add_receipt(APPROVAL, block_number=0x65)
    assert eth.recover_source(plan, cp).status == Status.SOURCE_SUBMISSION_PENDING
    log = deposited_log(SEPOLIA_XRESERVE, local_token=SEPOLIA_USDC, depositor=ACCT.address, remote_recipient32=ALEO32, value=2_000_000,
                        remote_domain=10002, remote_token32=REMOTE_TOKEN, max_fee=100_000, hook_data=hook, tx_hash=RECOVERED, log_index=3)
    w3.provider.history_logs.append(log)
    w3.provider.add_receipt(RECOVERED, logs=[log], sender=ACCT.address, to=SEPOLIA_XRESERVE)
    other = deposited_log(SEPOLIA_XRESERVE, local_token=SEPOLIA_USDC, depositor=OTHER, remote_recipient32=ALEO32, value=2_000_000,
                          remote_domain=10002, remote_token32=REMOTE_TOKEN, max_fee=100_000, hook_data=hook, tx_hash=RECOVERED_2, log_index=1)
    w3.provider.history_logs.append(other)
    w3.provider.add_receipt(RECOVERED_2, logs=[other], sender=OTHER, to=SEPOLIA_XRESERVE)
    receipt = eth.recover_source(plan, cp)
    assert receipt.status == Status.ATTESTATION_PENDING and receipt.source_tx_id == RECOVERED
    nonce = encoding.xreserve_deposit_nonce(0, bytes.fromhex(RECOVERED[2:]), 3)
    assert receipt.protocol_state["nonce"] == "0x" + nonce.hex() and receipt.id == receipt.protocol_state["messageHash"]


def test_xreserve_multiple_matches_refuse_to_choose():
    w3 = fake_web3(chain_id=11155111)
    eth = make_bridge(environment="testnet", ethereum=Ethereum(w3=w3)).eth
    plan = _plan_for(DEFAULT_REGISTRY, USDC_ROUTE, amount_atomic=2_000_000, recipient=ALEO, sender=ACCT.address)
    hook = bytes(65)
    cp = xreserve_checkpoint(plan, hook)
    w3.provider.add_receipt(APPROVAL, block_number=0x65)
    log = deposited_log(SEPOLIA_XRESERVE, local_token=SEPOLIA_USDC, depositor=ACCT.address, remote_recipient32=ALEO32, value=2_000_000,
                        remote_domain=10002, remote_token32=REMOTE_TOKEN, max_fee=100_000, hook_data=hook, tx_hash=RECOVERED, log_index=3)
    w3.provider.history_logs.append(log)
    w3.provider.add_receipt(RECOVERED, logs=[log], sender=ACCT.address, to=SEPOLIA_XRESERVE)
    log2 = deposited_log(SEPOLIA_XRESERVE, local_token=SEPOLIA_USDC, depositor=ACCT.address, remote_recipient32=ALEO32, value=2_000_000,
                         remote_domain=10002, remote_token32=REMOTE_TOKEN, max_fee=100_000, hook_data=hook, tx_hash=RECOVERED_2, log_index=1)
    w3.provider.history_logs.append(log2)
    w3.provider.add_receipt(RECOVERED_2, logs=[log2], sender=ACCT.address, to=SEPOLIA_XRESERVE)
    with pytest.raises(BridgeError, match="Multiple matching xReserve deposits"):
        eth.recover_source(plan, cp)


def test_xreserve_required_scan_needs_confirmed_approval():
    w3 = fake_web3(chain_id=11155111)
    eth = make_bridge(environment="testnet", ethereum=Ethereum(w3=w3)).eth
    plan = _plan_for(DEFAULT_REGISTRY, USDC_ROUTE, amount_atomic=2_000_000, recipient=ALEO, sender=ACCT.address)
    cp = xreserve_checkpoint(plan, bytes(65))
    with pytest.raises(BridgeError, match="no confirmed approval block"):
        eth.recover_source(plan, cp, required=True)


def test_xreserve_required_scan_with_no_matching_deposit_is_not_fatal():
    """Same ruling for xReserve: a completed scan (known sender, confirmed approval) that matches
    nothing is a valid answer, not an inability to scan."""
    w3 = fake_web3(chain_id=11155111)
    eth = make_bridge(environment="testnet", ethereum=Ethereum(w3=w3)).eth
    plan = _plan_for(DEFAULT_REGISTRY, USDC_ROUTE, amount_atomic=2_000_000, recipient=ALEO, sender=ACCT.address)
    cp = xreserve_checkpoint(plan, bytes(65))
    w3.provider.add_receipt(APPROVAL, block_number=0x65)
    receipt = eth.recover_source(plan, cp, required=True)
    assert receipt.status == Status.SOURCE_SUBMISSION_PENDING and receipt.id == APPROVAL
    assert "eth_getLogs" in w3.provider.methods and w3.provider.sent == []


def test_xreserve_private_recovery_uses_checkpointed_hook_and_wrapper_recipient():
    w3 = fake_web3(chain_id=11155111)
    eth = make_bridge(environment="testnet", ethereum=Ethereum(w3=w3)).eth
    plan = _plan_for(DEFAULT_REGISTRY, USDC_ROUTE, amount_atomic=2_000_000, recipient=ALEO, sender=ACCT.address, mint_mode="private")
    hook = encoding.xreserve_hook_data("private", ALEO, "testnet", "7scalar")
    wrapper32 = encoding.aleo_address_to_bytes32(encoding.aleo_program_address("shielded_usdcx_wrapper.aleo", "testnet"))
    cp = xreserve_checkpoint(plan, hook)
    cp = dataclasses.replace(cp, source={**cp.source, "transactionId": DISPATCH})
    log = deposited_log(SEPOLIA_XRESERVE, local_token=SEPOLIA_USDC, depositor=ACCT.address, remote_recipient32=wrapper32, value=2_000_000,
                        remote_domain=10002, remote_token32=REMOTE_TOKEN, max_fee=100_000, hook_data=hook, tx_hash=DISPATCH, log_index=2)
    w3.provider.add_receipt(DISPATCH, logs=[log], sender=ACCT.address, to=SEPOLIA_XRESERVE)
    receipt = eth.recover_source(plan, cp)
    assert receipt.status == Status.ATTESTATION_PENDING and receipt.protocol_state["hookData"] == "0x" + hook.hex()
    assert receipt.protocol_state["remoteRecipientBytes32"] == "0x" + wrapper32.hex() and receipt.protocol_state["mintMode"] == "private"
    bad = dataclasses.replace(cp, source={**cp.source, "hookData": "0x02"})
    with pytest.raises(CheckpointInvalidError, match="hook data"):
        eth.recover_source(plan, bad)
