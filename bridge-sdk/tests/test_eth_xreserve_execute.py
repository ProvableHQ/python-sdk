import dataclasses
import json

import pytest
from eth_abi import decode
from eth_account import Account
from eth_utils import keccak
from web3 import Web3

from aleo_bridge import encoding
from aleo_bridge.errors import (BridgeError, ConfigurationError, RegistryVersionMismatchError, RouteUnavailableError)
from aleo_bridge.eth import Ethereum
from aleo_bridge.types import DepositReceipt, Status
from tests.fakes.fake_web3 import deposited_log, fake_web3, make_bridge, tx_hash_for

KEY = "0x" + "11" * 32
ACCT = Account.from_key(KEY)
ALEO = "aleo1kypwp5m7qtk9mwazgcpg0tq8aal23mnrvwfvug65qgcg9xvsrqgspyjm6n"
USDC = "0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238"
XRESERVE = "0x008888878f94C0d87defdf0B07f46B93C1934442"
OTHER = "0x0000000000000000000000000000000000000009"
REMOTE_TOKEN = bytes.fromhex("b143ed52c774cd1d4a519d0e796f15916be5a9e1d45edcd9852dd23f68f53401")
APPROVE = keccak(text="approve(address,uint256)")[:4].hex()
DEPOSIT = keccak(text="depositToRemote(uint256,uint32,bytes32,address,uint256,bytes)")[:4].hex()


def deposit_fields(tx):
    """The DepositedToRemote fields the contract would emit for this depositToRemote calldata."""
    value, remote_domain, remote_recipient, local_token, max_fee, hook = decode(
        ["uint256", "uint32", "bytes32", "address", "uint256", "bytes"], bytes.fromhex(tx["data"][10:]))
    return {"local_token": Web3.to_checksum_address(local_token), "depositor": tx["from"],
            "remote_recipient32": remote_recipient, "value": value, "remote_domain": remote_domain,
            "remote_token32": REMOTE_TOKEN, "max_fee": max_fee, "hook_data": hook}


def deposit_logs(*, log_index=3, **overrides):
    """Echo the depositToRemote calldata back as a DepositedToRemote log, optionally corrupting fields."""
    def logs(tx):
        if tx["to"] != Web3.to_checksum_address(XRESERVE) or tx["data"][2:10] != DEPOSIT:
            return []
        return [deposited_log(XRESERVE, tx_hash=tx["hash"], log_index=log_index, **{**deposit_fields(tx), **overrides})]
    return logs


def setup(*, allowance=0, logs=None, checkpoints=None):
    w3 = fake_web3(chain_id=11155111, token_balances={(USDC, ACCT.address): 3_000_000},
                   allowances={(USDC, ACCT.address, XRESERVE): allowance})
    w3.provider.receipt_logs = logs or deposit_logs()
    bridge = make_bridge(environment="testnet", ethereum=Ethereum(w3=w3, private_key=KEY), checkpoints=checkpoints)
    return bridge.eth, w3


def test_build_lists_approve_then_deposit_with_zero_value():
    eth, w3 = setup()
    txs = eth.deposit_usdc(ALEO, amount="2", mint_mode="record").build()
    assert [t["to"] for t in txs] == [Web3.to_checksum_address(USDC), Web3.to_checksum_address(XRESERVE)]
    assert txs[0]["data"][2:10] == APPROVE and txs[1]["data"][2:10] == DEPOSIT
    assert txs[0]["value"] == 0 and txs[1]["value"] == 0 and w3.provider.sent == []


def test_record_mode_deposit_derives_nonce_payload_and_message_hash():
    eth, w3 = setup()
    seen = []
    result = eth.deposit_usdc(ALEO, amount="2", mint_mode="record").send(poll_seconds=0.001, on_checkpoint=seen.append)
    assert isinstance(result, DepositReceipt)
    sent = w3.provider.sent
    assert len(sent) == 2 and sent[1]["value"] == 0 and sent[1]["to"] == Web3.to_checksum_address(XRESERVE)
    assert sent[0]["data"][2:].lower() == APPROVE + XRESERVE[2:].lower().rjust(64, "0") + format(2_000_000, "064x")
    receipt = result.receipt
    assert receipt.status == Status.ATTESTATION_PENDING and receipt.protocol == "xreserve"
    assert receipt.source_tx_id == tx_hash_for(2) == result.transaction_id
    hook = b"\x01" + bytes(64)
    recipient32 = encoding.aleo_address_to_bytes32(ALEO)
    nonce = encoding.xreserve_deposit_nonce(0, bytes.fromhex(tx_hash_for(2)[2:]), 3)
    payload = encoding.xreserve_deposit_payload(amount=2_000_000, remote_domain=10002, remote_token=REMOTE_TOKEN,
                                                remote_recipient=recipient32, local_token=USDC, depositor=ACCT.address,
                                                max_fee=100_000, nonce=nonce, hook_data=hook)
    message_hash = "0x" + encoding.xreserve_message_hash(payload).hex()
    assert len(payload) == 305
    assert receipt.id == message_hash == result.message_hash == receipt.protocol_state["messageHash"]
    assert result.nonce == "0x" + nonce.hex() == receipt.protocol_state["nonce"]
    assert receipt.protocol_state["payload"] == "0x" + payload.hex()
    state = receipt.protocol_state
    assert state["routeId"] == "xreserve:sepolia/usdc->aleo-testnet/usdcx" and state["approvalTxIds"] == [tx_hash_for(1)]
    assert state["sourceSender"] == ACCT.address and state["mintMode"] == "record" and state["intendedRecipient"] == ALEO
    assert state["xReserveContract"] == Web3.to_checksum_address(XRESERVE) and state["tokenAddress"] == Web3.to_checksum_address(USDC)
    assert state["sourceChainId"] == 11155111 and state["sourceDomain"] == 0 and state["remoteDomain"] == 10002
    assert state["remoteRecipientBytes32"] == "0x" + recipient32.hex() and state["hookData"] == "0x" + hook.hex()
    assert state["amountAtomic"] == "2000000" and state["maxFeeAtomic"] == "100000" and state["depositLogIndex"] == 3
    assert state["bridgeProgram"] == "test_usdcx_bridge_v2.aleo" and state["wrapperProgram"] == "shielded_usdcx_wrapper.aleo"
    assert [cp.source for cp in seen] == [
        {"approvalTransactionIds": [tx_hash_for(1)], "hookData": "0x" + hook.hex()},
        {"approvalTransactionIds": [tx_hash_for(1)], "transactionId": tx_hash_for(2), "hookData": "0x" + hook.hex()},
        {"approvalTransactionIds": [tx_hash_for(1)], "transactionId": tx_hash_for(2), "hookData": "0x" + hook.hex()},
    ]
    assert seen[-1].id == message_hash and seen[-1].intent["mintMode"] == "record"


def test_sufficient_allowance_skips_approval():
    eth, w3 = setup(allowance=5_000_000)
    result = eth.deposit_usdc(ALEO, amount_atomic=2_000_000).send(poll_seconds=0.001)
    assert len(w3.provider.sent) == 1 and result.receipt.protocol_state["approvalTxIds"] == []
    assert result.receipt.protocol_state["hookData"] == "0x" + "00" * 65


def test_private_mode_deposits_to_wrapper_and_never_persists_the_secret():
    eth, _ = setup(allowance=5_000_000)
    seen = []
    result = eth.deposit_usdc(ALEO, amount="2", mint_mode="private", secret_nonce="7scalar").send(
        poll_seconds=0.001, on_checkpoint=seen.append)
    wrapper32 = encoding.aleo_address_to_bytes32(encoding.aleo_program_address("shielded_usdcx_wrapper.aleo", "testnet"))
    state = result.receipt.protocol_state
    assert state["remoteRecipientBytes32"] == "0x" + wrapper32.hex() and state["intendedRecipient"] == ALEO
    assert state["hookData"] == "0x" + encoding.xreserve_hook_data("private", ALEO, "testnet", "7scalar").hex()
    assert "7scalar" not in json.dumps(state) and all("7scalar" not in cp.to_json() for cp in seen)
    assert seen[0].source["hookData"].startswith("0x02") and len(seen[0].source["hookData"]) == 132


def test_event_mismatch_or_absence_raises():
    eth, _ = setup(allowance=5_000_000, logs=deposit_logs(value=1))
    with pytest.raises(BridgeError, match="does not match the prepared transfer"):
        eth.deposit_usdc(ALEO, amount="2").send(poll_seconds=0.001)
    eth, _ = setup(allowance=5_000_000, logs=deposit_logs(remote_token32=bytes(32)))
    with pytest.raises(BridgeError, match="does not match the prepared transfer"):
        eth.deposit_usdc(ALEO, amount="2").send(poll_seconds=0.001)
    eth, _ = setup(allowance=5_000_000, logs=lambda tx: [])
    with pytest.raises(BridgeError, match="DepositedToRemote"):
        eth.deposit_usdc(ALEO, amount="2").send(poll_seconds=0.001)


@pytest.mark.parametrize("field, corrupted", [
    ("local_token", OTHER),
    ("depositor", OTHER),
    ("value", 1_999_999),
    ("remote_domain", 10_003),
    ("remote_recipient32", bytes(32)),
    ("remote_token32", bytes(32)),
    ("max_fee", 99_999),
    ("hook_data", b"\x01" + bytes(64)),
])
def test_every_re_verified_deposit_field_must_match(field, corrupted):
    """All eight canonical DepositedToRemote fields are load-bearing: corrupting any one of them
    alone must make the event stop counting as this transfer's deposit."""
    eth, _ = setup(allowance=5_000_000, logs=deposit_logs(**{field: corrupted}))
    with pytest.raises(BridgeError, match="does not match the prepared transfer"):
        eth.deposit_usdc(ALEO, amount="2").send(poll_seconds=0.001)


@pytest.mark.parametrize("ours_first", [True, False])
def test_our_deposit_event_is_selected_among_other_accounts_deposits(ours_first):
    """A batched transaction carries several accounts' deposits; ours is whichever event matches all
    eight fields, not whichever happens to be last in the receipt."""
    def logs(tx):
        if tx["to"] != Web3.to_checksum_address(XRESERVE) or tx["data"][2:10] != DEPOSIT:
            return []
        fields = deposit_fields(tx)
        ours = deposited_log(XRESERVE, tx_hash=tx["hash"], log_index=3, **fields)
        theirs = deposited_log(XRESERVE, tx_hash=tx["hash"], log_index=7, **{**fields, "depositor": OTHER})
        return [ours, theirs] if ours_first else [theirs, ours]

    eth, _ = setup(allowance=5_000_000, logs=logs)
    result = eth.deposit_usdc(ALEO, amount="2").send(poll_seconds=0.001)
    assert result.receipt.status == Status.ATTESTATION_PENDING
    assert result.receipt.protocol_state["depositLogIndex"] == 3        # ours, whatever the order


def test_timeouts_return_pending_receipts():
    eth, w3 = setup()
    w3.provider.pending.add(tx_hash_for(1))
    result = eth.deposit_usdc(ALEO, amount="2").send(timeout_seconds=0.01, poll_seconds=0.001)
    assert result.receipt.status == Status.SOURCE_APPROVAL_PENDING and result.receipt.source_tx_id is None
    assert result.receipt.id == tx_hash_for(1) and result.message_hash == "" and result.nonce == ""
    assert len(w3.provider.sent) == 1
    eth, w3 = setup(allowance=5_000_000)
    w3.provider.pending.add(tx_hash_for(1))
    seen = []
    result = eth.deposit_usdc(ALEO, amount="2").send(timeout_seconds=0.01, poll_seconds=0.001, on_checkpoint=seen.append)
    assert result.receipt.status == Status.SOURCE_CONFIRMING and result.receipt.source_tx_id == tx_hash_for(1)
    assert [cp.source for cp in seen] == [{"transactionId": tx_hash_for(1), "hookData": "0x" + "00" * 65}]


def test_plan_driven_deposit_is_identical_to_the_recipient_driven_one():
    eth, w3 = setup(allowance=5_000_000)
    quote = eth.quote_deposit_usdc(ALEO, amount="2", mint_mode="record")
    by_recipient = eth.deposit_usdc(ALEO, amount="2", mint_mode="record").build()
    by_plan = eth.deposit_usdc(plan=quote.plan).build()                  # mint_mode comes from the plan
    assert by_plan == by_recipient and len(by_plan) == 1 and by_plan[0]["data"][2:10] == DEPOSIT
    assert eth.quote_deposit_usdc(plan=quote.plan) == quote
    assert w3.provider.sent == []


def test_plan_driven_deposit_rejects_a_tampered_stale_or_foreign_plan():
    eth, w3 = setup(allowance=5_000_000)
    plan = eth.quote_deposit_usdc(ALEO, amount="2").plan
    w3.provider.methods.clear()
    with pytest.raises(BridgeError, match="plan does not match the requested transfer: amount"):
        eth.deposit_usdc(plan=dataclasses.replace(plan, amount_atomic=2_000_001))
    with pytest.raises(BridgeError, match="plan does not match the requested transfer: steps"):
        eth.deposit_usdc(plan=dataclasses.replace(plan, mint_mode="private"))   # steps still say "protocol"
    with pytest.raises(BridgeError, match="plan does not match the requested transfer: mint_mode"):
        eth.deposit_usdc(mint_mode="record", plan=plan)
    with pytest.raises(RegistryVersionMismatchError):
        eth.deposit_usdc(plan=dataclasses.replace(plan, registry_version="0000-00-00.stale"))
    with pytest.raises(RouteUnavailableError, match="not a xreserve one"):
        eth.deposit_usdc(plan=dataclasses.replace(plan, route_id="hyperlane:ethereum/eth->aleo/eth"))
    with pytest.raises(ConfigurationError, match="does not match connected account"):
        eth.deposit_usdc(plan=dataclasses.replace(plan, sender=OTHER))
    with pytest.raises(ValueError, match="not both"):
        eth.quote_deposit_usdc(ALEO, amount="2", sender=ACCT.address, plan=plan)
    assert w3.provider.methods == [] and w3.provider.sent == []


def test_reverted_deposit_raises():
    eth, w3 = setup(allowance=5_000_000)
    w3.provider.reverted.add(tx_hash_for(1))
    with pytest.raises(BridgeError, match="reverted"):
        eth.deposit_usdc(ALEO, amount="2").send(poll_seconds=0.001)
