import pytest

from aleo_bridge.encoding import (xreserve_deposit_payload, xreserve_hook_data, xreserve_message_hash,
                                  xreserve_nonce_from_payload)
from aleo_bridge.errors import (BridgeError, CheckpointInvalidError, ConfigurationError,
                                 DeliveryUnknownError, UnsupportedRouteError)
from aleo_bridge.lifecycle import aleo_transaction_status, get_status, prepare
from aleo_bridge.types import Attestation, Receipt, Status
from tests.fakes.fake_bridge import ALEO_RECIPIENT, EVM_ADDRESS, SOL_ADDRESS, FakeBridge

SIG = "0x" + "11" * 65


def _inbound_private(b):
    plan = prepare(b.registry, source_chain="sepolia", source_asset="usdc", destination_chain="aleo-testnet", destination_asset="usdcx", amount="2",
                   recipient=ALEO_RECIPIENT, mint_mode="private")
    hook = xreserve_hook_data("private", ALEO_RECIPIENT, "testnet", "7scalar")
    payload = xreserve_deposit_payload(amount=2_000_000, remote_domain=10_002, remote_token=b"\x11" * 32,
                                       remote_recipient=b"\x22" * 32,
                                       local_token="0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238",
                                       depositor=EVM_ADDRESS, max_fee=100_000, nonce=b"\x00" * 32, hook_data=hook)
    message_hash = "0x" + xreserve_message_hash(payload).hex()
    receipt = Receipt(id=message_hash, protocol="xreserve", status=Status.ATTESTATION_PENDING,
                      source_tx_id="0x" + "22" * 32,
                      protocol_state={"routeId": plan.route_id, "mintMode": "private",
                                      "intendedRecipient": ALEO_RECIPIENT, "payload": "0x" + payload.hex(),
                                      "messageHash": message_hash, "bridgeProgram": "test_usdcx_bridge_v2.aleo"})
    return plan, payload, message_hash, receipt


def test_guards_and_terminal_passthrough():
    b = FakeBridge(ethereum=False)
    plan = prepare(b.registry, source_chain="aleo", source_asset="eth", destination_chain="ethereum", destination_asset="eth", amount="0.000000000000000001",
                   recipient=EVM_ADDRESS)
    with pytest.raises(CheckpointInvalidError, match="does not match"):
        get_status(b, plan, Receipt(id="x", protocol="hyperlane", status=Status.SOURCE_CONFIRMING,
                                    protocol_state={"routeId": "other"}))
    done = Receipt(id="x", protocol="hyperlane", status=Status.COMPLETED, protocol_state={"routeId": plan.route_id})
    assert get_status(b, plan, done) is done
    failed = done.replace(status=Status.FAILED)
    assert get_status(b, plan, failed) is failed and b.calls == [] and b.events == []
    expired = done.replace(status=Status.EXPIRED)
    assert get_status(b, plan, expired) is expired and b.calls == [] and b.events == []


def test_branch1_evm_approval_pending_delegates_to_eth_source_status():
    b = FakeBridge()
    plan = prepare(b.registry, source_chain="ethereum", source_asset="wbtc", destination_chain="aleo", destination_asset="wbtc", amount="0.001", recipient=ALEO_RECIPIENT)
    receipt = Receipt(id="0x" + "11" * 32, protocol="hyperlane", status=Status.SOURCE_APPROVAL_PENDING,
                      protocol_state={"routeId": plan.route_id, "approvalTxIds": ["0x" + "11" * 32]})
    b.eth.source_status_result = receipt.replace(status=Status.SOURCE_SUBMISSION_PENDING)
    out = get_status(b, plan, receipt)
    assert out.status is Status.SOURCE_SUBMISSION_PENDING and b.calls == [("eth.source_status", Status.SOURCE_APPROVAL_PENDING)]


@pytest.mark.parametrize("node_status,expected", [("accepted", Status.DELIVERY_PENDING), ("rejected", Status.FAILED)])
def test_branch2_aleo_source_confirming_reads_confirmed_transaction(node_status, expected):
    b = FakeBridge(ethereum=False)
    plan = prepare(b.registry, source_chain="aleo", source_asset="usdcx", destination_chain="ethereum", destination_asset="usdc", amount="2.1", recipient=EVM_ADDRESS)
    receipt = Receipt(id="at1burn", protocol="xreserve", status=Status.SOURCE_CONFIRMING, source_tx_id="at1burn",
                      protocol_state={"routeId": plan.route_id}, next_action={"kind": "stale"})
    pending = get_status(b, plan, receipt)
    assert pending is receipt                                  # TransactionNotFound → unchanged
    b.aleo.confirmed_transactions["at1burn"] = {"status": node_status, "type": "execute", "index": 3,
                                           "transaction": {"id": "at1burn"}, "finalize": []}
    out = get_status(b, plan, receipt)
    assert out.status is expected and out.next_action is None
    if expected is Status.FAILED:
        assert "rejected" in out.protocol_state["sourceError"]
    assert aleo_transaction_status(b, "at1burn")[0] == node_status
    assert aleo_transaction_status(b, "at1unknown") == ("pending", None)


def test_branch3_4_hyperlane_source_confirming_evm_and_solana():
    b = FakeBridge(solana=True)
    evm_plan = prepare(b.registry, source_chain="ethereum", source_asset="eth", destination_chain="aleo", destination_asset="eth", amount="0.000000000000000001",
                       recipient=ALEO_RECIPIENT)
    evm_receipt = Receipt(id="0x" + "aa" * 32, protocol="hyperlane", status=Status.SOURCE_CONFIRMING,
                          source_tx_id="0x" + "aa" * 32, protocol_state={"routeId": evm_plan.route_id})
    b.eth.source_status_result = evm_receipt.replace(status=Status.DELIVERY_PENDING,
                                                     protocol_state={**evm_receipt.protocol_state, "messageId": "0x" + "cd" * 32})
    assert get_status(b, evm_plan, evm_receipt).protocol_state["messageId"] == "0x" + "cd" * 32
    sol_plan = prepare(b.registry, source_chain="solana", source_asset="sol", destination_chain="aleo", destination_asset="sol", amount="0.000000001",
                       recipient=ALEO_RECIPIENT, sender=SOL_ADDRESS)
    sol_receipt = Receipt(id="sig", protocol="hyperlane", status=Status.SOURCE_CONFIRMING, source_tx_id="sig",
                          protocol_state={"routeId": sol_plan.route_id})
    b.sol.source_status_result = sol_receipt.replace(status=Status.EXPIRED,
                                                     protocol_state={**sol_receipt.protocol_state, "blockhashExpired": True,
                                                                     "sourceError": "Solana transaction expired before confirmation: sig"})
    assert get_status(b, sol_plan, sol_receipt).status is Status.EXPIRED
    assert b.calls == [("eth.source_status", Status.SOURCE_CONFIRMING), ("sol.source_status", Status.SOURCE_CONFIRMING)]


def test_branch5_hyperlane_delivery_via_destination_mailbox():
    b = FakeBridge()
    to_aleo = prepare(b.registry, source_chain="ethereum", source_asset="eth", destination_chain="aleo", destination_asset="eth", amount="0.000000000000000001",
                      recipient=ALEO_RECIPIENT)
    mid = "0x" + "cd" * 32
    receipt = Receipt(id=mid, protocol="hyperlane", status=Status.DELIVERY_PENDING, source_tx_id="0x" + "aa" * 32,
                      protocol_state={"routeId": to_aleo.route_id, "messageId": mid})
    assert get_status(b, to_aleo, receipt) is receipt
    b.hyperlane.delivered[mid] = True
    assert get_status(b, to_aleo, receipt).status is Status.COMPLETED
    assert b.calls[-1] == ("hyperlane.is_delivered", mid)

    to_evm = prepare(b.registry, source_chain="aleo", source_asset="eth", destination_chain="ethereum", destination_asset="eth", amount="0.000000000000000001",
                     recipient=EVM_ADDRESS)
    receipt2 = Receipt(id="at1x", protocol="hyperlane", status=Status.DELIVERY_PENDING, source_tx_id="at1x",
                       protocol_state={"routeId": to_evm.route_id, "messageId": mid})
    b.eth.delivered[mid] = True
    assert get_status(b, to_evm, receipt2).status is Status.COMPLETED
    assert b.calls[-1] == ("eth.is_delivered", mid)


def test_branch5_solana_delivery_pending_without_message_id_fills_from_logs():
    b = FakeBridge(solana=True)
    plan = prepare(b.registry, source_chain="solana", source_asset="sol", destination_chain="aleo", destination_asset="sol", amount="0.000000001",
                   recipient=ALEO_RECIPIENT, sender=SOL_ADDRESS)
    sig = "5igNature" * 8
    receipt = Receipt(id=sig, protocol="hyperlane", status=Status.DELIVERY_PENDING, source_tx_id=sig,
                      protocol_state={"routeId": plan.route_id, "messageIdUnavailable": True})

    # the log read raises -> unchanged, never falls through to is_delivered with the signature
    b.sol.transaction_logs_error = RuntimeError("solana RPC is down")
    assert get_status(b, plan, receipt) is receipt
    assert not any(call[0] == "hyperlane.is_delivered" for call in b.calls)
    b.sol.transaction_logs_error = None

    # the log read succeeds but carries no dispatch line -> still unavailable, unchanged
    b.sol.transaction_logs_result = ["Program log: something unrelated"]
    assert get_status(b, plan, receipt) is receipt
    assert not any(call[0] == "hyperlane.is_delivered" for call in b.calls)

    # the dispatch line is present -> message id filled in, then the destination Mailbox is checked
    mid = "0x" + "cd" * 32
    b.sol.transaction_logs_result = [f"Program log: Dispatched message to 1399811149, ID {mid}"]
    out = get_status(b, plan, receipt)
    assert out.id == mid
    assert out.protocol_state["messageId"] == mid
    assert "messageIdUnavailable" not in out.protocol_state
    assert out.status is Status.DELIVERY_PENDING
    assert b.calls[-1] == ("hyperlane.is_delivered", mid)

    b.hyperlane.delivered[mid] = True
    assert get_status(b, plan, receipt).status is Status.COMPLETED


def test_branch6_aleo_origin_balance_diff_fallback():
    b = FakeBridge(solana=True)
    plan = prepare(b.registry, source_chain="aleo", source_asset="sol", destination_chain="solana", destination_asset="sol", amount="0.000000001", recipient=SOL_ADDRESS)
    receipt = Receipt(id="at1source", protocol="hyperlane", status=Status.DELIVERY_PENDING, source_tx_id="at1source",
                      protocol_state={"routeId": plan.route_id, "destinationBalanceBeforeAtomic": "100",
                                      "expectedDestinationIncreaseAtomic": "1"})
    b.sol.balance_lamports = 100
    assert get_status(b, plan, receipt) is receipt
    b.sol.balance_lamports = 101
    assert get_status(b, plan, receipt).status is Status.COMPLETED
    no_baseline = receipt.replace(protocol_state={"routeId": plan.route_id})
    assert get_status(b, plan, no_baseline) is no_baseline                      # branch 7: unchanged
    b2 = FakeBridge(ethereum=False)
    with pytest.raises(DeliveryUnknownError, match="destination balance"):
        get_status(b2, plan, receipt)


def test_branch6_a_failing_destination_balance_read_propagates():
    """Carried from the Task 6 review (item 8): in branch 6 the destination balance is the DELIVERY
    SIGNAL, not an advisory baseline. A transport failure must surface here so ``wait``'s transient
    classifier can retry it — swallowing it would read as "not delivered yet" forever."""
    b = FakeBridge(solana=True)
    plan = prepare(b.registry, source_chain="aleo", source_asset="sol", destination_chain="solana", destination_asset="sol", amount="0.000000001",
                   recipient=SOL_ADDRESS)
    receipt = Receipt(id="at1source", protocol="hyperlane", status=Status.DELIVERY_PENDING,
                      source_tx_id="at1source",
                      protocol_state={"routeId": plan.route_id, "destinationBalanceBeforeAtomic": "100",
                                      "expectedDestinationIncreaseAtomic": "1"})

    def boom():
        raise BridgeError("Solana RPC request failed with HTTP status 429")

    b.sol.balance = boom
    with pytest.raises(BridgeError, match="429"):
        get_status(b, plan, receipt)


def test_branch8_and_9_xreserve_outbound_and_not_implemented():
    b = FakeBridge(ethereum=False)
    plan = prepare(b.registry, source_chain="aleo", source_asset="usdcx", destination_chain="ethereum", destination_asset="usdc", amount="2.1", recipient=EVM_ADDRESS)
    receipt = Receipt(id="at1burn", protocol="xreserve", status=Status.DELIVERY_PENDING, source_tx_id="at1burn",
                      protocol_state={"routeId": plan.route_id})
    assert get_status(b, plan, receipt) is receipt
    with pytest.raises(UnsupportedRouteError, match="not implemented"):
        get_status(b, plan, receipt.replace(status=Status.ATTESTATION_PENDING))


def test_branch8_xreserve_aleo_to_evm_terminates_on_the_recorded_balance_baseline():
    """I2: Circle exposes no delivery query for this direction, so veil leaves it pending forever.
    When ``execute`` could baseline the recipient's USDC balance, the same arithmetic branch 6 uses
    is the delivery signal here too — otherwise ``wait`` can never terminate an Aleo→EVM burn.

    The baseline is NET of the withdrawal fee (R1): the 2026-09-18 testnet burn of 2.000001 USDCx
    delivered 996_501 atomic USDC, far short of the 2_000_001 burned — a predicate on the full
    amount would never fire for the real transfer, and would fire for unrelated inflow."""
    b = FakeBridge()
    plan = prepare(b.registry, source_chain="aleo", source_asset="usdcx", destination_chain="ethereum", destination_asset="usdc", amount="2.000001",
                   recipient=EVM_ADDRESS)
    receipt = Receipt(id="at1burn", protocol="xreserve", status=Status.DELIVERY_PENDING, source_tx_id="at1burn",
                      next_action={"kind": "noop"},
                      protocol_state={"routeId": plan.route_id, "destinationBalanceBeforeAtomic": "100",
                                      "expectedDestinationIncreaseAtomic": "1"})
    b.eth.balances["ethereum/usdc"] = 100
    assert get_status(b, plan, receipt) is receipt                    # nothing delivered yet
    b.eth.balances["ethereum/usdc"] = 100 + 996_501                   # the real testnet delivery
    done = get_status(b, plan, receipt)
    assert done.status is Status.COMPLETED and done.next_action is None


def test_branch8_falls_back_to_veils_passthrough_without_a_baseline_or_a_reader():
    """The balance check is an addition, never a new failure mode: with no baseline in the receipt,
    or with no connection able to read the recipient's balance, branch 8 stays veil's passthrough
    rather than raising DeliveryUnknownError the way branch 6 does."""
    b = FakeBridge(ethereum=False)
    plan = prepare(b.registry, source_chain="aleo", source_asset="usdcx", destination_chain="ethereum", destination_asset="usdc", amount="2.000001",
                   recipient=EVM_ADDRESS)
    baselined = Receipt(id="at1burn", protocol="xreserve", status=Status.DELIVERY_PENDING, source_tx_id="at1burn",
                        protocol_state={"routeId": plan.route_id, "destinationBalanceBeforeAtomic": "100",
                                        "expectedDestinationIncreaseAtomic": "1"})
    assert get_status(b, plan, baselined) is baselined                # no reader: pending, not an error

    b2 = FakeBridge()
    bare = baselined.replace(protocol_state={"routeId": plan.route_id})
    assert get_status(b2, plan, bare) is bare
    assert not any(call[0] == "eth.balance" for call in b2.calls)


def test_branch10_nullifier_first_then_attestation_then_private_action():
    b = FakeBridge(environment="testnet")
    plan, payload, message_hash, receipt = _inbound_private(b)
    # nullifier read comes first for every inbound pending status, and derives the nonce from the payload
    b.xreserve.delivered_nonces.add("0x" + xreserve_nonce_from_payload(payload).hex())
    waiting = receipt.replace(status=Status.DESTINATION_ACTION_REQUIRED,
                              next_action={"kind": "xreserve-private-mint", "chainId": "aleo-testnet"})
    out = get_status(b, plan, waiting)
    assert out.status is Status.COMPLETED and out.next_action is None
    assert b.calls[0][0] == "xreserve.is_delivered"
    b.xreserve.delivered_nonces.clear()
    # stored nonce wins over the payload
    stored = receipt.replace(status=Status.DELIVERY_PENDING,
                             protocol_state={**receipt.protocol_state, "nonce": "0x" + "33" * 32})
    b.xreserve.delivered_nonces.add("0x" + "33" * 32)
    assert get_status(b, plan, stored).status is Status.COMPLETED
    b.xreserve.delivered_nonces.clear()
    # attestation pending → unchanged; complete + private → DESTINATION_ACTION_REQUIRED
    assert get_status(b, plan, receipt) is receipt
    b.xreserve.attestations[message_hash] = Attestation(payload=payload, message_hash=bytes.fromhex(message_hash[2:]),
                                                         attestation=bytes.fromhex(SIG[2:]), status="complete")
    ready = get_status(b, plan, receipt)
    assert ready.status is Status.DESTINATION_ACTION_REQUIRED
    assert ready.next_action == {"kind": "xreserve-private-mint", "chainId": "aleo-testnet"}
    assert ready.protocol_state["attestation"] == SIG
    assert get_status(b, plan, ready) is ready                                    # action required → unchanged
    # public mode → DELIVERY_PENDING with attestation kept
    public_plan = prepare(b.registry, source_chain="sepolia", source_asset="usdc", destination_chain="aleo-testnet", destination_asset="usdcx", amount="2",
                          recipient=ALEO_RECIPIENT)
    public = receipt.replace(protocol_state={**receipt.protocol_state, "mintMode": "public"})
    out = get_status(b, public_plan, public)
    assert out.status is Status.DELIVERY_PENDING and out.protocol_state["attestation"] == SIG
    with pytest.raises(CheckpointInvalidError, match="message hash"):
        get_status(b, plan, receipt.replace(protocol_state={"routeId": plan.route_id}))


def test_branch10_source_confirming_and_destination_confirming():
    b = FakeBridge(environment="testnet")
    plan, payload, message_hash, receipt = _inbound_private(b)
    confirming = receipt.replace(status=Status.SOURCE_CONFIRMING)
    b.eth.source_status_result = receipt
    assert get_status(b, plan, confirming) is receipt and b.calls[-1][0] == "eth.source_status"
    minting = receipt.replace(status=Status.DESTINATION_CONFIRMING, destination_tx_id="at1private",
                              protocol_state={**receipt.protocol_state, "attestation": SIG})
    assert get_status(b, plan, minting) is minting
    b.aleo.confirmed_transactions["at1private"] = {"status": "accepted", "type": "execute"}
    assert get_status(b, plan, minting).status is Status.COMPLETED
    b.aleo.confirmed_transactions["at1private"] = {"status": "rejected", "type": "execute", "rejected": {"type": "execution"}}
    out = get_status(b, plan, minting)
    assert out.status is Status.FAILED and "at1private" in out.protocol_state["destinationError"]
    with pytest.raises(CheckpointInvalidError, match="destination transaction id"):
        get_status(b, plan, minting.replace(destination_tx_id=None))


def test_branch10_delivered_wins_over_attestation_pending():
    # Carried from the Task 5 review (item 7): the destination nullifier check (invariant 6) must
    # run BEFORE the attestation read for EVERY inbound-pending status, including
    # ATTESTATION_PENDING itself, not just DESTINATION_ACTION_REQUIRED/DELIVERY_PENDING — so a
    # nonce that is already delivered short-circuits straight to COMPLETED even when a complete
    # attestation is also scripted, and xreserve.get_attestation is never called.
    b = FakeBridge(environment="testnet")
    plan, payload, message_hash, receipt = _inbound_private(b)
    nonce = "0x" + xreserve_nonce_from_payload(payload).hex()
    b.xreserve.delivered_nonces.add(nonce)
    b.xreserve.attestations[message_hash] = Attestation(payload=payload, message_hash=bytes.fromhex(message_hash[2:]),
                                                         attestation=bytes.fromhex(SIG[2:]), status="complete")
    out = get_status(b, plan, receipt)
    assert out.status is Status.COMPLETED
    assert not any(call[0] == "xreserve.get_attestation" for call in b.calls)


def test_message_id_never_falls_back_to_source_tx_id():
    # Carried from the Task 5 review (item 8): a DispatchId log that could not be read leaves
    # receipt.id == receipt.source_tx_id (the source transaction hash) — that must never be
    # mistaken for the Hyperlane message id, even though it has the same 0x + 64-hex shape.
    b = FakeBridge()
    to_aleo = prepare(b.registry, source_chain="ethereum", source_asset="eth", destination_chain="aleo", destination_asset="eth", amount="0.000000000000000001",
                      recipient=ALEO_RECIPIENT)
    tx = "0x" + "aa" * 32
    receipt = Receipt(id=tx, protocol="hyperlane", status=Status.DELIVERY_PENDING, source_tx_id=tx,
                      protocol_state={"routeId": to_aleo.route_id})
    assert get_status(b, to_aleo, receipt) is receipt
    assert not any(call[0] == "hyperlane.is_delivered" for call in b.calls)


def test_solana_message_id_fill_in_raises_on_missing_connection():
    # Carried from the Task 5 review (item 9): resolving the Solana connection must happen OUTSIDE
    # the log-read try/except, so a missing connection surfaces as ConfigurationError instead of
    # being swallowed as "still unavailable".
    b = FakeBridge(ethereum=False, solana=False)
    plan = prepare(b.registry, source_chain="solana", source_asset="sol", destination_chain="aleo", destination_asset="sol", amount="0.000000001",
                   recipient=ALEO_RECIPIENT, sender=SOL_ADDRESS)
    sig = "5igNature" * 8
    receipt = Receipt(id=sig, protocol="hyperlane", status=Status.DELIVERY_PENDING, source_tx_id=sig,
                      protocol_state={"routeId": plan.route_id, "messageIdUnavailable": True})
    with pytest.raises(ConfigurationError, match="Solana"):
        get_status(b, plan, receipt)


@pytest.mark.parametrize('source,destination,asset,address', [
    ('usdcx', 'ethereum', 'usdc', EVM_ADDRESS),
    ('sol', 'solana', 'sol', SOL_ADDRESS),
])
def test_destination_balance_reads_recipient_without_destination_signer(source, destination, asset, address):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from aleo_bridge.lifecycle import _read_destination_balance, resolve_route
    b = FakeBridge(solana=True)
    setattr(b, destination, SimpleNamespace(address=None))
    reader = b.eth if destination == 'ethereum' else b.sol
    reader.balance = Mock(return_value=123)
    plan = prepare(b.registry, source_chain='aleo', source_asset=source,
                   destination_chain=destination, destination_asset=asset,
                   amount='3', recipient=address)
    resolved = resolve_route(b.registry, plan)
    assert _read_destination_balance(b, plan, resolved) == 123
    assert reader.balance.call_args.kwargs == {'address': address}
