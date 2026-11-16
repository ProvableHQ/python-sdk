"""Observe Verulink delivery without treating screening holding as success."""
from __future__ import annotations

from typing import Any

from ._verulink_attestation import fetch_signatures, verified_signatures
from ._verulink_context import Context
from ._verulink_source import (aleo_transaction, events, get_transaction, hex_value, inputs, prepared_transaction,
                               source_packet, transition, tx_hash, validate_aleo_claim, validate_packet)
from ._verulink_wire import Packet, evm_packet, packet_hash, same, same_packet
from .errors import BridgeError
from .types import Plan, Receipt, Status


def update(receipt: Receipt, status: Status, **state: Any) -> Receipt:
    return receipt.replace(status=status, next_action=None, protocol_state={**receipt.protocol_state, **state})


def consumed(c: Context, p: Packet) -> bool:
    if c.outbound:
        value = c.read(c.evm_bridge, "consumedPackets", [int(p["sequence"])])
        if value == bytes(32):
            return False
        if value != packet_hash(p):
            raise BridgeError("Verulink sequence was consumed with a different packet hash")
        return True
    value = c.mapping(c.aleo_bridge, "in_packet_consumed", f"{{ chain_id: {p['sourceChain']}u128, sequence: {p['sequence']}u64 }}")
    if value is True:
        return True
    if value is False or value is None:
        return False
    raise BridgeError("Invalid Verulink consumed-packet flag")


def destination_status(c: Context, p: Packet, receipt: Receipt) -> Receipt | None:
    """Read the claim's outcome; ``None`` means it failed onchain and delivered nothing."""
    tx_id = receipt.destination_tx_id
    if not tx_id:
        raise BridgeError("Verulink claim transaction id is missing")
    if c.outbound:
        tx_hash(tx_id)
        result = c.evm.get_receipt(tx_id)
        if result is None:
            return update(receipt, Status.DESTINATION_CONFIRMING)
        if not same(hex_value(result.get("transactionHash")), tx_id) or result.get("blockNumber") is None:
            raise BridgeError("Verulink claim receipt identity mismatch")
        if result.get("status") not in (0, 1):
            raise BridgeError("Invalid Verulink claim receipt status")
        if c.evm.w3.eth.block_number < result["blockNumber"] + c.confirmations - 1:
            return update(receipt, Status.DESTINATION_CONFIRMING)
        if result["status"] == 0:
            return None
        tx = get_transaction(c, tx_id)
        if tx is None:  # A lagging RPC backend can serve the receipt before the transaction.
            return update(receipt, Status.DESTINATION_CONFIRMING)
        if (not same(hex_value(tx.get("hash")), tx_id) or not same(tx.get("to"), c.evm_service)
                or tx.get("blockNumber") != result["blockNumber"]):
            raise BridgeError("Verulink claim transaction identity mismatch")
        fn, args = c.contract(c.evm_service).decode_function_input(tx["input"])
        if fn.fn_name != "tokenReceive" or not same_packet(p, evm_packet(args["packet"])):
            raise BridgeError("Verulink claim calldata does not match the source packet")
        consume_event = any(e["chainId"] == int(p["sourceChain"]) and e["sequence"] == int(p["sequence"])
                            and e["packetHash"] == packet_hash(p) for e in events(c, result, c.evm_bridge, "Consumed"))
        if not consume_event or not consumed(c, p):
            raise BridgeError("Verulink claim receipt lacks the expected consumption evidence")
        mints = [e for e in events(c, result, c.evm_token, "Transfer")
                 if same(e["from"], "0x" + "00" * 20) and e["value"] == int(p["amount"])]
        if any(same(e["to"], p["recipient"]) for e in mints):
            return update(receipt, Status.COMPLETED)
        if mints and any(same(e["to"], c.read(c.evm_service, "holding")) for e in mints):
            return update(receipt, Status.FAILED, destinationError="Verulink screening placed the transferred ALEO in holding", held=True)
        raise BridgeError("Verulink claim receipt does not prove recipient delivery")
    tx = aleo_transaction(c, tx_id)
    if tx == "pending":
        return update(receipt, Status.DESTINATION_CONFIRMING)
    if tx == "rejected":
        return None
    claim = validate_aleo_claim(c, p, tx)
    if not consumed(c, p):
        raise BridgeError("Accepted Verulink claim has no consumed-packet flag")
    screening = next((o.get("value") for o in claim.get("outputs", []) if o.get("type") == "public"), None)
    if screening == "false":
        return update(receipt, Status.FAILED, destinationError="Verulink screening placed the transferred ALEO in holding", held=True)
    if screening != "true":
        raise BridgeError("Verulink Aleo claim has no screening result")
    transfer = inputs(transition(tx, "credits.aleo", "transfer_public"))
    if len(transfer) != 2 or transfer[0] != p["recipient"] or transfer[1] != int(p["amount"]):
        raise BridgeError("Verulink Aleo claim does not prove recipient delivery")
    return update(receipt, Status.COMPLETED)


def approval_status(c: Context, plan: Plan, receipt: Receipt) -> Receipt:
    ids = receipt.protocol_state.get("approvalTxIds")
    if not isinstance(ids, list) or not ids:
        raise BridgeError("Invalid Verulink approval checkpoint")
    for tx_id in ids:
        tx_hash(tx_id)
        result, tx = c.evm.get_receipt(tx_id), get_transaction(c, tx_id)
        if result is None or tx is None:
            return update(receipt, Status.SOURCE_APPROVAL_PENDING)
        if (not same(hex_value(result.get("transactionHash")), tx_id)
                or not same(hex_value(tx.get("hash")), tx_id) or not same(tx.get("to"), c.evm_token)
                or not same(tx.get("from"), plan.sender or receipt.protocol_state.get("sourceSender"))
                or result.get("blockNumber") is None or tx.get("blockNumber") != result["blockNumber"]):
            raise BridgeError("Verulink approval does not match the transfer")
        fn, args = c.contract(c.evm_token).decode_function_input(tx["input"])
        if fn.fn_name != "approve" or not same(args["spender"], c.evm_service) or args["amount"] != plan.amount_atomic:
            raise BridgeError("Verulink approval does not match the transfer")
        if result.get("status") == 0:
            return update(receipt, Status.FAILED, sourceError=f"Verulink approval reverted: {tx_id}")
        if result.get("status") != 1:
            raise BridgeError("Invalid Verulink approval status")
    return update(receipt, Status.SOURCE_SUBMISSION_PENDING)


def get_status(c: Context, plan: Plan, receipt: Receipt, session: Any) -> Receipt:
    if receipt.protocol != "verulink" or receipt.protocol_state.get("routeId") != c.route.id:
        raise BridgeError("Verulink receipt does not match the route")
    c.check_chain()
    if not receipt.source_tx_id:
        if not c.outbound and receipt.status in (Status.SOURCE_APPROVAL_PENDING, Status.SOURCE_SUBMISSION_PENDING):
            return approval_status(c, plan, receipt)
        raise BridgeError("Verulink receipt has no source transaction")
    p = receipt.protocol_state.get("packet")
    if isinstance(p, dict):
        # Verified from the confirmed source earlier in this process; a checkpoint never stores it.
        validate_packet(c, plan, p)
    else:
        p = source_packet(c, plan, receipt.source_tx_id)
        if p == "pending":
            return update(receipt, Status.SOURCE_CONFIRMING)
        if p == "rejected":
            return update(receipt, Status.FAILED, sourceError="Verulink source transaction was rejected")
        assert isinstance(p, dict)
    verified = update(receipt, Status.ATTESTATION_PENDING, packet=p, sourceSender=p["sender"])
    claim_id = receipt.destination_tx_id
    serialized = receipt.protocol_state.get("preparedDestinationTransaction")
    if not claim_id and serialized is not None:
        if c.outbound or not isinstance(serialized, str):
            raise BridgeError("Invalid Verulink prepared destination transaction")
        tx = prepared_transaction(serialized, receipt.id)
        validate_aleo_claim(c, p, tx)
        if aleo_transaction(c, tx["id"]) != "pending":
            claim_id = tx["id"]
    if claim_id:
        state = dict(verified.protocol_state)
        state.pop("preparedDestinationTransaction", None)
        result = destination_status(c, p, verified.replace(destination_tx_id=claim_id, protocol_state=state))
        if result is not None:
            return result
        # The claim reverted or was rejected, so it moved nothing. The packet's own state decides
        # what is left: still unconsumed means claim again, consumed means another claim delivered it.
        verified = verified.replace(id=receipt.source_tx_id, destination_tx_id=None,
                                    protocol_state={**state, "failedClaimTxId": claim_id})
    if consumed(c, p):
        return update(verified, Status.DELIVERY_PENDING, consumed=True,
                      destinationVerificationRequired="Packet consumed; a destination transaction id is required to distinguish public delivery from holding")
    signatures = verified_signatures(c, p, fetch_signatures(c, session, p))
    if signatures is None:
        return verified
    return verified.replace(status=Status.DESTINATION_ACTION_REQUIRED,
                            next_action={"kind": "verulink-claim", "chainId": c.destination.chain_id},
                            protocol_state={**verified.protocol_state, "signatures": signatures})
