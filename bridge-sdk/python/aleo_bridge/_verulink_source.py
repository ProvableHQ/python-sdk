"""Reconstruct Verulink packets from immutable, confirmed source evidence."""
from __future__ import annotations

import json
import re
from typing import Any

from ._verulink_context import Context
from ._verulink_wire import (Packet, aleo_packet, decimal, evm_packet, from_aleo_bytes,
                             integer, obj, parse, same)
from .errors import BridgeError
from .types import Plan
from .units import parse_decimal_amount


def tx_hash(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"0x[0-9a-fA-F]{64}", value):
        raise BridgeError("Invalid Verulink EVM transaction id")
    return value


def hex_value(value: Any) -> str:
    from web3 import Web3
    return value if isinstance(value, str) else Web3.to_hex(value)


def transition(tx: Any, program: str, name: str) -> dict[str, Any]:
    calls = [t for t in obj(obj(tx).get("execution")).get("transitions", [])
             if obj(t).get("program") == program and t.get("function") == name]
    if len(calls) != 1:
        raise BridgeError(f"Verulink transaction requires exactly one {program}/{name} transition")
    return obj(calls[0])


def inputs(call: dict[str, Any]) -> list[Any]:
    values = call.get("inputs")
    if not isinstance(values, list):
        raise BridgeError("Verulink transition has missing public inputs")
    result = []
    for raw in values:
        item = obj(raw)
        if item.get("type") != "public" or not isinstance(item.get("value"), str):
            raise BridgeError("Verulink transition has missing public inputs")
        result.append(parse(item["value"]))
    return result


def _root(tx: Any, program: str, name: str) -> None:
    transitions = obj(obj(tx).get("execution")).get("transitions")
    if (not isinstance(transitions, list) or not transitions or obj(transitions[-1]).get("program") != program
            or transitions[-1].get("function") != name):
        raise BridgeError("Verulink root call does not match the transfer")


def prepared_transaction(serialized: Any, tx_id: Any) -> dict[str, Any]:
    if not isinstance(tx_id, str) or not tx_id or not isinstance(serialized, str):
        raise BridgeError("Invalid Verulink prepared transaction")
    try:
        tx = obj(json.loads(serialized))
    except (ValueError, TypeError) as exc:
        raise BridgeError("Invalid Verulink prepared transaction") from exc
    if tx.get("id") != tx_id:
        raise BridgeError("Verulink prepared transaction id mismatch")
    return tx


def aleo_transaction(c: Context, tx_id: str) -> dict[str, Any] | str:
    from aleo.facade.errors import TransactionNotFound
    try:
        confirmed = obj(c.aleo.network.get_confirmed_transaction(tx_id))
    except TransactionNotFound:
        return "pending"
    if confirmed.get("status") == "rejected":
        return "rejected"
    if confirmed.get("status") != "accepted":
        return "pending"
    tx = obj(confirmed.get("transaction"))
    if tx.get("id") != tx_id:
        raise BridgeError("Verulink Aleo transaction identity mismatch")
    return tx


def validate_aleo_source(c: Context, plan: Plan, tx: Any) -> tuple[int, str]:
    _root(tx, c.aleo_service, "token_send_public")
    call = inputs(transition(tx, c.aleo_service, "token_send_public"))
    if (len(call) != 6 or not same(from_aleo_bytes(call[0]), plan.recipient)
            or type(call[1]) is not int or call[1] != plan.amount_atomic or call[2] != int(c.evm_domain)
            or from_aleo_bytes(call[3]) != c.evm_service or from_aleo_bytes(call[4]) != c.evm_token):
        raise BridgeError("Verulink Aleo source inputs do not match the transfer")
    amount = integer(call[1], "submitted amount") - integer(call[5], "submitted fee")
    published = inputs(transition(tx, c.aleo_bridge, "publish"))
    if (len(published) != 7 or published[0] != 1 or published[1] != int(c.evm_domain)
            or from_aleo_bytes(published[2]) != c.evm_service or from_aleo_bytes(published[3]) != c.evm_token
            or not same(from_aleo_bytes(published[5]), plan.recipient) or published[6] != amount
            or not isinstance(published[4], str) or (plan.sender is not None and published[4] != plan.sender)):
        raise BridgeError("Verulink published packet does not match the source inputs")
    if amount <= 0:
        raise BridgeError("Verulink source net amount must be positive")
    if plan.amount_out is not None and amount < parse_decimal_amount(plan.amount_out, 6):
        raise BridgeError("Verulink prepared source fee exceeds the quoted fee")
    return amount, published[4]


def validate_aleo_claim(c: Context, packet: Packet, tx: Any) -> dict[str, Any]:
    _root(tx, c.aleo_service, "token_receive_public")
    claim = transition(tx, c.aleo_service, "token_receive_public")
    v = inputs(claim)
    if (len(v) != 9 or not same(from_aleo_bytes(v[0]), packet["sender"]) or v[1] != packet["recipient"]
            or v[2] != int(packet["amount"]) or v[3] != int(packet["sequence"]) or v[4] != int(packet["height"])
            or v[7] != int(packet["sourceChain"]) or not same(from_aleo_bytes(v[8]), packet["sourceAddress"])):
        raise BridgeError("Verulink Aleo claim does not match the source packet")
    return claim


def validate_packet(c: Context, plan: Plan, p: Packet) -> None:
    if (p["version"] != "1" or int(p["sequence"]) < 1
            or p["sourceChain"] != (c.aleo_domain if c.outbound else c.evm_domain)
            or p["destinationChain"] != (c.evm_domain if c.outbound else c.aleo_domain)
            or not same(p["sourceAddress"], c.aleo_service_address if c.outbound else c.evm_service)
            or not same(p["destinationAddress"], c.evm_service if c.outbound else c.aleo_service_address)
            or not same(p["token"], c.evm_token if c.outbound else c.aleo_token_id)
            or not same(p["recipient"], plan.recipient) or (plan.sender is not None and not same(p["sender"], plan.sender))
            or not 0 < int(p["amount"]) <= plan.amount_atomic):
        raise BridgeError("Verulink source packet does not match the transfer")


def events(c: Context, result: Any, to: str, name: str) -> list[dict[str, Any]]:
    from web3.exceptions import MismatchedABI
    decoder = getattr(c.contract(to).events, name)()
    decoded = []
    for log in result.get("logs", []):
        if not same(log.get("address"), to):
            continue
        try:
            decoded.append(dict(decoder.process_log(log)["args"]))
        except (MismatchedABI, ValueError, KeyError, TypeError):
            continue
    return decoded


def get_transaction(c: Context, tx_id: str) -> Any:
    from web3.exceptions import TransactionNotFound
    try:
        return c.evm.w3.eth.get_transaction(tx_id)
    except TransactionNotFound:
        return None


def source_packet(c: Context, plan: Plan, tx_id: str) -> Packet | str:
    c.check_chain()
    if not c.outbound:
        tx_hash(tx_id)
        result = c.evm.get_receipt(tx_id)
        if result is None:
            return "pending"
        if not same(hex_value(result.get("transactionHash")), tx_id) or result.get("blockNumber") is None:
            raise BridgeError("Verulink source receipt identity mismatch")
        if result.get("status") == 0:
            return "rejected"
        if result.get("status") != 1:
            raise BridgeError("Invalid Verulink source receipt status")
        if c.evm.w3.eth.block_number < result["blockNumber"] + c.confirmations - 1:
            return "pending"
        tx = get_transaction(c, tx_id)
        if (tx is None or not same(hex_value(tx.get("hash")), tx_id) or not same(tx.get("to"), c.evm_service)
                or tx.get("blockNumber") != result["blockNumber"]):
            raise BridgeError("Verulink source transaction does not match its receipt")
        fn, args = c.contract(c.evm_service).decode_function_input(tx["input"])
        if (fn.fn_name != "publicTokenSend" or not same(args["tokenAddress"], c.evm_token)
                or args["amount"] != plan.amount_atomic or args["receiver"] != plan.recipient
                or (plan.sender is not None and not same(tx.get("from"), plan.sender))):
            raise BridgeError("Verulink source calldata does not match the transfer")
        packets = [evm_packet(e["packet"]) for e in events(c, result, c.evm_bridge, "PacketDispatched")]
        if len(packets) != 1:
            raise BridgeError("Verulink source receipt is missing its unique PacketDispatched event")
        p = packets[0]
        validate_packet(c, plan, p)
        if not same(p["sender"], tx["from"]) or p["height"] != str(result["blockNumber"]):
            raise BridgeError("Verulink packet does not match its source transaction")
        return p
    tx = aleo_transaction(c, tx_id)
    if isinstance(tx, str):
        return tx
    amount, sender = validate_aleo_source(c, plan, tx)
    # The facade has no transaction-to-block lookup; reuse its configured node transport.
    block_hash = c.aleo._client._get(f"/find/blockHash/{tx_id}", "findBlockHash")
    block = obj(c.aleo.network.get_block_by_hash(block_hash))
    height = decimal(obj(obj(block.get("header")).get("metadata")).get("height"))
    next_sequence = integer(c.mapping(c.aleo_bridge, "sequences", c.evm_domain + "u128"), "next sequence")

    def read(sequence: int) -> Packet:
        p = aleo_packet(c.mapping(c.aleo_bridge, "out_packets", f"{{ chain_id: {c.evm_domain}u128, sequence: {sequence}u64 }}"))
        if p["sequence"] != str(sequence):
            raise BridgeError("Verulink packet sequence does not match its mapping key")
        return p

    low, high = 1, next_sequence
    while low < high:
        middle = (low + high) // 2
        if int(read(middle)["height"]) < int(height):
            low = middle + 1
        else:
            high = middle
    matches = []
    for count, sequence in enumerate(range(low, next_sequence)):
        if count >= 128:
            raise BridgeError("Verulink source block packet scan exceeds its bound")
        p = read(sequence)
        if p["height"] != height:
            break
        if (same(p["sender"], sender) and same(p["recipient"], plan.recipient) and p["amount"] == str(amount)
                and same(p["sourceAddress"], c.aleo_service_address) and same(p["destinationAddress"], c.evm_service)
                and same(p["token"], c.evm_token)):
            matches.append(p)
    if len(matches) != 1:
        raise BridgeError("Ambiguous Verulink packets in the Aleo source block" if matches else
                          "Verulink source packet does not match the confirmed transaction")
    validate_packet(c, plan, matches[0])
    return matches[0]
