"""Canonical public packet codecs and transaction checks for Verulink."""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from aleo.codegen.runtime import parse_plaintext

from .errors import BridgeError

U64_MAX = 2**64 - 1
ZERO_ALEO = "aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3ljyzc"
Packet = dict[str, str]


def integer(value: Any, name: str = "integer", maximum: int = U64_MAX) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise BridgeError(f"Invalid Verulink {name}")
    return value


def decimal(value: Any, maximum: int = U64_MAX) -> str:
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        value = int(value)
    return str(integer(value, "integer encoding", maximum))


def obj(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BridgeError("Invalid Verulink object")
    return dict(value)


def same(a: Any, b: Any) -> bool:
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    return a.lower() == b.lower() if a.startswith("0x") and b.startswith("0x") else a == b


def same_packet(a: Packet, b: Packet) -> bool:
    return a.keys() == b.keys() and all(same(a[key], b[key]) for key in a)


def address(value: Any) -> str:
    from web3 import Web3
    if not isinstance(value, str) or not re.fullmatch(r"0x[0-9a-fA-F]{40}", value):
        raise BridgeError("Invalid Verulink EVM address")
    return Web3.to_checksum_address(value)


def address_literal(value: str, padded: bool = False) -> str:
    raw = bytes.fromhex(address(value)[2:])
    return "[" + ", ".join(f"{b}u8" for b in ((b"\0" * 12 if padded else b"") + raw)) + "]"


def from_aleo_bytes(value: Any) -> str:
    if (not isinstance(value, list) or len(value) not in (20, 32)
            or any(type(x) is not int or not 0 <= x <= 255 for x in value)
            or any(value[:-20])):
        raise BridgeError("Invalid Verulink address bytes")
    return address("0x" + bytes(value[-20:]).hex())


def collector_packet(value: Any) -> Packet:
    p = obj(value)
    m = obj(p.get("message"))
    return {"version": decimal(p.get("version")), "sequence": decimal(p.get("sequence")),
            "height": decimal(p.get("height")), "sourceChain": decimal(p.get("sourceChain"), 2**128 - 1),
            "destinationChain": decimal(p.get("destinationChain"), 2**128 - 1),
            "sourceAddress": str(p.get("sourceAddress")), "destinationAddress": str(p.get("destinationAddress")),
            "sender": str(m.get("sender")), "recipient": str(m.get("receiver")),
            "token": str(m.get("denom")), "amount": decimal(m.get("amount"))}


def aleo_packet(value: Any) -> Packet:
    p = obj(value)
    source, dest, m = obj(p.get("source")), obj(p.get("destination")), obj(p.get("message"))
    return {"version": decimal(p.get("version")), "sequence": decimal(p.get("sequence")), "height": decimal(p.get("height")),
            "sourceChain": decimal(source.get("chain_id"), 2**128 - 1), "destinationChain": decimal(dest.get("chain_id"), 2**128 - 1),
            "sourceAddress": str(source.get("addr")), "destinationAddress": from_aleo_bytes(dest.get("addr")),
            "sender": str(m.get("sender_address")), "recipient": from_aleo_bytes(m.get("receiver_address")),
            "token": from_aleo_bytes(m.get("dest_token_address")), "amount": decimal(m.get("amount"))}


def evm_packet(value: Any) -> Packet:
    p = obj(value)
    src, dst, m = obj(p.get("sourceTokenService")), obj(p.get("destTokenService")), obj(p.get("message"))
    return {"version": decimal(p.get("version")), "sequence": decimal(p.get("sequence")), "height": decimal(p.get("height")),
            "sourceChain": decimal(src.get("chainId"), 2**128 - 1), "destinationChain": decimal(dst.get("chainId"), 2**128 - 1),
            "sourceAddress": str(src.get("addr")), "destinationAddress": str(dst.get("addr")),
            "sender": str(m.get("senderAddress")), "recipient": str(m.get("receiverAddress")),
            "token": str(m.get("destTokenAddress")), "amount": decimal(m.get("amount"))}


def incoming(p: Packet) -> tuple:
    return (int(p["version"]), int(p["sequence"]), (int(p["sourceChain"]), p["sourceAddress"]),
            (int(p["destinationChain"]), address(p["destinationAddress"])),
            (p["sender"], address(p["token"]), int(p["amount"]), address(p["recipient"])), int(p["height"]))


def packet_hash(p: Packet) -> bytes:
    from eth_abi.packed import encode_packed
    from eth_utils.crypto import keccak
    return keccak(encode_packed(
        ["uint256", "uint256", "uint256", "string", "uint256", "address", "string", "address", "uint256", "address", "uint256"],
        [int(p["version"]), int(p["sequence"]), int(p["sourceChain"]), p["sourceAddress"], int(p["destinationChain"]),
         p["destinationAddress"], p["sender"], p["token"], int(p["amount"]), p["recipient"], int(p["height"])]))


def aleo_packet_literal(p: Packet) -> str:
    return (f"{{ version: {p['version']}u8, sequence: {p['sequence']}u64, "
            f"source: {{ chain_id: {p['sourceChain']}u128, addr: {address_literal(p['sourceAddress'], True)} }}, "
            f"destination: {{ chain_id: {p['destinationChain']}u128, addr: {p['destinationAddress']} }}, "
            f"message: {{ sender_address: {address_literal(p['sender'], True)}, dest_token_id: {p['token']}, "
            f"amount: {p['amount']}u128, receiver_address: {p['recipient']} }}, height: {p['height']}u64 }}")


def parse(value: Any) -> Any:
    try:
        return parse_plaintext(value)
    except (TypeError, ValueError) as exc:
        raise BridgeError("Invalid Verulink Aleo plaintext") from exc
