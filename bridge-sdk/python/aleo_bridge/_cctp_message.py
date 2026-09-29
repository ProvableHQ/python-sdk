"""CCTP V2 wire evidence. Offsets and hook frames pinned to Veil PR #148.

_source: packages/bridge/src/protocols/cctp/evm.ts @
3c3b457bd5f63620657321893a2487e489750d24.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .errors import AttestationError, ConfigurationError

FORWARD_HOOK = b"cctp-forward".ljust(24, b"\0") + bytes(8)
LEGACY_FORWARD_HOOK = b"cctp-forward".ljust(24, b"\0") + (1).to_bytes(4, "big") + bytes(4)


def address_bytes(address: str) -> bytes:
    if not isinstance(address, str) or not re.fullmatch(r"0x[0-9a-fA-F]{40}", address):
        raise ConfigurationError("CCTP requires a 20-byte EVM address")
    value = bytes.fromhex(address[2:])
    if value == bytes(20):
        raise ConfigurationError("CCTP addresses must not be zero")
    return value.rjust(32, b"\0")


@dataclass(frozen=True)
class CctpMessage:
    raw: bytes
    source_domain: int
    destination_domain: int
    nonce: bytes
    messenger: bytes
    destination_messenger: bytes
    destination_caller: bytes
    min_finality: int
    finality: int
    token: bytes
    recipient: bytes
    amount: int
    sender: bytes
    max_fee: int
    fee: int
    expiry: int
    hook: bytes


def decode_message(raw: bytes) -> CctpMessage:
    if not isinstance(raw, bytes) or len(raw) < 376:
        raise AttestationError("CCTP message is truncated")
    def uint(start: int, size: int) -> int:
        return int.from_bytes(raw[start:start + size], "big")
    if uint(0, 4) != 1 or uint(148, 4) != 1:
        raise AttestationError("Unsupported CCTP message version")
    return CctpMessage(raw, uint(4, 4), uint(8, 4), raw[12:44], raw[44:76], raw[76:108], raw[108:140],
                       uint(140, 4), uint(144, 4), raw[152:184], raw[184:216], uint(216, 32),
                       raw[248:280], uint(280, 32), uint(312, 32), uint(344, 32), raw[376:])


def immutable_message(raw: bytes) -> bytes:
    decode_message(raw)
    return raw[0:12] + raw[44:144] + raw[148:312] + raw[376:]


def validate_message(raw: bytes, *, source_domain: int, destination_domain: int, messenger: str,
                     source_token: str, sender: str, recipient: str, amount_atomic: int,
                     max_fee_atomic: int, finality: int, forwarding: bool) -> CctpMessage:
    m = decode_message(raw)
    if (m.source_domain != source_domain or m.destination_domain != destination_domain
            or m.messenger != address_bytes(messenger) or m.destination_messenger != address_bytes(messenger)
            or m.destination_caller != bytes(32) or m.token != address_bytes(source_token)
            or m.sender != address_bytes(sender) or m.recipient != address_bytes(recipient)
            or m.amount != amount_atomic or m.max_fee != max_fee_atomic or m.max_fee >= m.amount
            or m.min_finality != finality
            or m.hook not in ((FORWARD_HOOK, LEGACY_FORWARD_HOOK) if forwarding else (b"",))):
        raise AttestationError("CCTP source message does not match the transfer intent")
    return m
