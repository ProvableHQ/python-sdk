"""Pure wire encoders shared by every route family (port of veil utils/xreserve.ts, hyperlane.ts,
hyperlaneDelivery.ts). Nothing here touches the network; only ``xreserve_hook_data`` (private mode)
and ``aleo_program_address`` load the ``aleo.<network>`` bindings.

Spacing rules (invariant 8): ``u8_array_literal`` joins with ``","`` and NO space, every other
struct/array literal joins with ``", "``.
"""
from __future__ import annotations

import re
from typing import Any

from ._base58 import b58decode
from ._keccak import keccak256
from .errors import AttestationError, ConfigurationError, InvalidAmountError, InvalidRecipientError

BECH32_ALPHABET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
BECH32M_CONST = 0x2BC830A3
_BECH32_GENERATORS = (0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3)
_HRP = "aleo"
_EVM_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
_SCALAR_RE = re.compile(r"^(0|[1-9][0-9]*)scalar$")
NETWORKS = ("mainnet", "testnet")
MINT_MODES = ("public", "record", "private")
HOOK_DATA_BYTES = 65
XRESERVE_PAYLOAD_BYTES = 305
_PAYLOAD_HEADER = bytes.fromhex("5a2e0acd00000001")
_HOOK_LENGTH_FIELD = bytes.fromhex("00000041")


# ── Generic byte helpers ──────────────────────────────────────────────────────

def hex_to_bytes(value: "str | bytes | bytearray | memoryview", expected_len: "int | None" = None) -> bytes:
    """``0x``-prefixed (or bare) hex, or bytes, to ``bytes``; optionally assert the width."""
    if isinstance(value, str):
        text = value[2:] if value[:2] in ("0x", "0X") else value
        try:
            data = bytes.fromhex(text)
        except ValueError as exc:
            raise InvalidRecipientError(f"Not hexadecimal: {value!r}") from exc
    else:
        data = bytes(value)
    if expected_len is not None and len(data) != expected_len:
        raise InvalidRecipientError(f"Expected {expected_len} bytes, got {len(data)}")
    return data


def to_hex(data: "bytes | bytearray | memoryview") -> str:
    return "0x" + bytes(data).hex()


def network_module(network: str) -> Any:
    """``aleo.mainnet`` / ``aleo.testnet`` for *network*; ``ConfigurationError`` otherwise."""
    if network not in NETWORKS:
        raise ConfigurationError(f"network must be one of {NETWORKS}, got {network!r}")
    import aleo  # noqa: WPS433 — the bindings are a runtime dependency, imported lazily
    return getattr(aleo, network)


def field_bytes_le(value: Any) -> bytes:
    """``Field/Address.to_bytes_le()`` returns ``list[int]`` at runtime; normalise to bytes."""
    return bytes(value.to_bytes_le())


def validate_scalar(secret_nonce: str) -> str:
    if not isinstance(secret_nonce, str) or not _SCALAR_RE.match(secret_nonce):
        raise ConfigurationError(
            f"secret_nonce must be a non-negative Aleo scalar literal such as 0scalar, got {secret_nonce!r}")
    return secret_nonce


# ── Aleo bech32m ──────────────────────────────────────────────────────────────

def _bech32_polymod(values: list[int]) -> int:
    chk = 1
    for v in values:
        top = chk >> 25
        chk = ((chk & 0x1FFFFFF) << 5) ^ v
        for i in range(5):
            if (top >> i) & 1:
                chk ^= _BECH32_GENERATORS[i]
    return chk


def _hrp_expand() -> list[int]:
    return [ord(c) >> 5 for c in _HRP] + [0] + [ord(c) & 31 for c in _HRP]


def aleo_address_to_bytes32(address: str) -> bytes:
    """Decode a checksummed ``aleo1…`` bech32m address into its 32 payload bytes (xReserve/Hyperlane form)."""
    try:
        if not isinstance(address, str) or not address.startswith("aleo1") or len(address) != 63:
            raise ValueError("invalid prefix or length")
        words = [BECH32_ALPHABET.index(c) for c in address[5:]]  # ValueError on a foreign character
        if _bech32_polymod(_hrp_expand() + words) != BECH32M_CONST:
            raise ValueError("invalid checksum")
        acc = bits = 0
        out = bytearray()
        for word in words[:-6]:
            acc = (acc << 5) | word
            bits += 5
            while bits >= 8:
                bits -= 8
                out.append((acc >> bits) & 0xFF)
        if bits >= 5 or ((acc << (8 - bits)) & 0xFF) != 0:
            raise ValueError("invalid padding")
        if len(out) != 32:
            raise ValueError("invalid payload")
        return bytes(out)
    except ValueError as exc:
        raise InvalidRecipientError(f"Invalid Aleo recipient address: {address}") from exc


def bytes32_to_aleo_address(data: bytes) -> str:
    """Inverse of :func:`aleo_address_to_bytes32` — re-encode 32 bytes as a checksummed ``aleo1…`` address."""
    raw = bytes(data)
    if len(raw) != 32:
        raise InvalidRecipientError(f"Invalid 32-byte Aleo recipient: {to_hex(raw)}")
    acc = bits = 0
    words: list[int] = []
    for byte in raw:
        acc = (acc << 8) | byte
        bits += 8
        while bits >= 5:
            bits -= 5
            words.append((acc >> bits) & 31)
    if bits:
        words.append((acc << (5 - bits)) & 31)
    checksum = _bech32_polymod(_hrp_expand() + words + [0] * 6) ^ BECH32M_CONST
    words += [(checksum >> (5 * (5 - i))) & 31 for i in range(6)]
    return _HRP + "1" + "".join(BECH32_ALPHABET[w] for w in words)


# ── EVM addresses ─────────────────────────────────────────────────────────────

def to_checksum_address(address: str) -> str:
    """EIP-55 checksum form of a 20-byte hex address."""
    body = address[2:].lower()
    digest = keccak256(body.encode()).hex()
    return "0x" + "".join(c.upper() if int(digest[i], 16) >= 8 else c for i, c in enumerate(body))


def is_evm_address(value: Any) -> bool:
    """20-byte hex; mixed case must be a valid EIP-55 checksum (viem ``isAddress`` semantics)."""
    if not isinstance(value, str) or not _EVM_ADDRESS_RE.match(value):
        return False
    body = value[2:]
    if body == body.lower() or body == body.upper():
        return True
    return to_checksum_address(value) == value


def evm_address_to_bytes32(address: str) -> bytes:
    """Left-pad a checksum-validated EVM address to 32 bytes (xReserve burn recipient / Hyperlane bytes32)."""
    if not is_evm_address(address):
        raise InvalidRecipientError(f"Invalid Ethereum recipient address: {address}")
    return bytes(12) + bytes.fromhex(address[2:])


# ── Hyperlane recipient limbs and Aleo literals ───────────────────────────────

def bytes32_to_u128_limbs(data: bytes) -> tuple[int, int]:
    """Two little-endian u128 limbs over ``bytes[0:16]`` and ``bytes[16:32]``."""
    raw = bytes(data)
    if len(raw) != 32:
        raise InvalidRecipientError(f"Hyperlane recipient limbs need exactly 32 bytes, got {len(raw)}")
    return int.from_bytes(raw[:16], "little"), int.from_bytes(raw[16:], "little")


def evm_address_to_hyperlane_recipient(address: str) -> tuple[int, int]:
    if not is_evm_address(address):
        raise InvalidRecipientError(f"Invalid Ethereum Hyperlane recipient: {address}")
    return bytes32_to_u128_limbs(evm_address_to_bytes32(address))


def solana_address_to_hyperlane_recipient(address: str) -> tuple[int, int]:
    try:
        raw = b58decode(address)
    except ValueError as exc:
        raise InvalidRecipientError(f"Invalid Solana Hyperlane recipient: {address}") from exc
    if len(raw) != 32:
        raise InvalidRecipientError(f"Invalid Solana Hyperlane recipient: {address}")
    return bytes32_to_u128_limbs(raw)


def u128_pair_literal(limbs: tuple[int, int]) -> str:
    """``[{lo}u128, {hi}u128]`` — the Warp Route recipient input (space after the comma)."""
    lo, hi = limbs
    return f"[{lo}u128, {hi}u128]"


def u8_array_literal(data: bytes) -> str:
    """``[0u8,255u8]`` — veil ``xReserveHexToAleoBytes``: NO space after the comma."""
    return "[" + ",".join(f"{b}u8" for b in bytes(data)) + "]"


def hyperlane_delivery_key(message_id: bytes) -> str:
    """``hyp_mailbox.aleo/deliveries`` key: ``{ id: [{lo}u128, {hi}u128] }``."""
    lo, hi = bytes32_to_u128_limbs(hex_to_bytes(message_id, 32))
    return f"{{ id: [{lo}u128, {hi}u128] }}"


# ── Circle xReserve ───────────────────────────────────────────────────────────

def _uint_be(value: int, width: int) -> bytes:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value >= 1 << (8 * width):
        raise InvalidAmountError(f"Unsigned value does not fit in {width} bytes: {value!r}")
    return value.to_bytes(width, "big")


def xreserve_deposit_nonce(source_domain: int, tx_hash: "bytes | str", log_index: int) -> bytes:
    """Circle's deposit nonce: ``keccak(abi.encode(uint32 domain) ‖ txHash ‖ abi.encode(uint256 logIndex))``."""
    _uint_be(source_domain, 4)  # bound to uint32, consistent with the payload's remote_domain check below
    return keccak256(_uint_be(source_domain, 32) + hex_to_bytes(tx_hash, 32) + _uint_be(log_index, 32))


def xreserve_deposit_payload(*, amount: int, remote_domain: int, remote_token: bytes, remote_recipient: bytes,
                             local_token: str, depositor: str, max_fee: int, nonce: bytes,
                             hook_data: bytes) -> bytes:
    """The canonical 305-byte xReserve v2 deposit payload Circle signs.

    header[0..8) amount[8..40) domain[40..44) remoteToken[44..76) recipient[76..108)
    localToken[108..140) depositor[140..172) maxFee[172..204) nonce[204..236) hookLen[236..240) hook[240..305)
    """
    for name, value, width in (("remote_token", remote_token, 32), ("remote_recipient", remote_recipient, 32),
                               ("nonce", nonce, 32), ("hook_data", hook_data, HOOK_DATA_BYTES)):
        if len(bytes(value)) != width:
            raise InvalidRecipientError(f"{name} must contain {width} bytes")
    out = bytearray(XRESERVE_PAYLOAD_BYTES)
    out[0:8] = _PAYLOAD_HEADER
    out[8:40] = _uint_be(amount, 32)
    out[40:44] = _uint_be(remote_domain, 4)
    out[44:76] = bytes(remote_token)
    out[76:108] = bytes(remote_recipient)
    out[108:140] = evm_address_to_bytes32(local_token)
    out[140:172] = evm_address_to_bytes32(depositor)
    out[172:204] = _uint_be(max_fee, 32)
    out[204:236] = bytes(nonce)
    out[236:240] = _HOOK_LENGTH_FIELD
    out[240:305] = bytes(hook_data)
    return bytes(out)


def xreserve_message_hash(payload: bytes) -> bytes:
    """Circle's attestation lookup key: ``keccak256(payload)``."""
    return keccak256(bytes(payload))


def xreserve_nonce_from_payload(payload: bytes) -> bytes:
    """Deposit nonce (bytes 204..236) from a canonical payload; validates header, width and hook length."""
    raw = bytes(payload)
    if len(raw) != XRESERVE_PAYLOAD_BYTES or raw[0:8] != _PAYLOAD_HEADER or raw[236:240] != _HOOK_LENGTH_FIELD:
        raise AttestationError("xReserve payload has an invalid deposit layout")
    return raw[204:236]


def xreserve_hook_data(mode: str, recipient: str, network: str, secret_nonce: str = "0scalar") -> bytes:
    """65-byte xReserve hook: byte 0 selects the mint transition (0 public, 1 record, 2 private);
    private mode carries ``BHP256.commit(bits(recipient), secret_nonce)`` in bytes 1..33."""
    if mode not in MINT_MODES:
        raise ConfigurationError(f"Unsupported mint mode {mode!r}; expected one of {MINT_MODES}")
    aleo_address_to_bytes32(recipient)  # InvalidRecipientError before any FFI work
    out = bytearray(HOOK_DATA_BYTES)
    out[0] = MINT_MODES.index(mode)
    if mode == "private":
        validate_scalar(secret_nonce)
        net = network_module(network)
        bits = net.Plaintext.from_string(recipient).to_bits_le()
        commitment = field_bytes_le(net.BHP256().commit(bits, net.Scalar.from_string(secret_nonce)))
        if len(commitment) != 32:
            raise ConfigurationError("Private mint commitment must contain 32 bytes")
        out[1:33] = commitment
    return bytes(out)


def aleo_program_address(program_id: str, network: str) -> str:
    """The ``aleo1…`` account owned by a deployed program (private xReserve deposits target the wrapper's)."""
    return str(network_module(network).Address.from_program_id(program_id))


__all__ = [
    "BECH32_ALPHABET", "HOOK_DATA_BYTES", "MINT_MODES", "NETWORKS", "XRESERVE_PAYLOAD_BYTES",
    "aleo_address_to_bytes32", "aleo_program_address", "bytes32_to_aleo_address", "bytes32_to_u128_limbs",
    "evm_address_to_bytes32", "evm_address_to_hyperlane_recipient", "field_bytes_le", "hex_to_bytes",
    "hyperlane_delivery_key", "is_evm_address", "network_module", "solana_address_to_hyperlane_recipient",
    "to_checksum_address", "to_hex", "u128_pair_literal", "u8_array_literal", "validate_scalar",
    "xreserve_deposit_nonce", "xreserve_deposit_payload", "xreserve_hook_data", "xreserve_message_hash",
    "xreserve_nonce_from_payload",
]
