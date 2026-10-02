"""Shared machinery of the remote-wallet signers (:mod:`aleo_bridge.privy`, :mod:`aleo_bridge.dynamic`).

A remote wallet signs on the provider's side and never exposes a key. The bridge still builds every
transaction, broadcasts it through the caller's RPC, checkpoints it and recovers it; the provider
contributes exactly one signature per transaction. Everything a provider returns is verified here
before it can reach the wire: an EVM transaction must recover to the configured sender, a Solana
signature must verify against the configured public key over the exact message that was built.

Nothing in this module retries. A provider SDK may retry its own HTTP calls; a signature request
that fails or times out leaves the transaction unsigned and the call usable.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from dataclasses import dataclass
from typing import Any, Awaitable, Mapping, TypeVar

from .errors import BridgeError, ConfigurationError, MissingExtraError

T = TypeVar("T")


# --- EVM -----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RemoteSignedTransaction:
    """The two things :meth:`aleo_bridge.Ethereum.send_transaction` reads from a signed transaction:
    the raw bytes it broadcasts and their hash, fixed by the signature before the broadcast."""

    raw_transaction: bytes
    hash: bytes


def _eth_utils() -> Any:
    try:
        import eth_utils
    except ImportError as exc:  # pragma: no cover - ships with web3
        raise MissingExtraError("evm", "Ethereum signing") from exc
    return eth_utils


def checksum_address(address: Any, *, what: str) -> str:
    """*address* as a checksummed ``0x`` string, or ``ConfigurationError`` naming *what* was wrong."""
    utils = _eth_utils()
    if not isinstance(address, str) or not utils.is_address(address):
        raise ConfigurationError(f"{what} must be a 0x-prefixed Ethereum address")
    return str(utils.to_checksum_address(address))


def unsigned_evm_fields(tx: Mapping[str, Any], *, provider: str) -> dict[str, Any]:
    """The fields a remote signer needs from a transaction :class:`aleo_bridge.Ethereum` prepared.

    ``Ethereum.send_transaction`` fills ``nonce``/``gas``/the fee fields before it calls the signer;
    a caller using a signer directly must do the same, so a missing field is a configuration error
    here rather than a provider rejection later.
    """
    missing = [key for key in ("to", "chainId", "nonce", "gas") if key not in tx]
    if missing:
        raise ConfigurationError(f"{provider} signing needs {', '.join(missing)} in the transaction; "
                                 "let Ethereum.send_transaction fill them or set them yourself")
    if "gasPrice" not in tx and "maxFeePerGas" not in tx:
        raise ConfigurationError(f"{provider} signing needs gasPrice or maxFeePerGas/maxPriorityFeePerGas")
    data = tx.get("data", "0x")
    if isinstance(data, (bytes, bytearray, memoryview)):
        data = "0x" + bytes(data).hex()
    elif not isinstance(data, str):
        raise ConfigurationError("transaction data must be a 0x hex string or bytes")
    elif not data.startswith("0x"):
        data = "0x" + data
    fields: dict[str, Any] = {
        "to": checksum_address(tx["to"], what="transaction to"),
        "value": int(tx.get("value", 0)),
        "chainId": int(tx["chainId"]),
        "nonce": int(tx["nonce"]),
        "gas": int(tx["gas"]),
        "data": data,
    }
    if "gasPrice" in tx:
        fields["gasPrice"] = int(tx["gasPrice"])
    else:
        fields["maxFeePerGas"] = int(tx["maxFeePerGas"])
        fields["maxPriorityFeePerGas"] = int(tx.get("maxPriorityFeePerGas", 0))
    return fields


def verified_evm_transaction(raw: bytes, *, sender: str, provider: str) -> RemoteSignedTransaction:
    """Recover the signer of *raw* and refuse it unless it is *sender* — the only proof that the
    provider signed with the wallet this connection was configured for."""
    from eth_account import Account

    raw = bytes(raw)
    try:
        recovered = Account.recover_transaction(raw)
    except Exception as exc:  # noqa: BLE001 — any decoding/recovery failure is a bad signature
        raise BridgeError(f"{provider} returned a malformed signed Ethereum transaction") from exc
    if checksum_address(recovered, what="recovered signer") != sender:
        raise BridgeError(f"{provider} signature does not recover to the configured wallet {sender}; "
                          f"it recovers to {recovered} — refusing to broadcast")
    return RemoteSignedTransaction(raw_transaction=raw, hash=bytes(_eth_utils().keccak(raw)))


def legacy_transaction_from_signature(fields: Mapping[str, Any], signature: bytes, *, sender: str,
                                      provider: str) -> RemoteSignedTransaction:
    """Assemble a legacy (EIP-155) transaction from its unsigned *fields* and a raw 65-byte
    ``r ‖ s ‖ v`` signature, then verify it recovers to *sender*.

    A provider may report ``v`` as the recovery id (0/1), as 27/28, or already EIP-155 encoded;
    anything below 35 is folded into ``chain_id * 2 + 35 + parity``. The recovery check afterwards
    is what actually proves the choice right.
    """
    from eth_account._utils.legacy_transactions import encode_transaction, serializable_unsigned_transaction_from_dict

    if len(signature) != 65:
        raise BridgeError(f"{provider} returned a {len(signature)}-byte signature; expected 65 bytes (r, s, v)")
    r, s, v = int.from_bytes(signature[:32], "big"), int.from_bytes(signature[32:64], "big"), signature[64]
    chain_id = int(fields["chainId"])
    if v < 35:
        parity = v - 27 if v >= 27 else v
        if parity not in (0, 1):
            raise BridgeError(f"{provider} returned an invalid signature recovery id {v}")
        v = chain_id * 2 + 35 + parity
    unsigned = serializable_unsigned_transaction_from_dict({
        "to": fields["to"], "value": int(fields["value"]), "gas": int(fields["gas"]),
        "gasPrice": int(fields["gasPrice"]), "nonce": int(fields["nonce"]), "chainId": chain_id,
        "data": fields["data"],
    })
    raw = encode_transaction(unsigned, vrs=(v, r, s))
    return verified_evm_transaction(bytes(raw), sender=sender, provider=provider)


# --- Solana --------------------------------------------------------------------------------------

def solana_pubkey(address: Any, *, what: str) -> Any:
    """*address* as a solders ``Pubkey`` (32 bytes, base58), or ``ConfigurationError`` naming *what*."""
    from .sol import _libs

    libs = _libs()
    if not isinstance(address, str) or not address:
        raise ConfigurationError(f"{what} must be a base58 Solana address")
    try:
        return libs.Pubkey.from_string(address)
    except Exception as exc:  # noqa: BLE001 — solders raises its own parse error types
        raise ConfigurationError(f"{what} must be a base58 Solana address") from exc


def signer_index(message_bytes: bytes, pubkey: Any, *, provider: str) -> tuple[Any, int]:
    """Decode the versioned *message_bytes* the connection asked us to sign and locate *pubkey*
    among its required signers. Returns ``(message, index)``."""
    from solders.message import from_bytes_versioned

    try:
        message = from_bytes_versioned(bytes(message_bytes))
    except Exception as exc:  # noqa: BLE001
        raise BridgeError(f"{provider} signer received bytes that are not a Solana message") from exc
    keys = list(message.account_keys)
    required = int(message.header.num_required_signatures)
    if pubkey not in keys[:required]:
        raise BridgeError(f"{provider} wallet {pubkey} is not a required signer of this Solana transaction; "
                          "the bridge source wallet pays the fee and must be the connected signer")
    return message, keys.index(pubkey)


def verified_solana_signature(signature: Any, *, pubkey: Any, message_bytes: bytes, provider: str) -> Any:
    """*signature* (64 raw bytes or a solders ``Signature``) verified against *pubkey* over
    *message_bytes*; anything else is refused before it can reach the wire."""
    from .sol import _libs

    libs = _libs()
    if isinstance(signature, (bytes, bytearray, memoryview)):
        if len(signature) != 64:
            raise BridgeError(f"{provider} returned a {len(signature)}-byte Solana signature; expected 64 bytes")
        signature = libs.Signature.from_bytes(bytes(signature))
    if not isinstance(signature, libs.Signature):
        raise BridgeError(f"{provider} returned an invalid Solana signature")
    if signature == libs.Signature.default() or not signature.verify(pubkey, bytes(message_bytes)):
        raise BridgeError(f"{provider} signature does not verify against wallet {pubkey} over the transaction "
                          "message — refusing to broadcast")
    return signature


# --- async providers -----------------------------------------------------------------------------

class _SigningLoop:
    """One private event loop on a daemon thread, shared by every async provider adapter in the
    process. The bridge is synchronous; a provider SDK that is ``async``-only runs here, and its HTTP
    connections stay bound to one loop that is never closed underneath them."""

    _shared: "_SigningLoop | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="aleo-bridge-remote-signing", daemon=True)
        self.thread.start()

    @classmethod
    def shared(cls) -> "_SigningLoop":
        with cls._lock:
            if cls._shared is None:
                cls._shared = cls()
            return cls._shared

    def run(self, coroutine: Awaitable[T], *, timeout: float) -> T:
        if threading.current_thread() is self.thread:  # pragma: no cover - adapters are called from sync code
            raise RuntimeError("remote signing cannot be awaited from its own loop thread")
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)  # type: ignore[arg-type]
        try:
            return future.result(timeout)
        except concurrent.futures.TimeoutError as exc:    # the builtin TimeoutError on 3.11+, distinct before
            if future.done():
                raise                                     # raised INSIDE the provider coroutine: not our deadline
            future.cancel()
            raise BridgeError(f"remote signing did not answer within {timeout:g} seconds; nothing was broadcast") from exc


def run_async(coroutine: Awaitable[T], *, timeout: float) -> T:
    """Run *coroutine* to completion on the shared signing loop and return its result, or raise
    ``BridgeError`` once *timeout* seconds pass without an answer."""
    return _SigningLoop.shared().run(coroutine, timeout=timeout)


__all__ = ["RemoteSignedTransaction", "checksum_address", "legacy_transaction_from_signature", "run_async",
           "signer_index", "solana_pubkey", "unsigned_evm_fields", "verified_evm_transaction",
           "verified_solana_signature"]
