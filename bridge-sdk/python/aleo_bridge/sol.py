"""Solana transport, connection and the ``bridge.sol`` module (SOL → Aleo over Hyperlane).

solders is imported lazily through :func:`_libs`; an install without the ``solana`` extra
raises :class:`MissingExtraError` at the point of use, never at import. Layouts live in
:mod:`aleo_bridge._sealevel` (pure); this module adds a synchronous JSON-RPC transport
(solana-py ≥ 0.36 is async-only), signing, broadcast and status polling.
"""
from __future__ import annotations

import asyncio
import base64
import inspect
import json
import os
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import requests

from . import _sealevel as sl
from ._calls import SolCall
from ._plan import build_plan
from .encoding import aleo_address_to_bytes32
from .errors import (
    BridgeError,
    CheckpointInvalidError,
    ConfigurationError,
    InsufficientBalanceError,
    InvalidAmountError,
    MissingExtraError,
    RegistryVersionMismatchError,
    RouteNotFoundError,
    UnsupportedRouteError,
)
from .registry import Route
from .types import DispatchReceipt, Fee, Plan, Receipt, SolanaHyperlaneQuote, Status, Step
from .units import format_decimal_amount, resolve_amount

DEFAULT_SOLANA_RPC_URL = "https://api.mainnet-beta.solana.com"
CONFIRMED = "confirmed"
COMMITMENTS = ("processed", "confirmed", "finalized")
SOLANA_CHAIN_ID = "solana"
SOLANA_SOL_ASSET_ID = "solana/sol"
ALEO_SOL_ASSET_ID = "aleo/sol"


@dataclass(frozen=True)
class _SolanaLibs:
    Keypair: Any
    Pubkey: Any
    Signature: Any
    Hash: Any
    Instruction: Any
    AccountMeta: Any
    MessageV0: Any
    to_bytes_versioned: Any
    VersionedTransaction: Any
    set_compute_unit_limit: Any


_LIBS: _SolanaLibs | None = None


def _libs() -> _SolanaLibs:
    """Import solders once; translate a missing extra into MissingExtraError."""
    global _LIBS
    if _LIBS is None:
        try:
            from solders.compute_budget import set_compute_unit_limit
            from solders.hash import Hash
            from solders.instruction import AccountMeta, Instruction
            from solders.keypair import Keypair
            from solders.message import MessageV0, to_bytes_versioned
            from solders.pubkey import Pubkey
            from solders.signature import Signature
            from solders.transaction import VersionedTransaction
        except ImportError as exc:
            raise MissingExtraError("solana", "Solana connections and SOL transfers") from exc
        _LIBS = _SolanaLibs(Keypair, Pubkey, Signature, Hash, Instruction, AccountMeta, MessageV0,
                            to_bytes_versioned, VersionedTransaction, set_compute_unit_limit)
    return _LIBS


# --- synchronous JSON-RPC transport (veil src/solana/rpc.ts) ------------------------------------

@dataclass(frozen=True)
class SendOptions:
    """Broadcast options; attribute-compatible with solana-py's ``TxOpts``."""
    skip_preflight: bool = False
    preflight_commitment: str = CONFIRMED
    skip_confirmation: bool = True


@dataclass(frozen=True)
class RpcResult:
    value: Any


@dataclass(frozen=True)
class LatestBlockhash:
    blockhash: Any                 # solders Hash
    last_valid_block_height: int


@dataclass(frozen=True)
class AccountInfo:
    data: bytes
    lamports: int
    owner: str


@dataclass(frozen=True)
class SignatureStatus:
    err: Any
    confirmation_status: str | None


@dataclass(frozen=True)
class TransactionMeta:
    log_messages: list[str] | None


@dataclass(frozen=True)
class TransactionWithMeta:
    meta: TransactionMeta | None


@dataclass(frozen=True)
class ConfirmedTransaction:
    transaction: TransactionWithMeta
    slot: int


class SolanaRpcClient:
    """Minimal synchronous Solana JSON-RPC client exposing the solana-py method surface SolModule uses.

    Every method issues one POST, validates the envelope (HTTP status, JSON, JSON-RPC ``error``,
    ``result`` presence) and the result shape, and returns an object with ``.value`` shaped like
    solana-py's response types. No method signs or retries.
    """

    def __init__(self, url: str, *, commitment: str = CONFIRMED, session: Any = None, timeout: float = 30.0) -> None:
        if commitment not in COMMITMENTS:
            raise ConfigurationError(f"Solana commitment must be one of {COMMITMENTS}, got {commitment!r}")
        self.url = url
        self.commitment = commitment
        self.timeout = timeout
        self._session = session or requests.Session()

    def _commitment(self, commitment: str | None) -> dict[str, str]:
        return {"commitment": str(commitment or self.commitment)}

    def _call(self, method: str, params: list[Any]) -> Any:
        try:
            response = self._session.post(self.url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
                                          timeout=self.timeout, headers={"content-type": "application/json", "cache-control": "no-cache"})
        except requests.RequestException as exc:
            raise BridgeError(f"Solana RPC {method} request failed: {exc}") from exc
        if not 200 <= response.status_code < 300:
            raise BridgeError(f"Solana RPC {method} request failed with HTTP status {response.status_code}")
        try:
            body = response.json()
        except ValueError as exc:
            raise BridgeError(f"Solana RPC {method} returned invalid JSON") from exc
        if not isinstance(body, dict):
            raise BridgeError(f"Solana RPC {method} returned an invalid JSON-RPC response")
        error = body.get("error")
        if error:
            details = f"; {json.dumps(error['data'])}" if isinstance(error, dict) and "data" in error else ""
            message = error.get("message", "unknown error") if isinstance(error, dict) else str(error)
            raise BridgeError(f"Solana RPC {method} returned a JSON-RPC error: {message}{details}")
        if "result" not in body:
            raise BridgeError(f"Solana RPC {method} returned an invalid result envelope")
        return body["result"]

    @staticmethod
    def _integer(method: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BridgeError(f"Solana RPC {method} returned an invalid result")
        return value

    @staticmethod
    def _contextual(method: str, result: Any) -> Any:
        if not isinstance(result, dict) or "value" not in result:
            raise BridgeError(f"Solana RPC {method} returned an invalid contextual result")
        return result["value"]

    def get_latest_blockhash(self, commitment: str | None = None) -> RpcResult:
        value = self._contextual("getLatestBlockhash", self._call("getLatestBlockhash", [self._commitment(commitment)]))
        if not isinstance(value, dict) or not isinstance(value.get("blockhash"), str) or not value["blockhash"]:
            raise BridgeError("Solana RPC getLatestBlockhash returned an invalid result")
        return RpcResult(LatestBlockhash(_libs().Hash.from_string(value["blockhash"]),
                                         self._integer("getLatestBlockhash", value.get("lastValidBlockHeight"))))

    def get_block_height(self, commitment: str | None = None) -> RpcResult:
        return RpcResult(self._integer("getBlockHeight", self._call("getBlockHeight", [self._commitment(commitment)])))

    def is_blockhash_valid(self, blockhash: Any, commitment: str | None = None) -> RpcResult:
        value = self._contextual("isBlockhashValid", self._call("isBlockhashValid", [str(blockhash), self._commitment(commitment)]))
        if not isinstance(value, bool):
            raise BridgeError("Solana RPC isBlockhashValid returned an invalid result")
        return RpcResult(value)

    def get_balance(self, pubkey: Any, commitment: str | None = None) -> RpcResult:
        value = self._contextual("getBalance", self._call("getBalance", [str(pubkey), self._commitment(commitment)]))
        return RpcResult(self._integer("getBalance", value))

    def get_account_info(self, pubkey: Any, commitment: str | None = None, encoding: str = "base64") -> RpcResult:
        if encoding != "base64":
            raise BridgeError("SolanaRpcClient.get_account_info supports base64 encoding only")
        value = self._contextual("getAccountInfo", self._call("getAccountInfo", [str(pubkey), {"encoding": "base64", **self._commitment(commitment)}]))
        if value is None:
            return RpcResult(None)
        data = value.get("data") if isinstance(value, dict) else None
        if not isinstance(data, list) or len(data) != 2 or not isinstance(data[0], str) or data[1] != "base64":
            raise BridgeError("Solana RPC getAccountInfo returned invalid base64 account data")
        try:
            raw = base64.b64decode(data[0], validate=True)
        except ValueError as exc:
            raise BridgeError("Solana RPC getAccountInfo returned invalid base64 account data") from exc
        return RpcResult(AccountInfo(raw, int(value.get("lamports", 0)), str(value.get("owner", ""))))

    def get_fee_for_message(self, message: Any, commitment: str | None = None) -> RpcResult:
        raw = bytes(message) if isinstance(message, (bytes, bytearray)) else _libs().to_bytes_versioned(message)
        value = self._contextual("getFeeForMessage", self._call("getFeeForMessage", [base64.b64encode(raw).decode(), self._commitment(commitment)]))
        return RpcResult(None if value is None else self._integer("getFeeForMessage", value))

    def get_minimum_balance_for_rent_exemption(self, usize: int, commitment: str | None = None) -> RpcResult:
        if isinstance(usize, bool) or not isinstance(usize, int) or usize < 0:
            raise BridgeError("Solana rent data length must be a non-negative integer")
        return RpcResult(self._integer("getMinimumBalanceForRentExemption",
                                       self._call("getMinimumBalanceForRentExemption", [usize, self._commitment(commitment)])))

    def send_raw_transaction(self, txn: bytes, opts: Any = None) -> RpcResult:
        opts = opts or SendOptions()
        config = {"encoding": "base64", "skipPreflight": bool(opts.skip_preflight), "preflightCommitment": str(opts.preflight_commitment)}
        result = self._call("sendTransaction", [base64.b64encode(bytes(txn)).decode(), config])
        if not isinstance(result, str) or not result:
            raise BridgeError("Solana RPC sendTransaction returned an invalid signature")
        return RpcResult(_libs().Signature.from_string(result))

    def get_signature_statuses(self, signatures: Sequence[Any], search_transaction_history: bool = False) -> RpcResult:
        value = self._contextual("getSignatureStatuses", self._call(
            "getSignatureStatuses", [[str(s) for s in signatures], {"searchTransactionHistory": bool(search_transaction_history)}]))
        if not isinstance(value, list) or len(value) != len(signatures):
            raise BridgeError("Solana RPC getSignatureStatuses returned an invalid result")
        statuses: list[SignatureStatus | None] = []
        for status in value:
            if status is None:
                statuses.append(None)
                continue
            if not isinstance(status, dict) or "err" not in status:
                raise BridgeError("Solana RPC getSignatureStatuses returned an invalid status")
            confirmation = status.get("confirmationStatus")
            if confirmation is not None and confirmation not in COMMITMENTS:
                raise BridgeError(f"Solana RPC getSignatureStatuses returned unsupported confirmation status: {confirmation}")
            statuses.append(SignatureStatus(status["err"], confirmation))
        return RpcResult(statuses)

    def get_transaction(self, tx_sig: Any, encoding: str = "json", commitment: str | None = None,
                        max_supported_transaction_version: int | None = None) -> RpcResult:
        config: dict[str, Any] = {"encoding": encoding, **self._commitment(commitment)}
        if max_supported_transaction_version is not None:
            config["maxSupportedTransactionVersion"] = max_supported_transaction_version
        result = self._call("getTransaction", [str(tx_sig), config])
        if result is None:
            return RpcResult(None)
        if not isinstance(result, dict) or "meta" not in result:
            raise BridgeError("Solana RPC getTransaction returned an invalid result")
        meta = result["meta"]
        if meta is None:
            return RpcResult(ConfirmedTransaction(TransactionWithMeta(None), int(result.get("slot", 0))))
        if not isinstance(meta, dict) or "logMessages" not in meta:
            raise BridgeError("Solana RPC getTransaction returned invalid metadata")
        logs = meta["logMessages"]
        if logs is not None and (not isinstance(logs, list) or not all(isinstance(line, str) for line in logs)):
            raise BridgeError("Solana RPC getTransaction returned invalid logs")
        return RpcResult(ConfirmedTransaction(TransactionWithMeta(TransactionMeta(logs)), int(result.get("slot", 0))))


class _AsyncClientAdapter:
    """Drives a solana-py ``AsyncClient`` (0.36+ is async-only) synchronously on a private event-loop
    thread, and supplies ``is_blockhash_valid`` (missing from solana-py) and ``send_raw_transaction``
    with our ``SendOptions`` translated to ``TxOpts``. Other methods are forwarded unchanged."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="aleo-bridge-solana-rpc", daemon=True)
        self._thread.start()
        self._closed = False

    def _run(self, coroutine: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result()

    def close(self, timeout: float = 5.0) -> None:
        """Stop the private event-loop thread and close the loop. Idempotent — a second call is a no-op."""
        if self._closed:
            return
        self._closed = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=timeout)
        if not self._loop.is_closed():
            self._loop.close()

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._client, name)
        if inspect.iscoroutinefunction(attribute):
            return lambda *args, **kwargs: self._run(attribute(*args, **kwargs))
        return attribute

    def is_blockhash_valid(self, blockhash: Any, commitment: str | None = None) -> Any:
        from solders.commitment_config import CommitmentLevel
        from solders.rpc.config import RpcContextConfig
        from solders.rpc.requests import IsBlockhashValid
        from solders.rpc.responses import IsBlockhashValidResp

        level = {"processed": CommitmentLevel.Processed, "confirmed": CommitmentLevel.Confirmed,
                 "finalized": CommitmentLevel.Finalized}[commitment or CONFIRMED]
        request = IsBlockhashValid(blockhash, RpcContextConfig(commitment=level))
        return self._run(self._client._provider.make_request(request, IsBlockhashValidResp))

    def send_raw_transaction(self, txn: bytes, opts: Any = None) -> Any:
        try:
            from solana.rpc.models import TxOpts
        except ImportError:  # solana-py < 0.36 kept TxOpts in solana.rpc.types
            try:
                from solana.rpc.types import TxOpts
            except ImportError as exc:
                raise MissingExtraError("solana", "solana-py AsyncClient transport") from exc
        opts = opts or SendOptions()
        tx_opts = TxOpts(skip_confirmation=True, skip_preflight=bool(opts.skip_preflight), preflight_commitment=str(opts.preflight_commitment))
        return self._run(self._client.send_raw_transaction(bytes(txn), tx_opts))


@runtime_checkable
class SolanaSigner(Protocol):
    """solana-py's signer shape: a solders ``Keypair`` or any wallet exposing these two methods."""

    def pubkey(self) -> Any: ...

    def sign_message(self, message: bytes) -> Any: ...


def keypair_from_private_key(private_key: str | bytes) -> Any:
    """Parse a Solana secret: base58 (Phantom export), a JSON array of 64 ints (solana-cli ``id.json``),
    64 raw bytes (seed ‖ pubkey) or a 32-byte seed."""
    libs = _libs()
    if isinstance(private_key, (bytes, bytearray, memoryview)):
        raw = bytes(private_key)
    else:
        text = private_key.strip()
        if text.startswith("["):
            try:
                values = json.loads(text)
            except ValueError as exc:
                raise ConfigurationError("Solana private key JSON array is malformed; expected the 64 integers of a solana-cli id.json") from exc
            if not isinstance(values, list) or not all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 255 for v in values):
                raise ConfigurationError("Solana private key JSON array must hold integers 0–255")
            raw = bytes(values)
        else:
            try:
                return libs.Keypair.from_base58_string(text)
            except Exception as exc:  # solders raises its own parse error types
                raise ConfigurationError("Solana private key is not a valid base58 64-byte secret") from exc
    if len(raw) == 64:
        return libs.Keypair.from_bytes(raw)
    if len(raw) == 32:
        return libs.Keypair.from_seed(raw)
    raise ConfigurationError(f"Solana private key must be 64 bytes (seed || pubkey) or a 32-byte seed, got {len(raw)}")


class Solana:
    """Solana transport plus an optional signer (spec §3.2).

    ``Solana(rpc_url)`` builds ``SolanaRpcClient(rpc_url, commitment="confirmed")``;
    ``Solana(client=…)`` reuses a caller-configured client: anything with the solana-py read/send
    method surface (its own commitment, timeout, headers), or a solana-py ``AsyncClient``, which is
    driven synchronously through :class:`_AsyncClientAdapter`. Exactly one of ``rpc_url``/``client``
    may be given; at most one of ``signer``/``private_key``. A connection without a signer is
    read-only (quotes and status reads work, ``send`` does not).
    """

    def __init__(self, rpc_url: str | None = None, *, client: Any = None, signer: Any = None,
                 private_key: str | bytes | None = None) -> None:
        if rpc_url is not None and client is not None:
            raise ConfigurationError("Solana(): pass rpc_url or client, not both")
        if signer is not None and private_key is not None:
            raise ConfigurationError("Solana(): pass signer or private_key, not both")
        if client is None:
            _libs()                                   # SolanaRpcClient returns solders Hash/Signature values
            rpc_url = rpc_url or DEFAULT_SOLANA_RPC_URL
            client = SolanaRpcClient(rpc_url, commitment=CONFIRMED)
        elif inspect.iscoroutinefunction(getattr(client, "get_balance", None)):
            client = _AsyncClientAdapter(client)      # solana-py ≥ 0.36 AsyncClient
        self._rpc_url = rpc_url
        self._client = client
        if private_key is not None:
            signer = keypair_from_private_key(private_key)
        if signer is not None and not (callable(getattr(signer, "pubkey", None)) and callable(getattr(signer, "sign_message", None))):
            raise ConfigurationError("Solana signer must expose pubkey() and sign_message(bytes) — a solders Keypair or a solana-py Signer")
        self._signer = signer

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Solana | None":
        """Private key from ``SOLANA_PRIVATE_KEY`` else ``BRIDGE_SOLANA_PRIVATE_KEY`` (the user's shell
        exports the latter); RPC from ``SOLANA_RPC_URL`` else ``BRIDGE_LIVE_SOLANA_RPC_URL`` else the
        default. A key (either name) → signing connection; URL alone → read-only; neither → ``None``."""
        env = os.environ if env is None else env
        key = env.get("SOLANA_PRIVATE_KEY") or env.get("BRIDGE_SOLANA_PRIVATE_KEY")
        url = env.get("SOLANA_RPC_URL") or env.get("BRIDGE_LIVE_SOLANA_RPC_URL") or None
        if key:
            return cls(url, private_key=key)
        if url:
            return cls(url)
        return None

    @property
    def client(self) -> Any:
        return self._client

    @property
    def signer(self) -> Any:
        return self._signer

    @property
    def rpc_url(self) -> str | None:
        return self._rpc_url

    @property
    def can_sign(self) -> bool:
        return self._signer is not None

    @property
    def pubkey(self) -> Any:
        if self._signer is None:
            raise ConfigurationError("Solana connection is read-only: pass signer= or private_key= to Solana() to sign")
        return self._signer.pubkey()

    @property
    def address(self) -> str | None:
        return None if self._signer is None else str(self._signer.pubkey())

    def sign_message(self, message: bytes) -> Any:
        """Fee-payer signature over compiled message bytes (``to_bytes_versioned`` for v0 messages)."""
        if self._signer is None:
            raise ConfigurationError("Solana connection is read-only: pass signer= or private_key= to Solana() to sign")
        return self._signer.sign_message(bytes(message))

    def close(self) -> None:
        """Release the wrapped client's resources (idempotent). A no-op unless the client exposes its own
        ``close()`` — e.g. the private event-loop thread behind an adapted async solana-py client."""
        closer = getattr(self._client, "close", None)
        if callable(closer):
            closer()

    def __enter__(self) -> "Solana":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def _confirmation_name(status: Any) -> str | None:
    """Normalise solders' TransactionConfirmationStatus enum (or a plain string) to 'processed'|'confirmed'|'finalized'."""
    value = getattr(status, "confirmation_status", None)
    if value is None:
        return None
    name = getattr(value, "name", None) or str(value)
    return str(name).rsplit(".", 1)[-1].lower()


@dataclass
class SolBuild:
    """Everything ``send`` needs after ``build``: the quote, the compiled message, the partially signed
    transaction, the unique-message address that seeds the PDAs, and the blockhash lifetime."""
    quote: SolanaHyperlaneQuote
    message: Any
    transaction: Any
    unique_message_address: str
    blockhash: str
    last_valid_block_height: int
    sender: str
    destination_domain: int


def _signature_status(client: Any, signature: Any) -> str | None:
    """'failed' | 'processed' | 'confirmed' | 'finalized' | None (unknown); raises on RPC errors."""
    value = client.get_signature_statuses([signature], search_transaction_history=True).value
    status = value[0] if value else None
    if status is None:
        return None
    if getattr(status, "err", None) is not None:
        return "failed"
    name = _confirmation_name(status)
    if name not in (None, "processed", "confirmed", "finalized"):
        raise BridgeError(f"Solana RPC getSignatureStatuses returned unsupported confirmation status: {name}")
    return name


def _poll_for_confirmation(client: Any, signature: str, blockhash: str, timeout_seconds: float,
                           poll_seconds: float) -> str | None:
    """Poll until confirmed/finalized ('confirmed'|'finalized'), the blockhash expires ('expired'), or the
    deadline passes (None). A status read that raises is swallowed — the transaction is already broadcast,
    so a transient RPC error must not be reported as a transfer failure. An on-chain ``err`` raises.

    The blockhash probe runs only while the signature has NO status at all: a ``processed`` transaction has
    already landed, and reporting it 'expired' would invite a resend (the same rule ``source_status`` applies).
    ``processed`` therefore keeps polling until it confirms or the deadline passes."""
    libs = _libs()
    sig = libs.Signature.from_string(signature)
    hash_ = libs.Hash.from_string(blockhash)
    interval = max(float(poll_seconds), 0.1)
    deadline = time.monotonic() + max(float(timeout_seconds), 0.0)
    while True:
        try:
            status = _signature_status(client, sig)
        except Exception:                           # noqa: BLE001 — transport/decoding errors are transient here
            status = None
        if status == "failed":
            raise BridgeError(f"Solana Hyperlane transfer failed on-chain: {signature}")
        if status in ("confirmed", "finalized"):
            return status
        if status is None:                          # no status at all: only then can the blockhash have expired
            try:
                if not client.is_blockhash_valid(hash_, commitment=CONFIRMED).value:
                    return "expired"
            except Exception:                       # noqa: BLE001
                pass                                # advisory while the signature may still land
        if time.monotonic() >= deadline:
            return None
        time.sleep(interval)


class SolModule:
    """``bridge.sol`` — Solana-origin SOL → Aleo over the Hyperlane warp route (spec §6).

    Reads (``balance``, ``quote_transfer_remote``, ``source_status``) work on a read-only
    connection; ``transfer_remote(...).send()`` needs a signer. Route metadata is re-validated
    from the live registry on every call.
    """

    def __init__(self, bridge: Any, conn: Solana) -> None:
        self._bridge = bridge
        self.conn = conn

    @property
    def client(self) -> Any:
        return self.conn.client

    @property
    def registry(self) -> Any:
        return self._bridge.registry

    @property
    def environment(self) -> str:
        return self._bridge.environment

    def outbound_route(self) -> Route:
        """The environment's SOL → Aleo Hyperlane route (RouteNotFoundError on testnet, which has none).

        ``SOLANA_SOL_ASSET_ID``/``ALEO_SOL_ASSET_ID`` are fixed mainnet asset ids (there is no
        testnet Solana chain in the registry), so ``find_route`` alone would resolve the mainnet
        route regardless of ``self.environment``; this guards that the route's own environment
        matches the module's, mirroring how ``EthModule`` scopes its route lookups by environment.
        """
        route = self.registry.find_route(SOLANA_SOL_ASSET_ID, ALEO_SOL_ASSET_ID, protocol="hyperlane")
        if route.environment != self.environment:
            raise RouteNotFoundError(f"No Solana Hyperlane route to Aleo for environment {self.environment!r}")
        return route

    def metadata(self, route: Route | None = None) -> sl.SolanaRouteMetadata:
        return sl.solana_route_metadata(route or self.outbound_route())

    # --- reads ------------------------------------------------------------------------------

    def _pubkey(self, address: str) -> Any:
        return _libs().Pubkey.from_string(address)

    def _account_data(self, address: str) -> bytes | None:
        value = self.client.get_account_info(self._pubkey(address), commitment=CONFIRMED, encoding="base64").value
        if value is None:
            return None
        data = value.data
        if isinstance(data, (list, tuple)):          # raw JSON shape: ["<base64>", "base64"]
            return base64.b64decode(data[0])
        return bytes(data)

    def _balance_of(self, address: str) -> int:
        return int(self.client.get_balance(self._pubkey(address), commitment=CONFIRMED).value)

    def balance(self) -> int:
        """Lamports held by the connected wallet."""
        address = self.conn.address
        if address is None:
            raise ConfigurationError("Solana connection is read-only: pass signer= or private_key= to Solana() to read the wallet balance")
        return self._balance_of(address)

    # --- quote ------------------------------------------------------------------------------

    def _make_plan(self, route: Route, *, recipient: str, amount_atomic: int, sender: str, decimals: int) -> Plan:
        return build_plan(self.registry, route, amount_atomic=amount_atomic, recipient=recipient, sender=sender)

    def _compile_message(self, metadata: sl.SolanaRouteMetadata, *, sender: str, unique_message: str,
                         recipient32: bytes, amount_atomic: int) -> tuple[Any, str, int]:
        """v0 message: [SetComputeUnitLimit(400_000), TransferRemote] with a confirmed blockhash."""
        libs = _libs()
        data = sl.build_transfer_remote_instruction_data(metadata.destination_domain, recipient32, amount_atomic)
        metas = [libs.AccountMeta(libs.Pubkey.from_string(m.address), is_signer=m.signer, is_writable=m.writable)
                 for m in sl.account_metas(metadata, sender, unique_message)]
        instruction = libs.Instruction(libs.Pubkey.from_string(metadata.warp_program_address), data, metas)
        latest = self.client.get_latest_blockhash(commitment=CONFIRMED).value
        message = libs.MessageV0.try_compile(
            libs.Pubkey.from_string(sender),
            [libs.set_compute_unit_limit(sl.COMPUTE_UNIT_LIMIT), instruction],
            [],
            latest.blockhash,
        )
        return message, str(latest.blockhash), int(latest.last_valid_block_height)

    def quote_transfer_remote(self, recipient: str, *, amount: str | None = None, amount_atomic: int | None = None,
                              sender: str | None = None, plan: Plan | None = None) -> SolanaHyperlaneQuote:
        """Lamports required for a SOL → Aleo transfer: amount + IGP payment + network fee + rent (spec §5 kind
        ``solana-hyperlane``). Reads Solana; never signs. ``sender`` defaults to the connected wallet and is required
        for the fee estimate; ``plan`` (from ``Bridge.quote``) pins recipient/amount/sender and must match the live
        registry version."""
        libs = _libs()
        route = self.outbound_route()
        metadata = sl.solana_route_metadata(route)
        decimals = self.registry.asset(route.source_asset_id).decimals
        if plan is not None:
            if plan.registry_version != self.registry.version:
                raise RegistryVersionMismatchError(
                    f"plan was prepared against registry {plan.registry_version}; this client runs {self.registry.version} — re-run quote()")
            if plan.route_id != route.id:
                raise UnsupportedRouteError(f"plan route {plan.route_id} is not the Solana Hyperlane route {route.id}")
            recipient, amount_atomic, amount, sender = plan.recipient, plan.amount_atomic, None, plan.sender
        amount_atomic = resolve_amount(amount=amount, amount_atomic=amount_atomic, decimals=decimals)
        if amount_atomic <= 0:
            raise InvalidAmountError("amount must be positive")
        recipient32 = aleo_address_to_bytes32(recipient)
        sender = sender or self.conn.address
        if sender is None:
            raise ConfigurationError("Solana sender is required to quote the transaction fee: configure a signer or pass sender=<base58 address>")
        if plan is None:
            plan = self._make_plan(route, recipient=recipient, amount_atomic=amount_atomic, sender=sender, decimals=decimals)

        igp_data = self._account_data(metadata.igp_account)
        if igp_data is None:
            raise BridgeError(f"Solana IGP account does not exist: {metadata.igp_account}")
        igp = sl.quote_igp_lamports(igp_data, metadata.destination_domain, metadata.destination_gas_amount)

        unique = libs.Keypair()                       # disposable: only its pubkey seeds the fee-estimate message
        message, _blockhash, _height = self._compile_message(
            metadata, sender=sender, unique_message=str(unique.pubkey()), recipient32=recipient32, amount_atomic=amount_atomic)
        fee_value = self.client.get_fee_for_message(message, commitment=CONFIRMED).value
        if fee_value is None:
            raise BridgeError("Solana RPC getFeeForMessage returned no fee (the blockhash is unknown to the node); retry")
        fee = int(fee_value)
        rent = sum(int(self.client.get_minimum_balance_for_rent_exemption(size).value)
                   for size in (sl.GAS_PAYMENT_ACCOUNT_DATA_LENGTH, sl.DISPATCHED_MESSAGE_ACCOUNT_DATA_LENGTH, 0))
        total = amount_atomic + igp + fee + rent
        fees = (
            Fee("interchain-gas", SOLANA_CHAIN_ID, route.source_asset_id, format_decimal_amount(igp, decimals), True),
            Fee("network", SOLANA_CHAIN_ID, route.source_asset_id, format_decimal_amount(fee, decimals), True),
            Fee("rent", SOLANA_CHAIN_ID, route.source_asset_id, format_decimal_amount(rent, decimals), True),
        )
        return SolanaHyperlaneQuote(
            kind="solana-hyperlane", plan=plan, fees=fees, amount_out=plan.amount,
            igp_lamports=igp, network_fee_lamports=fee, rent_lamports=rent, total_lamports=total,
            unique_message_address=str(unique.pubkey()),
        )

    # --- write ------------------------------------------------------------------------------

    def transfer_remote(self, recipient: str, *, amount: str | None = None, amount_atomic: int | None = None,
                        plan: Plan | None = None) -> SolCall[DispatchReceipt]:
        """Send native SOL to an Aleo address over the Hyperlane warp route (spec §6).

        Returns a :class:`SolCall`: ``build()`` previews the partially signed transaction,
        ``send()`` moves funds (amount + IGP payment + network fee + rent leave the wallet).
        ``plan`` (from ``Bridge.execute``) must have been prepared for the connected wallet; its
        registry version and route id are re-checked against the live registry when the call runs.
        """
        route = self.outbound_route()
        sl.solana_route_metadata(route)                                  # refuse inactive/malformed routes early
        decimals = self.registry.asset(route.source_asset_id).decimals
        if plan is not None:
            recipient, amount_atomic, amount = plan.recipient, plan.amount_atomic, None
        amount_atomic = resolve_amount(amount=amount, amount_atomic=amount_atomic, decimals=decimals)
        if amount_atomic <= 0:
            raise InvalidAmountError("amount must be positive")
        aleo_address_to_bytes32(recipient)

        def build_result(receipt: Receipt) -> DispatchReceipt:
            return DispatchReceipt(transaction_id=receipt.source_tx_id or receipt.id, route_id=route.id,
                                   message_id=receipt.protocol_state.get("messageId"),
                                   amount_atomic=amount_atomic, receipt=receipt)

        return SolCall(self, route=route, recipient=recipient, amount_atomic=amount_atomic, plan=plan,
                       build_result=build_result, store=getattr(self._bridge, "checkpoints", None))

    def _build_transaction(self, *, route: Route, recipient: str, amount_atomic: int, plan: Plan | None) -> SolBuild:
        libs = _libs()
        sender = self.conn.address
        if sender is None:
            raise ConfigurationError("Solana connection is read-only: pass signer= or private_key= to Solana() to build transactions")
        if plan is not None and plan.sender and plan.sender != sender:
            raise ConfigurationError(f"Prepared sender {plan.sender} does not match connected account {sender}")
        quote = self.quote_transfer_remote(recipient, amount_atomic=amount_atomic, sender=sender, plan=plan)
        metadata = sl.solana_route_metadata(route)
        unique = libs.Keypair()                       # fresh per build: seeds the dispatched-message and gas-payment PDAs
        message, blockhash, last_valid_block_height = self._compile_message(
            metadata, sender=sender, unique_message=str(unique.pubkey()),
            recipient32=aleo_address_to_bytes32(quote.plan.recipient), amount_atomic=quote.plan.amount_atomic)
        keys = list(message.account_keys)
        if keys[0] != libs.Pubkey.from_string(sender):
            raise BridgeError("compiled Solana message does not list the sender as fee payer")
        signatures = [libs.Signature.default()] * message.header.num_required_signatures
        signatures[keys.index(unique.pubkey())] = unique.sign_message(libs.to_bytes_versioned(message))
        transaction = libs.VersionedTransaction.populate(message, signatures)
        return SolBuild(quote=replace(quote, unique_message_address=str(unique.pubkey())), message=message,
                        transaction=transaction, unique_message_address=str(unique.pubkey()), blockhash=blockhash,
                        last_valid_block_height=last_valid_block_height, sender=sender,
                        destination_domain=metadata.destination_domain)

    def _source_receipt(self, built: SolBuild, signature: str) -> Receipt:
        return Receipt(
            id=signature, protocol="hyperlane", status=Status.SOURCE_CONFIRMING, source_tx_id=signature,
            protocol_state={
                "routeId": built.quote.plan.route_id,
                "signature": signature,
                "uniqueMessageAddress": built.unique_message_address,
                "destinationDomain": built.destination_domain,
                "quotedLamports": str(built.quote.total_lamports),
                "blockhash": built.blockhash,
                "lastValidBlockHeight": str(built.last_valid_block_height),
            },
        )

    def _transaction_logs(self, signature: str) -> list[str] | None:
        libs = _libs()
        value = self.client.get_transaction(libs.Signature.from_string(signature), encoding="json",
                                            commitment=CONFIRMED, max_supported_transaction_version=0).value
        if value is None:
            return None
        meta = value.transaction.meta
        return None if meta is None else meta.log_messages

    def _delivery_pending(self, receipt: Receipt, signature: str) -> Receipt:
        """Settle a confirmed signature as DELIVERY_PENDING, with the Mailbox message id when readable.

        The logs supply nothing but the message id, and ``messageIdUnavailable`` already covers a missing
        dispatch line, so a failing ``getTransaction`` degrades to that fallback instead of turning a
        transfer that is already on-chain into a reported failure.
        """
        try:
            logs = self._transaction_logs(signature)
        except Exception:                           # noqa: BLE001 — RPC or decode failure reading the logs
            logs = None
        message_id = sl.extract_hyperlane_message_id(logs)
        state = dict(receipt.protocol_state)
        if message_id:
            state["messageId"] = message_id
        else:
            state["messageIdUnavailable"] = True
        return receipt.replace(id=message_id or signature, status=Status.DELIVERY_PENDING, protocol_state=state)

    def _checkpoint(self, plan: Plan, receipt: Receipt, on_checkpoint: "Callable[[Checkpoint], None] | None",
                    store: "CheckpointStore | None", signature: str) -> None:
        """Emit the checkpoint for a just-broadcast *signature* to the caller first, then the store.

        Mirrors ``EvmCall._checkpoint``: the caller's callback runs before the store because the
        transaction is already on the wire; a store failure is then fatal and names the signature,
        because losing it silently would strand funds.
        """
        from .checkpoint import create_checkpoint

        checkpoint = create_checkpoint(plan, receipt, self.registry)
        if on_checkpoint is not None:
            on_checkpoint(checkpoint)                     # the caller's own callback: errors are theirs
        if store is not None:
            try:
                store.save(checkpoint)
            except Exception as exc:  # noqa: BLE001 — any store backend failure
                raise BridgeError(
                    f"Solana transaction {signature} WAS broadcast but its checkpoint {checkpoint.id} could not be "
                    f"saved ({exc}); record the signature before retrying — resending would double-spend") from exc

    def _submit(self, built: SolBuild, *, wait: bool, timeout_seconds: float, poll_seconds: float,
                on_checkpoint: "Callable[[Checkpoint], None] | None" = None,
                store: "CheckpointStore | None" = None) -> Receipt:
        libs = _libs()
        quote = built.quote
        balance = self._balance_of(built.sender)
        if balance < quote.total_lamports:
            raise InsufficientBalanceError(
                f"Insufficient Solana balance for this Hyperlane transfer: balance {balance} lamports, "
                f"required {quote.total_lamports} lamports (amount {quote.plan.amount_atomic} "
                f"+ gas {quote.igp_lamports + quote.network_fee_lamports} + rent {quote.rent_lamports})")
        signatures = list(built.transaction.signatures)
        payer_index = list(built.message.account_keys).index(libs.Pubkey.from_string(built.sender))
        signatures[payer_index] = self.conn.sign_message(libs.to_bytes_versioned(built.message))
        signed = libs.VersionedTransaction.populate(built.message, signatures)
        opts = SendOptions(skip_preflight=False, preflight_commitment=CONFIRMED)
        signature = str(self.client.send_raw_transaction(bytes(signed), opts=opts).value)
        receipt = self._source_receipt(built, signature)
        self._checkpoint(quote.plan, receipt, on_checkpoint, store, signature)
        if not wait:
            return receipt
        try:
            outcome = _poll_for_confirmation(self.client, signature, built.blockhash, timeout_seconds, poll_seconds)
            if outcome is None:
                return receipt
            if outcome == "expired":
                return receipt.replace(status=Status.EXPIRED, protocol_state={
                    **receipt.protocol_state, "blockhashExpired": True,
                    "sourceError": f"Solana transaction expired before confirmation: {signature}"})
            return self._delivery_pending(receipt, signature)
        except BridgeError as exc:
            if signature in str(exc):
                raise
            raise BridgeError(f"Solana Hyperlane transfer {signature} failed after broadcast: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 — any post-broadcast failure names the signature
            raise BridgeError(f"Solana Hyperlane transfer {signature} failed after broadcast: {exc}") from exc

    # --- status -----------------------------------------------------------------------------

    def source_status(self, plan: Plan, receipt: Receipt) -> Receipt:
        """One refresh of a SOURCE_CONFIRMING Solana receipt (veil ``getSourceStatus``): unknown → unchanged, or
        EXPIRED once the checkpointed blockhash is invalid; processed → unchanged; failed → raises;
        confirmed/finalized → DELIVERY_PENDING with the Mailbox message id when the log is available."""
        if receipt.protocol != "hyperlane" or receipt.status is not Status.SOURCE_CONFIRMING or not receipt.source_tx_id:
            raise BridgeError("Solana Hyperlane source status requires a source-confirming Hyperlane receipt with a signature")
        if receipt.protocol_state.get("routeId") != plan.route_id:
            raise BridgeError(f"receipt route {receipt.protocol_state.get('routeId')} does not match plan route {plan.route_id}")
        libs = _libs()
        signature = receipt.source_tx_id
        status = _signature_status(self.client, libs.Signature.from_string(signature))
        if status is None:
            blockhash = receipt.protocol_state.get("blockhash")
            height = receipt.protocol_state.get("lastValidBlockHeight")
            if blockhash is None and height is None:
                return receipt
            if not isinstance(blockhash, str) or not blockhash or not isinstance(height, str) or not height.isdigit():
                raise CheckpointInvalidError("Solana Hyperlane source receipt has an invalid blockhash lifetime")
            try:
                hash_ = libs.Hash.from_string(blockhash)
            except Exception as exc:
                raise CheckpointInvalidError("Solana Hyperlane source receipt has an invalid blockhash lifetime") from exc
            try:
                valid = bool(self.client.is_blockhash_valid(hash_, commitment=CONFIRMED).value)
            except Exception:
                return receipt                          # advisory read; keep waiting
            if not valid:
                return receipt.replace(status=Status.EXPIRED, protocol_state={
                    **receipt.protocol_state, "blockhashExpired": True,
                    "sourceError": f"Solana transaction expired before confirmation: {signature}"})
            return receipt
        if status == "processed":
            return receipt
        if status == "failed":
            raise BridgeError(f"Solana Hyperlane transfer failed on-chain: {signature}")
        return self._delivery_pending(receipt, signature)
