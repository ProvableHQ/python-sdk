"""Ethereum connection and the ``bridge.eth`` module (Hyperlane + xReserve, Ethereum origin).

``web3`` and ``eth_account`` are imported lazily so ``import aleo_bridge`` works
without the ``evm`` extra; the first call that needs them raises
``MissingExtraError("evm", ...)``.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, fields
from typing import Any, Mapping

from . import encoding
from ._calls import EvmCall, EvmOutcome, EvmStep
from ._evm_abi import ERC20_ABI, EVM_CHAIN_BY_ENVIRONMENT, MAILBOX_ABI, WARP_ROUTE_ABI, XRESERVE_ABI, ZERO_ADDRESS
from ._plan import build_plan as _plan_for   # kept as a module-level name: existing callers import eth._plan_for
from .checkpoint import Checkpoint
from .errors import (AmbiguousRouteError, BridgeError, ChainMismatchError, CheckpointInvalidError, ConfigurationError,
                     InsufficientBalanceError, InvalidAmountError, InvalidRecipientError, MissingExtraError,
                     RegistryVersionMismatchError, RouteNotFoundError, RouteUnavailableError, UnsupportedRouteError)
from .registry import Asset, Chain, Registry, Route
from .types import (ChainStatus, DepositReceipt, DispatchReceipt, EvmHyperlaneQuote, EvmXReserveQuote, Fee, Plan,
                    Receipt, Status)
from .units import format_decimal_amount, parse_decimal_amount, resolve_amount


def _web3():
    try:
        import web3
    except ImportError as exc:  # pragma: no cover - exercised by test_import_without_web3
        raise MissingExtraError("evm", "Ethereum connections") from exc
    return web3


_HASH_RE = re.compile(r"^0x[0-9a-fA-F]{64}$")

MIN_PRIORITY_FEE_WEI = 100_000_000
"""Floor for the EIP-1559 priority tip, in wei (0.1 gwei).

Public RPC endpoints answer ``eth_maxPriorityFeePerGas`` with 0 (``ethereum-rpc.publicnode.com``
does), and a zero-tip transaction is not picked up by builders: on 2026-09-22 a mainnet WBTC
dispatch sat unmined until another transaction of ours took its nonce and replaced it. Every
suggestion is therefore raised to this floor; ``Ethereum(..., min_priority_fee_wei=...)`` overrides it.
"""

LOG_SCAN_CHUNK_BLOCKS = 5_000
"""Default block span per ``eth_getLogs`` request during recovery scans.

Public RPC endpoints cap the range (and the result size) of a single ``eth_getLogs``; an
unbounded ``{"fromBlock": n}`` filter is refused outright by most of them once ``n`` is far
enough behind the head. Recovery therefore walks the range in chunks of this many blocks.
"""


HEAD_RACE_RE = re.compile(r"beyond current head|head block|exceeds.*head", re.IGNORECASE)
"""A JSON-RPC error that means "your ``toBlock`` is ahead of the block I have", not a bad request.

A load-balanced endpoint answers ``eth_blockNumber`` and ``eth_getLogs`` from different nodes, so a
range bounded by one node's head can be beyond another's: on 2026-09-22 publicnode answered a
recovery scan with ``-32602 block range extends beyond current head block`` for a range it had
itself just handed out. Kept deliberately narrow — only phrasings about the HEAD block match, so a
genuine "range too large" or "query returned more than N results" still fails loudly rather than
being retried forever. ``lifecycle._is_transient_error`` reads it too, so ``wait`` retries instead
of aborting.
"""

LOG_SCAN_HEAD_RACE_RETRIES = 5
"""How many times one ``eth_getLogs`` chunk re-reads the head and retries before giving up."""

LOG_SCAN_HEAD_RACE_SLEEP_SECONDS = 0.5
"""Pause between those retries — long enough for a lagging node to catch up, short enough to stay
inside a recovery call."""


def _provider_errors() -> tuple[type[BaseException], ...]:
    """Everything a JSON-RPC provider can throw for one ``eth_getLogs``: web3's own errors, the
    ``ValueError`` older/raw providers raise for a JSON-RPC error response, and transport errors."""
    from web3.exceptions import Web3Exception

    errors: list[type[BaseException]] = [Web3Exception, ValueError]
    try:
        import requests
    except ImportError:  # pragma: no cover - requests ships with web3's HTTP provider
        pass
    else:
        errors.append(requests.RequestException)
    return tuple(errors)


def _eth_account():
    try:
        from eth_account import Account
    except ImportError as exc:  # pragma: no cover
        raise MissingExtraError("evm", "Ethereum signing") from exc
    return Account


class Ethereum:
    """Transport + optional signer for Ethereum-origin bridge actions.

    Three interchangeable forms::

        Ethereum(rpc_url, private_key=key)          # SDK builds Web3(HTTPProvider(rpc_url))
        Ethereum(w3=my_w3, signer=local_account)    # caller's Web3, caller's eth_account signer
        Ethereum(w3=my_w3)                          # signs via w3.eth.default_account + caller middleware,
                                                    # else read-only

    Sending: with a ``LocalAccount`` the SDK fills nonce/gas/fee fields, signs, and
    ``send_raw_transaction``s; in default-account mode it calls
    ``w3.eth.send_transaction`` so the caller's middleware signs. Receipts are
    polled on the same ``Web3``.
    """

    def __init__(self, rpc_url: str | None = None, *, w3: Any = None, signer: Any = None,
                 private_key: str | None = None, min_priority_fee_wei: int = MIN_PRIORITY_FEE_WEI) -> None:
        if (rpc_url is None) == (w3 is None):
            raise ConfigurationError("Pass exactly one of rpc_url or w3 to Ethereum(...)")
        if signer is not None and private_key is not None:
            raise ConfigurationError("Pass at most one of signer or private_key to Ethereum(...)")
        if isinstance(min_priority_fee_wei, bool) or not isinstance(min_priority_fee_wei, int) or min_priority_fee_wei < 0:
            raise ConfigurationError("min_priority_fee_wei must be a non-negative integer number of wei")
        if w3 is None:
            web3 = _web3()
            w3 = web3.Web3(web3.HTTPProvider(rpc_url))
        if private_key is not None:
            signer = _eth_account().from_key(private_key)
        self._w3 = w3
        self._signer = signer
        self._chain_id: int | None = None
        self.min_priority_fee_wei = min_priority_fee_wei
        self._last_nonce: int | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Ethereum | None":
        """``EVM_PRIVATE_KEY`` + ``ETHEREUM_RPC_URL`` (both or neither) → signing connection; neither → None.

        Aliases (the user's live shell / veil config export these names instead):
        ``BRIDGE_EVM_PRIVATE_KEY`` for the key, ``BRIDGE_LIVE_ETHEREUM_RPC_URL`` for the RPC url.
        The primary variable wins when both a primary and its alias are set; the both-or-neither
        rule applies to whichever pair resolves (primary, falling back to alias, per variable).
        """
        env = os.environ if env is None else env
        key = env.get("EVM_PRIVATE_KEY") or env.get("BRIDGE_EVM_PRIVATE_KEY")
        url = env.get("ETHEREUM_RPC_URL") or env.get("BRIDGE_LIVE_ETHEREUM_RPC_URL")
        if bool(key) != bool(url):
            raise ConfigurationError(
                "Set both EVM_PRIVATE_KEY and ETHEREUM_RPC_URL or neither "
                "(aliases: BRIDGE_EVM_PRIVATE_KEY, BRIDGE_LIVE_ETHEREUM_RPC_URL)")
        if not key:
            return None
        return cls(url, private_key=key)

    @property
    def w3(self) -> Any:
        return self._w3

    @property
    def address(self) -> str | None:
        """Checksummed sender address: signer → ``w3.eth.default_account`` → ``None``."""
        if self._signer is not None:
            return self._signer.address
        default = getattr(self._w3.eth, "default_account", None)
        if isinstance(default, str) and default:
            return _web3().Web3.to_checksum_address(default)
        return None

    @property
    def can_sign(self) -> bool:
        return self.address is not None

    @property
    def last_broadcast_nonce(self) -> int | None:
        """Highest nonce this connection has put on the wire, or ``None`` before the first send."""
        return self._last_nonce

    @property
    def chain_id(self) -> int:
        """``eth_chainId``, read once and cached."""
        if self._chain_id is None:
            self._chain_id = int(self._w3.eth.chain_id)
        return self._chain_id

    def require_address(self) -> str:
        address = self.address
        if address is None:
            raise ConfigurationError(
                "This Ethereum connection is read-only: pass private_key= or signer= to Ethereum(...), "
                "or set w3.eth.default_account with signing middleware")
        return address

    def send_transaction(self, tx: dict) -> str:
        """Broadcast one transaction and return its ``0x`` hash.

        Fills ``from``/``chainId``/``value`` when missing. Local signer: also fills
        ``nonce``/``gas``/fee fields, signs, ``send_raw_transaction``. Default-account
        mode: ``send_transaction`` (the caller's middleware signs and fills gas).
        Read-only: ``ConfigurationError``.

        Two rules protect a transaction the node is slow to mine. The EIP-1559 tip is
        ``max(eth_maxPriorityFeePerGas, min_priority_fee_wei)`` — public RPCs suggest 0, and a
        zero-tip transaction can sit unmined for hours. The nonce is
        ``max(pending count, last nonce this connection broadcast + 1)`` — a load-balanced RPC
        can stop reporting our own pending transaction, and reusing its nonce replaces it.
        """
        sender = self.require_address()
        Web3 = _web3().Web3
        tx = dict(tx)
        tx.setdefault("from", sender)
        if Web3.to_checksum_address(tx["from"]) != sender:
            raise ConfigurationError(f"Transaction sender {tx['from']} does not match the configured account {sender}")
        tx.setdefault("chainId", self.chain_id)
        tx.setdefault("value", 0)
        if self._signer is None:
            # Default-account mode: the caller's middleware signs, so the hash only exists once the node
            # answers. There is nothing to capture beforehand and nothing to compare the answer against,
            # so the ambiguous-send protections below (lost response, echo mismatch, EvmCall's single-use
            # guard) cannot apply here — a failed send leaves the caller unable to tell whether the
            # transaction is in the mempool. Prefer a local signer (private_key=/signer=) for anything
            # that moves funds.
            return Web3.to_hex(self._w3.eth.send_transaction(tx))
        if "nonce" not in tx:
            pending = int(self._w3.eth.get_transaction_count(sender, "pending"))
            # A transaction we broadcast and the RPC has since forgotten is still in SOMEONE's
            # mempool; handing its nonce out again is how one leg silently replaces another.
            tx["nonce"] = pending if self._last_nonce is None else max(pending, self._last_nonce + 1)
        if "gas" not in tx:
            estimate_fields = {k: v for k, v in tx.items() if k in ("from", "to", "data", "value")}
            tx["gas"] = int(self._w3.eth.estimate_gas(estimate_fields)) * 12 // 10
        if "gasPrice" not in tx and "maxFeePerGas" not in tx:
            base_fee = self._w3.eth.get_block("latest").get("baseFeePerGas")
            if base_fee is None:
                tx["gasPrice"] = int(self._w3.eth.gas_price)
            else:
                tip = max(int(self._w3.eth.max_priority_fee), self.min_priority_fee_wei)
                tx["maxPriorityFeePerGas"] = tip
                tx["maxFeePerGas"] = int(base_fee) * 2 + tip
        signed = self._signer.sign_transaction(tx)
        # Reserve the nonce before the broadcast, not after: a send whose RPC response is lost may
        # still have reached the node, and that transaction must never have its nonce reused.
        nonce = int(tx["nonce"])
        self._last_nonce = nonce if self._last_nonce is None else max(self._last_nonce, nonce)
        # The hash is fixed by the signature, so it exists before the broadcast. Capture it first: if the
        # RPC answer is lost the node may still have accepted the bytes, and a caller who never learns the
        # hash cannot tell a failed send from a landed one (and would resend, risking a double spend).
        local_hash = Web3.to_hex(signed.hash)
        try:
            echoed = Web3.to_hex(self._w3.eth.send_raw_transaction(signed.raw_transaction))
        except Exception as exc:  # noqa: BLE001 — any transport/JSON-RPC failure loses the response, not the send
            error = BridgeError(
                f"Ethereum transaction {local_hash} may have been broadcast; the RPC response was lost: {exc}"
                " — check bridge.eth.source_status / the explorer before retrying")
            # EvmCall reads this to arm its single-use guard: an ambiguous send must not be retried.
            error.broadcast_id = local_hash          # type: ignore[attr-defined]
            raise error from exc
        if echoed.lower() != local_hash.lower():
            error = BridgeError(
                f"Ethereum RPC echoed transaction hash {echoed} for a transaction signed as {local_hash}; "
                "refusing to checkpoint or follow the wrong hash — check bridge.eth.source_status / the "
                f"explorer for {local_hash} before retrying")
            # The node ANSWERED, so it took the bytes: they may sit in its mempool under local_hash even
            # though it echoed something else. That is the same ambiguity as a lost response, so arm
            # EvmCall's single-use guard here too rather than letting a retry sign a second transfer.
            error.broadcast_id = local_hash          # type: ignore[attr-defined]
            raise error
        return local_hash

    def wait_for_receipt(self, tx_hash: str, *, timeout_seconds: float, poll_seconds: float) -> dict | None:
        """Poll ``wait_for_transaction_receipt``; ``None`` on timeout (a timeout is not a failure)."""
        from web3.exceptions import TimeExhausted

        try:
            return self._w3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout_seconds, poll_latency=poll_seconds)
        except TimeExhausted:
            return None

    def get_receipt(self, tx_hash: str) -> dict | None:
        """One ``eth_getTransactionReceipt`` read; ``None`` while the transaction is unmined or unknown."""
        from web3.exceptions import TransactionNotFound

        try:
            return self._w3.eth.get_transaction_receipt(tx_hash)
        except TransactionNotFound:
            return None


_REGISTRY_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)


@dataclass(frozen=True)
class _HyperlaneRouteMetadata:
    """Validated Hyperlane route metadata (mirrors veil ``protocols/hyperlane/evm.ts`` ``routeMetadata``).

    Every address/domain that reaches a contract call is checked here first, so a corrupted or
    malformed registry entry fails with ``ConfigurationError`` before any RPC read.
    """

    router: str
    router_type: str            # "native" | "collateral"
    token: str | None           # collateral ERC-20; None on native
    mailbox: str
    interchain_gas_paymaster: str
    interchain_security_module: str
    source_chain_id: int
    destination_domain: int
    destination_router: str
    registry_commit: str
    requires_approval_reset: bool


@dataclass(frozen=True)
class _HyperlaneQuote:
    """Router-level facts behind an ``EvmHyperlaneQuote`` (addresses never leave the module)."""

    router: str
    router_type: str            # "native" | "collateral"
    token: str | None           # collateral ERC-20
    destination_domain: int
    recipient_bytes32: bytes
    amount_atomic: int
    native_value_atomic: int
    native_fee_atomic: int
    token_amount_atomic: int    # 0 on native routes
    allowance_atomic: int | None
    requires_approval_reset: bool


_DIGIT_STRING_RE = re.compile(r"^[0-9]+$")


@dataclass(frozen=True)
class _XReserveRouteMetadata:
    """Validated xReserve route metadata (mirrors veil ``protocols/xreserve/evmToAleo.ts`` ``routeMetadata``).

    Every address, domain, program name and fee that reaches a contract call or an Aleo encoder is
    checked here first, so a corrupted or malformed registry entry fails with ``ConfigurationError``
    before any RPC read.
    """

    xreserve_contract: str
    source_chain_id: int
    source_domain: int
    remote_domain: int
    remote_token_bytes32: bytes
    minimum_amount_atomic: int
    withdrawal_fee_atomic: int
    max_fee_atomic: int
    bridge_program: str
    wrapper_program: str
    remote_token: str
    attestation_base_url: str


@dataclass(frozen=True)
class _XReserveQuote:
    """Contract-level facts behind an ``EvmXReserveQuote``; also rebuilt from receipts during status/recovery."""

    xreserve_contract: str
    token: str
    source_chain_id: int
    source_domain: int
    remote_domain: int
    remote_token_bytes32: bytes
    remote_recipient_bytes32: bytes
    amount_atomic: int
    max_fee_atomic: int
    hook_data: bytes
    balance_atomic: int
    allowance_atomic: int
    bridge_program: str
    wrapper_program: str


class EthModule:
    """``bridge.eth`` — Ethereum-origin Hyperlane and xReserve actions (reads return values, writes return ``EvmCall``)."""

    def __init__(self, bridge: Any, conn: Ethereum, *, log_scan_chunk_blocks: int = LOG_SCAN_CHUNK_BLOCKS) -> None:
        self.bridge = bridge
        self.conn = conn
        self.registry: Registry = bridge.registry
        self.network: str = bridge.network            # "mainnet" | "testnet" → aleo.<network> for encoders
        self.chain: Chain = self.registry.chain(EVM_CHAIN_BY_ENVIRONMENT[bridge.environment])
        self.log_scan_chunk_blocks = log_scan_chunk_blocks        # recovery eth_getLogs span; lower it for strict RPCs
        self.log_scan_head_race_retries = LOG_SCAN_HEAD_RACE_RETRIES
        self.log_scan_head_race_sleep = LOG_SCAN_HEAD_RACE_SLEEP_SECONDS
        self.sleep: Any = time.sleep          # replaced in tests so the retry pause is recorded, never waited

    @property
    def log_scan_chunk_blocks(self) -> int:
        """Blocks per ``eth_getLogs`` request during recovery scans; lower it for strict RPCs.

        Validated on every assignment, not just in the constructor: ``_scan_logs`` advances its
        cursor by this many blocks per pass, so a zero or negative chunk would loop forever
        against a live chain rather than fail.
        """
        return self._log_scan_chunk_blocks

    @log_scan_chunk_blocks.setter
    def log_scan_chunk_blocks(self, value: Any) -> None:
        if int(value) < 1:
            raise ConfigurationError("log_scan_chunk_blocks must be at least 1")
        self._log_scan_chunk_blocks = int(value)

    # -- resolution ---------------------------------------------------------------------------

    def _asset(self, ref: Any) -> Asset:
        """Accept an ``Asset``, ``"chain/key"``, ``(chain, key)``, or a bare key/symbol on this chain."""
        if isinstance(ref, Asset):
            return ref
        if isinstance(ref, tuple) or (isinstance(ref, str) and "/" in ref):
            return self.registry.asset(ref)
        matches = [a for a in self.registry.assets(chain=self.chain.id)
                   if a.key.lower() == str(ref).lower() or a.symbol.lower() == str(ref).lower()]
        if len(matches) != 1:
            raise RouteNotFoundError(f"No unique asset {ref!r} on {self.chain.id}; use 'chain/key'")
        return matches[0]

    def _hyperlane_route(self, asset: Asset) -> Route:
        if asset.chain_id != self.chain.id:
            raise RouteNotFoundError(f"{asset.id} is not on {self.chain.id}; bridge.eth drives {self.chain.id} only")
        candidates = [r for r in self.registry.routes(protocol="hyperlane", include_unavailable=True,
                                                     environment=self.bridge.environment)
                      if r.source_asset_id == asset.id]
        if not candidates:
            if any(r.source_asset_id == asset.id for r in self.registry.routes(include_unavailable=True,
                                                                                environment=self.bridge.environment)):
                raise BridgeError(f"{asset.id} is not a Hyperlane route source; use deposit_usdc for xReserve")
            raise RouteNotFoundError(f"No Hyperlane route from {asset.id}")
        active = [r for r in candidates if r.availability == "active"]
        if not active:
            raise RouteUnavailableError(f"Hyperlane route is not executable ({candidates[0].availability}): {candidates[0].id}")
        if len(active) > 1:
            raise AmbiguousRouteError(f"{len(active)} active Hyperlane routes from {asset.id}; pass route=")
        return active[0]

    def _xreserve_route(self) -> Route:
        routes = [r for r in self.registry.routes(protocol="xreserve", environment=self.bridge.environment)
                  if self.registry.asset(r.source_asset_id).chain_id == self.chain.id]
        if len(routes) != 1:
            raise RouteNotFoundError(f"Expected exactly one xReserve deposit route from {self.chain.id}, found {len(routes)}")
        if routes[0].availability != "active":
            raise RouteUnavailableError(f"xReserve route is not executable: {routes[0].id}")
        return routes[0]

    def _route_for_plan(self, plan: Plan) -> Route:
        """Re-resolve the route from the live registry (invariant 1); never trust plan-carried addresses."""
        if plan.registry_version != self.registry.version:
            raise RegistryVersionMismatchError(
                f"Plan uses registry {plan.registry_version}; this client has {self.registry.version}")
        route = self.registry.route(plan.route_id)
        if route.source_asset_id != plan.source_asset_id or route.destination_asset_id != plan.destination_asset_id:
            raise BridgeError(f"Plan assets do not match configured route {route.id}")
        if route.availability != "active":
            raise RouteUnavailableError(f"Route is not executable: {route.id}")
        return route

    def _plan_route(self, plan: Plan, protocol: str) -> Route:
        """Re-resolve a caller-supplied ``Plan``'s route by id (never trust plan-carried addresses)."""
        if plan.registry_version != self.registry.version:
            raise RegistryVersionMismatchError(
                f"Plan uses registry {plan.registry_version}; this client has {self.registry.version}")
        try:
            route = self.registry.route(plan.route_id)
        except RouteNotFoundError as exc:
            raise RouteUnavailableError(f"Plan route {plan.route_id} is not in registry {self.registry.version}") from exc
        if route.protocol != protocol:
            raise RouteUnavailableError(f"{route.id} is a {route.protocol} route, not a {protocol} one")
        if route.availability != "active":
            raise RouteUnavailableError(f"Route is not executable ({route.availability}): {route.id}")
        return route

    def _plan_sender(self, plan: Plan, *, require_signer: bool) -> str:
        """``plan.sender`` as a checksummed EVM address, bound to the connected account when there is one."""
        Web3 = _web3().Web3
        sender = plan.sender
        if not isinstance(sender, str) or not Web3.is_address(sender) or Web3.to_checksum_address(sender) != sender:
            raise BridgeError(f"Plan sender must be a checksummed EVM address; got {sender!r}")
        connected = self.conn.require_address() if require_signer else self.conn.address
        if connected is not None and sender != connected:                 # the rule EvmCall.send() applies
            raise ConfigurationError(f"Prepared sender {sender} does not match connected account {connected}")
        return sender

    def _assert_plan_matches(self, plan: Plan, route: Route, *, sender: str, recipient: str, amount_atomic: int,
                             mint_mode: str) -> None:
        """The plan must be exactly what this module would have prepared for the same transfer."""
        rebuilt = _plan_for(self.registry, route, amount_atomic=amount_atomic, recipient=recipient, sender=sender,
                            mint_mode=mint_mode)
        for field in fields(Plan):
            mine, theirs = getattr(rebuilt, field.name), getattr(plan, field.name)
            if mine != theirs:
                raise BridgeError(f"plan does not match the requested transfer: {field.name} is {theirs!r} "
                                  f"but this transfer prepares {mine!r}")

    def _from_plan(self, plan: Plan, protocol: str, *, recipient: str | None, amount: Any, amount_atomic: int | None,
                   mint_mode: str | None, require_signer: bool) -> tuple[Route, str, str, int, str]:
        """Validate ``plan=`` and return ``(route, sender, recipient, amount_atomic, mint_mode)``.

        Explicit ``recipient``/``amount``/``mint_mode`` arguments override the plan's own values and are
        then caught by the field-by-field equality check, so a plan can never silently disagree with
        the call that carries it.
        """
        route = self._plan_route(plan, protocol)
        sender = self._plan_sender(plan, require_signer=require_signer)
        recipient = plan.recipient if recipient is None else recipient
        mint_mode = plan.mint_mode if mint_mode is None else mint_mode
        if amount is None and amount_atomic is None:
            amount_atomic = plan.amount_atomic
        atomic = self._amount_atomic(route, amount, amount_atomic)
        self._assert_plan_matches(plan, route, sender=sender, recipient=recipient, amount_atomic=atomic,
                                  mint_mode=mint_mode)
        return route, sender, recipient, atomic, mint_mode

    def assert_chain(self, route: Route) -> None:
        expected = int(route.metadata["sourceChainId"])
        actual = self.conn.chain_id
        if actual != expected:
            raise ChainMismatchError(f"EVM connection is on chain {actual}; expected {expected} for {route.id}")

    def _recipient_bytes32(self, route: Route, recipient: str) -> bytes:
        destination = self.registry.asset(route.destination_asset_id)
        if destination.address_regex and not re.fullmatch(destination.address_regex, recipient):
            raise InvalidRecipientError(f"Recipient does not match the {destination.chain_id} address format: {recipient}")
        return encoding.aleo_address_to_bytes32(recipient)

    def _amount_atomic(self, route: Route, amount: Any, amount_atomic: int | None) -> int:
        source = self.registry.asset(route.source_asset_id)
        destination = self.registry.asset(route.destination_asset_id)
        atomic = resolve_amount(amount=amount, amount_atomic=amount_atomic, decimals=source.decimals)
        if atomic <= 0:
            raise InvalidAmountError("Amount must be positive")
        parse_decimal_amount(format_decimal_amount(atomic, source.decimals), destination.decimals)  # precision on both sides
        return atomic

    def _owner(self, sender: str | None) -> str | None:
        if sender is None:
            return self.conn.address
        return _web3().Web3.to_checksum_address(sender)

    # -- contracts ----------------------------------------------------------------------------

    def _contract(self, address: str, abi: list) -> Any:
        return self.conn.w3.eth.contract(address=_web3().Web3.to_checksum_address(address), abi=abi)

    def _erc20(self, address: str) -> Any:
        return self._contract(address, ERC20_ABI)

    def _native_fee(self, amount_wei: int) -> Fee:
        native = [a for a in self.registry.assets(chain=self.chain.id) if a.kind == "native"]
        asset_id = native[0].id if native else f"{self.chain.id}/{self.chain.native_symbol.lower()}"
        decimals = native[0].decimals if native else 18   # only an unlisted chain falls back to the EVM default
        return Fee(kind="network", chain_id=self.chain.id, asset_id=asset_id,
                   amount=format_decimal_amount(amount_wei, decimals), estimated=True)

    # -- Hyperlane quote ----------------------------------------------------------------------

    def _metadata_address(self, meta: Mapping[str, Any], key: str, route_id: str) -> str:
        """Checksum a metadata address field; any malformed value is a ``ConfigurationError``, never
        a raw ``ValueError``/``KeyError`` (a missing key reads as ``None`` via ``.get``, which also fails here)."""
        value = meta.get(key)
        try:
            return _web3().Web3.to_checksum_address(str(value))
        except (ValueError, TypeError) as exc:
            raise ConfigurationError(f"Hyperlane route metadata {key!r} is not a valid address ({route_id}): {value!r}") from exc

    def _hyperlane_metadata(self, route: Route) -> _HyperlaneRouteMetadata:
        """Brief §3.1 route metadata validator (mirrors veil ``protocols/hyperlane/evm.ts`` ``routeMetadata``).

        Called by every path that is about to touch a Hyperlane contract (``_quote_hyperlane``, and
        transitively ``transfer_remote``'s step builder through the ``_HyperlaneQuote`` it returns) so
        no unvalidated address or domain from the registry ever reaches an RPC call.
        """
        if route is None or route.protocol != "hyperlane" or route.availability != "active":
            raise RouteUnavailableError(f"Hyperlane route is not executable: {getattr(route, 'id', route)!r}")
        meta = route.metadata
        router = self._metadata_address(meta, "routerAddress", route.id)
        mailbox = self._metadata_address(meta, "mailboxAddress", route.id)
        igp = self._metadata_address(meta, "interchainGasPaymaster", route.id)
        ism = self._metadata_address(meta, "interchainSecurityModule", route.id)
        source_chain_id = meta.get("sourceChainId")
        if isinstance(source_chain_id, bool) or not isinstance(source_chain_id, int) or source_chain_id <= 0:
            raise ConfigurationError(
                f"Hyperlane route metadata sourceChainId must be a positive int ({route.id}): {source_chain_id!r}")
        destination_domain = meta.get("destinationDomain")
        if isinstance(destination_domain, bool) or not isinstance(destination_domain, int) \
                or not (0 <= destination_domain <= 2**32 - 1):
            raise ConfigurationError(
                f"Hyperlane route metadata destinationDomain must be a uint32 ({route.id}): {destination_domain!r}")
        router_type = meta.get("routerType")
        if router_type not in ("native", "collateral"):
            raise ConfigurationError(
                f"Hyperlane route metadata routerType must be native or collateral ({route.id}): {router_type!r}")
        token = self._metadata_address(meta, "tokenAddress", route.id) if router_type == "collateral" else None
        destination_router = meta.get("destinationRouter")
        if not isinstance(destination_router, str) or not destination_router.strip():
            raise ConfigurationError(f"Hyperlane route metadata destinationRouter must be non-empty ({route.id})")
        registry_commit = meta.get("registryCommit")
        if not isinstance(registry_commit, str) or not _REGISTRY_COMMIT_RE.fullmatch(registry_commit):
            raise ConfigurationError(
                f"Hyperlane route metadata registryCommit must be 40 hex chars ({route.id}): {registry_commit!r}")
        return _HyperlaneRouteMetadata(
            router=router, router_type=router_type, token=token, mailbox=mailbox,
            interchain_gas_paymaster=igp, interchain_security_module=ism, source_chain_id=source_chain_id,
            destination_domain=destination_domain, destination_router=destination_router,
            registry_commit=registry_commit, requires_approval_reset=meta.get("requiresApprovalReset") is True)

    def _quote_hyperlane(self, route: Route, recipient_bytes32: bytes, amount_atomic: int, owner: str | None) -> _HyperlaneQuote:
        """Brief §3.1: chain assert → metadata validation → quoteTransferRemote → native/collateral split → allowance."""
        self.assert_chain(route)
        meta = self._hyperlane_metadata(route)
        Web3 = _web3().Web3
        quotes = self._contract(meta.router, WARP_ROUTE_ABI).functions.quoteTransferRemote(
            meta.destination_domain, recipient_bytes32, amount_atomic).call()
        native_value = sum(int(q[1]) for q in quotes if Web3.to_checksum_address(q[0]) == ZERO_ADDRESS)
        if meta.router_type == "native":
            if native_value < amount_atomic:
                raise BridgeError("Native Hyperlane quote does not cover the transfer amount")
            return _HyperlaneQuote(meta.router, "native", None, meta.destination_domain, recipient_bytes32,
                                   amount_atomic, native_value, native_value - amount_atomic, 0, None, False)
        token_amount = sum(int(q[1]) for q in quotes if Web3.to_checksum_address(q[0]) == meta.token)
        if token_amount < amount_atomic:
            raise BridgeError("Collateral Hyperlane quote does not cover the transfer amount")
        allowance = int(self._erc20(meta.token).functions.allowance(owner, meta.router).call()) if owner else None
        return _HyperlaneQuote(meta.router, "collateral", meta.token, meta.destination_domain, recipient_bytes32,
                               amount_atomic, native_value, native_value, token_amount, allowance,
                               meta.requires_approval_reset)

    def quote_transfer_remote(self, asset: Any = None, recipient: str | None = None, *, amount: Any = None,
                              amount_atomic: int | None = None, route: Route | None = None, sender: str | None = None,
                              plan: Plan | None = None) -> EvmHyperlaneQuote:
        """Quote an Ethereum → Aleo Hyperlane transfer without signing.

        Native routes (ETH): ``msg.value`` carries the asset and the relayer fee, so
        ``native_fee_atomic = native_value_atomic - amount``. Collateral routes (WBTC, USDT):
        ``msg.value`` is fee only and ``approval_required`` reflects the router's ERC-20
        allowance for ``sender`` (or the connection's account); it is ``None`` when no account is known.

        ``plan=`` re-quotes a plan prepared earlier: it supplies the route, sender, recipient and
        amount, and is validated against the live registry. It is mutually exclusive with
        ``asset=``/``route=``/``sender=``.
        """
        if plan is None and recipient is None:
            raise InvalidRecipientError("recipient is required when no plan is given")
        if plan is not None:
            if asset is not None or route is not None or sender is not None:
                raise ValueError("Pass plan= or asset=/route=/sender=, not both")
            route, owner, recipient, atomic, _ = self._from_plan(
                plan, "hyperlane", recipient=recipient, amount=amount, amount_atomic=amount_atomic,
                mint_mode=None, require_signer=False)
            recipient32 = self._recipient_bytes32(route, recipient)
        else:
            if asset is None and route is None:
                raise ValueError("quote_transfer_remote needs asset=, route= or plan=")
            route = route or self._hyperlane_route(self._asset(asset))
            if route.protocol != "hyperlane":
                raise BridgeError(f"{route.id} is not a Hyperlane route; use quote_deposit_usdc for xReserve")
            atomic = self._amount_atomic(route, amount, amount_atomic)
            recipient32 = self._recipient_bytes32(route, recipient)
            owner = self._owner(sender)
        q = self._quote_hyperlane(route, recipient32, atomic, owner)
        destination = self.registry.asset(route.destination_asset_id)
        plan = _plan_for(self.registry, route, amount_atomic=atomic, recipient=recipient, sender=owner)
        approval_required = None if q.allowance_atomic is None else q.allowance_atomic < q.token_amount_atomic
        return EvmHyperlaneQuote(kind="evm-hyperlane", plan=plan, fees=(self._native_fee(q.native_fee_atomic),),
                                 amount_out=format_decimal_amount(atomic, destination.decimals),
                                 recipient_bytes32=recipient32, native_value_atomic=q.native_value_atomic,
                                 native_fee_atomic=q.native_fee_atomic, approval_required=approval_required)

    # -- xReserve quote -------------------------------------------------------------------------

    def _metadata_digits(self, meta: Mapping[str, Any], key: str, route_id: str) -> int:
        """A digit-string atomic-amount metadata field as ``int``; never bool/float/junk."""
        value = meta.get(key)
        if not isinstance(value, str) or not _DIGIT_STRING_RE.fullmatch(value):
            raise ConfigurationError(f"xReserve route metadata {key!r} must be a digit string ({route_id}): {value!r}")
        return int(value)

    def _metadata_aleo_program(self, meta: Mapping[str, Any], key: str, route_id: str) -> str:
        value = meta.get(key)
        if not isinstance(value, str) or not value.endswith(".aleo") or value == ".aleo":
            raise ConfigurationError(f"xReserve route metadata {key!r} must be an .aleo program name ({route_id}): {value!r}")
        return value

    def _xreserve_metadata(self, route: Route) -> _XReserveRouteMetadata:
        """Brief §3.2 route metadata validator (mirrors veil ``protocols/xreserve/evmToAleo.ts`` ``routeMetadata``).

        Called by every path that is about to touch the xReserve contract or an Aleo program address
        (``_quote_xreserve``, ``_xreserve_recipient_bytes32``, and Task 6's deposit execute) so no
        unvalidated address, domain, fee, or program name from the registry ever reaches an RPC call.
        """
        if route is None or route.protocol != "xreserve" or route.availability != "active":
            raise RouteUnavailableError(f"xReserve route is not executable: {getattr(route, 'id', route)!r}")
        source_chain = self.registry.chain(self.registry.asset(route.source_asset_id).chain_id)
        if source_chain.family != "evm":
            raise ConfigurationError(f"xReserve route source chain must be an EVM chain ({route.id}): {source_chain.id!r}")
        expected_aleo_chain = "aleo-testnet" if self.network == "testnet" else "aleo"
        destination_chain_id = self.registry.asset(route.destination_asset_id).chain_id
        if destination_chain_id != expected_aleo_chain:
            raise ConfigurationError(
                f"xReserve route destination chain must be {expected_aleo_chain!r} ({route.id}): {destination_chain_id!r}")
        meta = route.metadata
        xreserve_contract = self._metadata_address(meta, "xReserveContract", route.id)
        source_chain_id = meta.get("sourceChainId")
        if isinstance(source_chain_id, bool) or not isinstance(source_chain_id, int) or source_chain_id <= 0:
            raise ConfigurationError(
                f"xReserve route metadata sourceChainId must be a positive int ({route.id}): {source_chain_id!r}")
        source_domain = meta.get("sourceDomain")
        if isinstance(source_domain, bool) or not isinstance(source_domain, int) or source_domain < 0:
            raise ConfigurationError(
                f"xReserve route metadata sourceDomain must be a non-negative int ({route.id}): {source_domain!r}")
        remote_domain = meta.get("remoteDomain")
        if isinstance(remote_domain, bool) or not isinstance(remote_domain, int) or remote_domain < 0:
            raise ConfigurationError(
                f"xReserve route metadata remoteDomain must be a non-negative int ({route.id}): {remote_domain!r}")
        remote_token_bytes32_raw = meta.get("remoteTokenBytes32")
        if not isinstance(remote_token_bytes32_raw, str):
            raise ConfigurationError(
                f"xReserve route metadata remoteTokenBytes32 must be a hex string ({route.id}): {remote_token_bytes32_raw!r}")
        hex_text = remote_token_bytes32_raw[2:] if remote_token_bytes32_raw[:2] in ("0x", "0X") else remote_token_bytes32_raw
        try:
            remote_token_bytes32 = bytes.fromhex(hex_text)
        except ValueError as exc:
            raise ConfigurationError(
                f"xReserve route metadata remoteTokenBytes32 is not valid hex ({route.id}): {remote_token_bytes32_raw!r}") from exc
        if len(remote_token_bytes32) != 32:
            raise ConfigurationError(
                f"xReserve route metadata remoteTokenBytes32 must be exactly 32 bytes ({route.id}): {remote_token_bytes32_raw!r}")
        minimum_amount_atomic = self._metadata_digits(meta, "minimumAmountAtomic", route.id)
        withdrawal_fee_atomic = self._metadata_digits(meta, "withdrawalFeeAtomic", route.id)
        max_fee_atomic = self._metadata_digits(meta, "maxFeeAtomic", route.id)
        bridge_program = self._metadata_aleo_program(meta, "bridgeProgram", route.id)
        wrapper_program = self._metadata_aleo_program(meta, "wrapperProgram", route.id)
        remote_token = self._metadata_aleo_program(meta, "remoteToken", route.id)
        attestation_base_url = meta.get("attestationBaseUrl")
        if not isinstance(attestation_base_url, str) or not attestation_base_url.startswith("https://"):
            raise ConfigurationError(
                f"xReserve route metadata attestationBaseUrl must start with https:// ({route.id}): {attestation_base_url!r}")
        return _XReserveRouteMetadata(
            xreserve_contract=xreserve_contract, source_chain_id=source_chain_id, source_domain=source_domain,
            remote_domain=remote_domain, remote_token_bytes32=remote_token_bytes32,
            minimum_amount_atomic=minimum_amount_atomic, withdrawal_fee_atomic=withdrawal_fee_atomic,
            max_fee_atomic=max_fee_atomic, bridge_program=bridge_program, wrapper_program=wrapper_program,
            remote_token=remote_token, attestation_base_url=attestation_base_url)

    def _xreserve_recipient_bytes32(self, route: Route, meta: _XReserveRouteMetadata, recipient: str,
                                    mint_mode: str) -> bytes:
        """Invariant 7: private deposits are addressed to the wrapper program's account address."""
        self._recipient_bytes32(route, recipient)                       # validates the intended recipient
        if mint_mode == "private":
            return encoding.aleo_address_to_bytes32(encoding.aleo_program_address(meta.wrapper_program, self.network))
        return encoding.aleo_address_to_bytes32(recipient)

    def _quote_xreserve(self, route: Route, recipient: str, amount_atomic: int, owner: str | None,
                        mint_mode: str, secret_nonce: str) -> _XReserveQuote:
        """Brief §3.2 quote: chain assert → metadata validation → minimum → hook data → wire recipient →
        balanceOf/allowance."""
        if mint_mode not in ("public", "record", "private"):
            raise BridgeError(f"mint_mode must be public, record or private; got {mint_mode!r}")
        self.assert_chain(route)
        meta = self._xreserve_metadata(route)
        if amount_atomic < meta.minimum_amount_atomic:
            raise InvalidAmountError(f"xReserve minimum deposit is {meta.minimum_amount_atomic} atomic units")
        if owner is None:
            raise ConfigurationError("xReserve quotes read the depositor's balance: pass sender= or configure a signer")
        source = self.registry.asset(route.source_asset_id)
        if source.locator is None or source.locator.kind != "evm-contract":
            raise RouteUnavailableError(f"xReserve source token contract is missing: {route.id}")
        token = _web3().Web3.to_checksum_address(source.locator.value)
        hook_data = encoding.xreserve_hook_data(mint_mode, recipient, self.network, secret_nonce)
        remote_recipient = self._xreserve_recipient_bytes32(route, meta, recipient, mint_mode)
        erc20 = self._erc20(token)
        balance = int(erc20.functions.balanceOf(owner).call())
        allowance = int(erc20.functions.allowance(owner, meta.xreserve_contract).call())
        if balance < amount_atomic:
            raise InsufficientBalanceError(f"Insufficient {source.symbol} balance: {balance} < {amount_atomic} atomic units")
        return _XReserveQuote(
            xreserve_contract=meta.xreserve_contract, token=token, source_chain_id=meta.source_chain_id,
            source_domain=meta.source_domain, remote_domain=meta.remote_domain,
            remote_token_bytes32=meta.remote_token_bytes32,
            remote_recipient_bytes32=remote_recipient, amount_atomic=amount_atomic,
            max_fee_atomic=meta.max_fee_atomic, hook_data=hook_data,
            balance_atomic=balance, allowance_atomic=allowance,
            bridge_program=meta.bridge_program, wrapper_program=meta.wrapper_program)

    def quote_deposit_usdc(self, recipient: str | None = None, *, amount: Any = None, amount_atomic: int | None = None,
                           mint_mode: str | None = None, secret_nonce: str = "0scalar",
                           sender: str | None = None, route: Route | None = None,
                           plan: Plan | None = None) -> EvmXReserveQuote:
        """Quote a USDC → USDCx xReserve deposit without signing.

        Checks the 2 USDC minimum, derives the 65-byte hook (``public``/``record``/``private``;
        private commits ``recipient`` with ``secret_nonce`` via BHP256) and the wire recipient
        (the shielded wrapper program's address for ``private``), and reads the depositor's
        USDC balance and xReserve allowance. ``secret_nonce`` is never stored by the SDK.
        ``mint_mode`` defaults to ``plan.mint_mode`` when a plan is given, else ``"public"``.

        ``plan=`` re-quotes a plan prepared earlier: it supplies the route, sender, recipient, amount
        and mint mode, and is validated against the live registry. It is mutually exclusive with
        ``route=``/``sender=``.
        """
        if plan is None and recipient is None:
            raise InvalidRecipientError("recipient is required when no plan is given")
        if plan is not None:
            if route is not None or sender is not None:
                raise ValueError("Pass plan= or route=/sender=, not both")
            route, owner, recipient, atomic, mint_mode = self._from_plan(
                plan, "xreserve", recipient=recipient, amount=amount, amount_atomic=amount_atomic,
                mint_mode=mint_mode, require_signer=False)
        else:
            route = route or self._xreserve_route()
            mint_mode = "public" if mint_mode is None else mint_mode
            atomic = self._amount_atomic(route, amount, amount_atomic)
            owner = self._owner(sender)
        q = self._quote_xreserve(route, recipient, atomic, owner, mint_mode, secret_nonce)
        destination = self.registry.asset(route.destination_asset_id)
        plan = _plan_for(self.registry, route, amount_atomic=atomic, recipient=recipient, sender=owner, mint_mode=mint_mode)
        return EvmXReserveQuote(kind="evm-xreserve", plan=plan, fees=(),
                                amount_out=format_decimal_amount(atomic, destination.decimals),
                                hook_data=q.hook_data, remote_recipient_bytes32=q.remote_recipient_bytes32,
                                balance_atomic=q.balance_atomic, allowance_atomic=q.allowance_atomic,
                                approval_required=q.allowance_atomic < atomic, max_fee_atomic=q.max_fee_atomic)

    # -- Hyperlane execute --------------------------------------------------------------------

    def _message_id_from_receipt(self, route: Route, receipt: Any) -> str | None:
        """Hyperlane Mailbox ``DispatchId(bytes32 indexed messageId)`` from a confirmed receipt; ``None`` if absent."""
        from web3.logs import DISCARD

        Web3 = _web3().Web3
        address = self._hyperlane_metadata(route).mailbox
        mailbox = self._contract(address, MAILBOX_ABI)
        # process_receipt decodes by topic alone: another contract's DispatchId(bytes32) would
        # otherwise be read as this transfer's message id, so filter on the emitting address first.
        events = [ev for ev in mailbox.events.DispatchId().process_receipt(receipt, errors=DISCARD)
                  if Web3.to_checksum_address(ev["address"]) == address]
        if not events:
            return None
        return Web3.to_hex(events[0]["args"]["messageId"])           # veil messageIdFromReceipt: first match wins

    @staticmethod
    def _hyperlane_protocol_state(route: Route, *, recipient_bytes32: bytes, destination_domain: int,
                                  native_value_atomic: int, amount_atomic: int, approval_tx_ids: list[str],
                                  sender: str | None, message_id: str | None = None,
                                  source_nonce: int | str | None = None) -> dict[str, Any]:
        state: dict[str, Any] = {
            "routeId": route.id, "approvalTxIds": list(approval_tx_ids), "sourceSender": sender,
            "recipientBytes32": "0x" + recipient_bytes32.hex(), "destinationDomain": destination_domain,
            "nativeValueAtomic": str(native_value_atomic), "amountAtomic": str(amount_atomic),
        }
        if message_id is not None:
            state["messageId"] = message_id
        if source_nonce is not None:
            state["sourceNonce"] = str(source_nonce)
        return state

    def _hyperlane_result(self, route: Route, q: "_HyperlaneQuote", outcome: EvmOutcome) -> DispatchReceipt:
        approvals = list(outcome.approval_tx_ids)
        if outcome.status == "CONFIRMED":
            message_id = self._message_id_from_receipt(route, outcome.receipt)
            status, rid = Status.DELIVERY_PENDING, message_id or outcome.source_tx_id
        else:
            message_id, status = None, Status(outcome.status)
            rid = outcome.source_tx_id or approvals[-1]
        state = self._hyperlane_protocol_state(
            route, recipient_bytes32=q.recipient_bytes32, destination_domain=q.destination_domain,
            native_value_atomic=q.native_value_atomic, amount_atomic=q.amount_atomic,
            approval_tx_ids=approvals, sender=outcome.sender, message_id=message_id,
            source_nonce=outcome.source_nonce)
        receipt = Receipt(id=rid, protocol="hyperlane", status=status, source_tx_id=outcome.source_tx_id, protocol_state=state)
        return DispatchReceipt(transaction_id=outcome.source_tx_id or approvals[-1], route_id=route.id,
                               message_id=message_id, amount_atomic=q.amount_atomic, receipt=receipt)

    def transfer_remote(self, asset: Any = None, recipient: str | None = None, *, amount: Any = None,
                        amount_atomic: int | None = None, plan: Plan | None = None) -> EvmCall[DispatchReceipt]:
        """Send ETH, WBTC or USDT to Aleo through its Hyperlane Warp Route.

        Re-quotes ``quoteTransferRemote`` at send time. Collateral routes approve exactly the
        quoted token amount only when the allowance is short (USDT: a non-zero allowance is
        reset to 0 first). Native ETH sends amount + fee as ``msg.value``; collateral routes
        send the fee only. Each hash is checkpointed before polling; a timeout returns a
        pending ``DispatchReceipt``. The message id comes from the Mailbox ``DispatchId`` log.

        ``plan=`` executes a plan prepared earlier (typically ``quote.plan``): the route is
        re-resolved by id against the live registry, the sender must be the connected account, and
        the plan must equal what this call would have prepared itself. Mutually exclusive with ``asset=``.
        """
        if plan is None and recipient is None:
            raise InvalidRecipientError("recipient is required when no plan is given")
        if plan is not None:
            if asset is not None:
                raise ValueError("Pass plan= or asset=, not both")
            route, sender, recipient, atomic, _ = self._from_plan(
                plan, "hyperlane", recipient=recipient, amount=amount, amount_atomic=amount_atomic,
                mint_mode=None, require_signer=True)
        else:
            if asset is None:
                raise ValueError("transfer_remote needs asset= or plan=")
            route = self._hyperlane_route(self._asset(asset))
            sender = self.conn.require_address()
            atomic = self._amount_atomic(route, amount, amount_atomic)
            plan = _plan_for(self.registry, route, amount_atomic=atomic, recipient=recipient, sender=sender)
        recipient32 = self._recipient_bytes32(route, recipient)
        latest: dict[str, _HyperlaneQuote] = {}

        def steps(owner: str) -> list[EvmStep]:
            q = self._quote_hyperlane(route, recipient32, atomic, owner)      # last responsible moment
            latest["q"] = q
            out: list[EvmStep] = []
            if q.router_type == "collateral" and (q.allowance_atomic or 0) < q.token_amount_atomic:
                token = self._erc20(q.token)
                if (q.allowance_atomic or 0) > 0 and q.requires_approval_reset:
                    out.append(EvmStep("approve", q.token, token.encode_abi("approve", args=[q.router, 0])))
                out.append(EvmStep("approve", q.token, token.encode_abi("approve", args=[q.router, q.token_amount_atomic])))
            warp = self._contract(q.router, WARP_ROUTE_ABI)
            out.append(EvmStep("main", q.router,
                               warp.encode_abi("transferRemote", args=[q.destination_domain, recipient32, atomic]),
                               q.native_value_atomic))
            return out

        def finish(outcome: EvmOutcome) -> DispatchReceipt:
            return self._hyperlane_result(route, latest["q"], outcome)

        return EvmCall(self.conn, plan=plan, registry=self.registry, steps=steps, finish=finish,
                       store=self.bridge.checkpoints)

    # -- xReserve execute -----------------------------------------------------------------------

    @staticmethod
    def _xreserve_protocol_state(route: Route, q: _XReserveQuote, *, approval_tx_ids: list[str], sender: str | None,
                                 mint_mode: str, intended_recipient: str,
                                 source_nonce: int | str | None = None) -> dict[str, Any]:
        state: dict[str, Any] = {
            "routeId": route.id, "approvalTxIds": list(approval_tx_ids), "sourceSender": sender,
            "mintMode": mint_mode, "intendedRecipient": intended_recipient,
            "xReserveContract": q.xreserve_contract, "tokenAddress": q.token, "sourceChainId": q.source_chain_id,
            "remoteDomain": q.remote_domain, "remoteRecipientBytes32": "0x" + q.remote_recipient_bytes32.hex(),
            "hookData": "0x" + q.hook_data.hex(), "amountAtomic": str(q.amount_atomic), "maxFeeAtomic": str(q.max_fee_atomic),
        }
        if source_nonce is not None:
            state["sourceNonce"] = str(source_nonce)
        return state

    def _confirmed_deposit_receipt(self, route: Route, q: _XReserveQuote, *, owner: str, approval_tx_ids: list[str],
                                   source_tx_id: str, receipt: Any, mint_mode: str, intended_recipient: str) -> Receipt:
        """Brief §3.2 confirm: find the xReserve ``DepositedToRemote`` log, re-verify every field, derive nonce/payload/hash."""
        from web3.logs import DISCARD

        Web3 = _web3().Web3
        # Kept even though send() already asserted success: _recover_xreserve_from_history reaches
        # this with receipts nothing has checked, so the revert test must live here too.
        if int(receipt["status"]) == 0:
            raise BridgeError(f"EVM transaction reverted: {source_tx_id}")
        xreserve = self._contract(q.xreserve_contract, XRESERVE_ABI)
        events = [ev for ev in xreserve.events.DepositedToRemote().process_receipt(receipt, errors=DISCARD)
                  if Web3.to_checksum_address(ev["address"]) == q.xreserve_contract]
        if not events:
            raise BridgeError("Confirmed receipt does not contain a valid DepositedToRemote event")

        def matches(args: Mapping[str, Any]) -> bool:
            return (Web3.to_checksum_address(args["localToken"]) == q.token
                    and Web3.to_checksum_address(args["localDepositor"]) == owner
                    and int(args["value"]) == q.amount_atomic
                    and int(args["remoteDomain"]) == q.remote_domain
                    and bytes(args["remoteRecipient"]) == q.remote_recipient_bytes32
                    and bytes(args["remoteToken"]) == q.remote_token_bytes32
                    and int(args["maxFee"]) == q.max_fee_atomic
                    and bytes(args["hookData"]) == q.hook_data)

        # One transaction can batch several accounts' deposits, so take OUR event rather than the
        # last one: every one of the eight canonical fields has to match for it to be ours.
        ev = next((e for e in events if matches(e["args"])), None)
        if ev is None:
            raise BridgeError("DepositedToRemote event does not match the prepared transfer")
        a = ev["args"]
        log_index = int(ev["logIndex"])
        if log_index < 0:
            raise BridgeError("DepositedToRemote log index is missing or invalid")
        # xReserve identifies a deposit by (source domain, tx hash, log index); the ordered payload is
        # what Circle signs, so its keccak is the only safe attestation lookup key.
        nonce = encoding.xreserve_deposit_nonce(q.source_domain, bytes.fromhex(source_tx_id[2:]), log_index)
        payload = encoding.xreserve_deposit_payload(
            amount=int(a["value"]), remote_domain=int(a["remoteDomain"]), remote_token=bytes(a["remoteToken"]),
            remote_recipient=bytes(a["remoteRecipient"]), local_token=Web3.to_checksum_address(a["localToken"]),
            depositor=Web3.to_checksum_address(a["localDepositor"]), max_fee=int(a["maxFee"]), nonce=nonce,
            hook_data=bytes(a["hookData"]))
        message_hash = "0x" + encoding.xreserve_message_hash(payload).hex()
        state = self._xreserve_protocol_state(route, q, approval_tx_ids=approval_tx_ids, sender=owner,
                                              mint_mode=mint_mode, intended_recipient=intended_recipient)
        state.update({"sourceDomain": q.source_domain, "remoteDomain": q.remote_domain, "depositLogIndex": log_index,
                      "nonce": "0x" + nonce.hex(), "payload": "0x" + payload.hex(), "messageHash": message_hash,
                      "bridgeProgram": q.bridge_program, "wrapperProgram": q.wrapper_program})
        return Receipt(id=message_hash, protocol="xreserve", status=Status.ATTESTATION_PENDING,
                       source_tx_id=source_tx_id, protocol_state=state)

    def _xreserve_result(self, route: Route, q: _XReserveQuote, outcome: EvmOutcome, *, mint_mode: str,
                         intended_recipient: str) -> DepositReceipt:
        approvals = list(outcome.approval_tx_ids)
        if outcome.status == "CONFIRMED":
            receipt = self._confirmed_deposit_receipt(route, q, owner=outcome.sender, approval_tx_ids=approvals,
                                                      source_tx_id=outcome.source_tx_id, receipt=outcome.receipt,
                                                      mint_mode=mint_mode, intended_recipient=intended_recipient)
            return DepositReceipt(transaction_id=outcome.source_tx_id, route_id=route.id, message_hash=receipt.id,
                                  nonce=receipt.protocol_state["nonce"], receipt=receipt)
        rid = outcome.source_tx_id or approvals[-1]
        receipt = Receipt(id=rid, protocol="xreserve", status=Status(outcome.status), source_tx_id=outcome.source_tx_id,
                          protocol_state=self._xreserve_protocol_state(route, q, approval_tx_ids=approvals, sender=outcome.sender,
                                                                       mint_mode=mint_mode, intended_recipient=intended_recipient,
                                                                       source_nonce=outcome.source_nonce))
        return DepositReceipt(transaction_id=rid, route_id=route.id, message_hash="", nonce="", receipt=receipt)

    def deposit_usdc(self, recipient: str | None = None, *, amount: Any = None, amount_atomic: int | None = None,
                     mint_mode: str | None = None, secret_nonce: str = "0scalar",
                     plan: Plan | None = None) -> EvmCall[DepositReceipt]:
        """Deposit USDC into Circle xReserve for USDCx on Aleo (minimum 2 USDC; irreversible once confirmed).

        ``mint_mode``: ``public`` (public USDCx balance), ``record`` (protocol-minted private
        record), or ``private`` (deposit addressed to the shielded wrapper program; you must later
        run ``bridge.xreserve.private_mint`` / plan 4's ``complete`` with the same ``secret_nonce``,
        which the SDK never stores). Approves exactly the amount only when the allowance is
        short, then ``depositToRemote`` with no ``msg.value``. The confirmed ``DepositReceipt``
        carries Circle's message hash (receipt id) and the deposit nonce. ``mint_mode`` defaults to
        ``plan.mint_mode`` when a plan is given, else ``"public"``.

        ``plan=`` executes a plan prepared earlier (typically ``quote.plan``): the route is
        re-resolved by id against the live registry, the sender must be the connected account, and
        the plan must equal what this call would have prepared itself. ``secret_nonce`` is never
        part of a plan, so a private deposit must still pass the same one it was quoted with.
        """
        if plan is None and recipient is None:
            raise InvalidRecipientError("recipient is required when no plan is given")
        if plan is not None:
            route, sender, recipient, atomic, mint_mode = self._from_plan(
                plan, "xreserve", recipient=recipient, amount=amount, amount_atomic=amount_atomic,
                mint_mode=mint_mode, require_signer=True)
        else:
            route = self._xreserve_route()
            mint_mode = "public" if mint_mode is None else mint_mode
            sender = self.conn.require_address()
            atomic = self._amount_atomic(route, amount, amount_atomic)
            plan = _plan_for(self.registry, route, amount_atomic=atomic, recipient=recipient, sender=sender, mint_mode=mint_mode)
        latest: dict[str, _XReserveQuote] = {}

        def steps(owner: str) -> list[EvmStep]:
            q = self._quote_xreserve(route, recipient, atomic, owner, mint_mode, secret_nonce)   # fresh balance/allowance
            latest["q"] = q
            out: list[EvmStep] = []
            if q.allowance_atomic < atomic:
                out.append(EvmStep("approve", q.token, self._erc20(q.token).encode_abi("approve", args=[q.xreserve_contract, atomic])))
            xreserve = self._contract(q.xreserve_contract, XRESERVE_ABI)
            out.append(EvmStep("main", q.xreserve_contract, xreserve.encode_abi(
                "depositToRemote", args=[atomic, q.remote_domain, q.remote_recipient_bytes32, q.token, q.max_fee_atomic, q.hook_data]), 0))
            return out

        def finish(outcome: EvmOutcome) -> DepositReceipt:
            return self._xreserve_result(route, latest["q"], outcome, mint_mode=mint_mode, intended_recipient=recipient)

        return EvmCall(self.conn, plan=plan, registry=self.registry, steps=steps, finish=finish, store=self.bridge.checkpoints)

    # -- status ---------------------------------------------------------------------------------

    @staticmethod
    def _require_hash(value: Any, what: str) -> str:
        if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
            raise CheckpointInvalidError(f"Receipt is missing a valid {what}")
        return value

    @staticmethod
    def _failed(receipt: Receipt, key: str, message: str) -> Receipt:
        return receipt.replace(status=Status.FAILED, next_action=None,
                               protocol_state={**receipt.protocol_state, key: message})

    @staticmethod
    def _dropped(receipt: Receipt, message: str) -> Receipt:
        """Terminal ``EXPIRED``: the checkpointed transaction can never mine, and moved nothing."""
        return receipt.replace(status=Status.EXPIRED, next_action=None,
                               protocol_state={**receipt.protocol_state, "sourceError": message, "dropped": True})

    def _dropped_error(self, receipt: Receipt, tx_hash: str) -> str | None:
        """The "dropped or replaced" message when *tx_hash* can never mine; ``None`` otherwise.

        Three facts have to line up, and the cheap ones are checked first: the receipt records the
        nonce the transaction was broadcast at (older checkpoints do not — they simply keep
        waiting), the node no longer knows the transaction at all (``TransactionNotFound``, not
        merely "no receipt yet"), and the sender's ``latest`` account nonce has moved past it, so
        something else consumed that nonce. A transaction the node still knows is pending, however
        long it has been pending, is left alone.
        """
        from web3.exceptions import TransactionNotFound

        state = receipt.protocol_state
        nonce_text = state.get("sourceNonce")
        sender = state.get("sourceSender")
        Web3 = _web3().Web3
        if not isinstance(nonce_text, str) or not nonce_text.isdigit():
            return None
        if not isinstance(sender, str) or not Web3.is_address(sender):
            return None
        try:
            if self.conn.w3.eth.get_transaction(tx_hash) is not None:
                return None                      # still in a mempool: pending, not replaced
        except TransactionNotFound:
            pass
        nonce = int(nonce_text)
        account_nonce = int(self.conn.w3.eth.get_transaction_count(Web3.to_checksum_address(sender), "latest"))
        if account_nonce <= nonce:
            return None                          # the nonce is still unused: nothing has replaced it
        return (f"transaction {tx_hash} (nonce {nonce}) was dropped or replaced before it mined; "
                "no funds moved by it — recover() then resume() re-dispatches")

    def _scan_can_run(self, approvals: list[str]) -> bool:
        """Whether a source-history log scan could actually run, i.e. a confirmed approval gives it a
        block to start from (the other precondition, a known sender, ``_dropped_error`` already proved).

        Recovery only treats a dropped transaction as resumable once the scan has PROVED no
        dispatch/deposit of ours exists; when the scan cannot run there is nothing to prove it
        against, so the dropped transaction stays terminal instead of inviting a second send.
        """
        return bool(approvals) and self._approval_scan_block(approvals) is not None

    def _validate_hyperlane_state(self, route: Route, plan: Plan, receipt: Receipt) -> bytes:
        """Bind every value that affects the dispatch before trusting checkpointed transaction ids."""
        state = receipt.protocol_state
        recipient32 = encoding.aleo_address_to_bytes32(plan.recipient)
        if (receipt.protocol != "hyperlane"
                or state.get("destinationDomain") != self._hyperlane_metadata(route).destination_domain
                or state.get("amountAtomic") != str(plan.amount_atomic)
                or not isinstance(state.get("recipientBytes32"), str)
                or state["recipientBytes32"].lower() != "0x" + recipient32.hex()):
            raise CheckpointInvalidError("Hyperlane checkpoint does not match the prepared transfer")
        ids = state.get("approvalTxIds", [])
        if not isinstance(ids, list) or any(not isinstance(i, str) or not _HASH_RE.fullmatch(i) for i in ids):
            raise CheckpointInvalidError("Hyperlane checkpoint contains invalid approval transaction ids")
        return recipient32

    def _hyperlane_source_status(self, route: Route, plan: Plan, receipt: Receipt, *,
                                 detect_dropped: bool = True) -> Receipt:
        self._validate_hyperlane_state(route, plan, receipt)
        self.assert_chain(route)
        source_tx_id = self._require_hash(receipt.source_tx_id, "source transaction id")
        observed = self.conn.get_receipt(source_tx_id)
        if observed is None:
            dropped = self._dropped_error(receipt, source_tx_id) if detect_dropped else None
            return receipt if dropped is None else self._dropped(receipt, dropped)
        if int(observed["status"]) == 0:
            return self._failed(receipt, "sourceError", f"EVM transaction reverted: {source_tx_id}")
        message_id = self._message_id_from_receipt(route, observed)
        state = dict(receipt.protocol_state)
        if message_id is not None:
            state["messageId"] = message_id
        return receipt.replace(id=message_id or source_tx_id, status=Status.DELIVERY_PENDING, protocol_state=state)

    def _xreserve_quote_from_state(self, route: Route, plan: Plan, receipt: Receipt) -> _XReserveQuote:
        """veil ``resumeQuote``: rebuild the deposit arguments from saved state and bind them to the plan."""
        Web3 = _web3().Web3
        s = receipt.protocol_state
        if receipt.protocol != "xreserve":
            raise CheckpointInvalidError("Checkpoint does not match the prepared xReserve route")
        if s.get("mintMode") != plan.mint_mode or s.get("intendedRecipient") != plan.recipient:
            raise CheckpointInvalidError("Checkpoint does not match the prepared xReserve recipient")
        try:
            ok = (Web3.is_address(s["xReserveContract"]) and Web3.is_address(s["tokenAddress"])
                  and isinstance(s["sourceChainId"], int) and isinstance(s["remoteDomain"], int)
                  and _HASH_RE.fullmatch(s["remoteRecipientBytes32"]) is not None
                  and isinstance(s["hookData"], str) and len(s["hookData"]) == 132 and s["hookData"].startswith("0x")
                  and str(s["amountAtomic"]).isdigit() and str(s["maxFeeAtomic"]).isdigit())
        except (KeyError, TypeError):
            ok = False
        if not ok:
            raise CheckpointInvalidError("Checkpoint contains invalid xReserve submission state")
        ids = s.get("approvalTxIds", [])
        if not isinstance(ids, list) or any(not isinstance(i, str) or not _HASH_RE.fullmatch(i) for i in ids):
            raise CheckpointInvalidError("Checkpoint contains invalid xReserve approval transaction ids")
        meta = self._xreserve_metadata(route)   # reuse the registry validator rather than trusting raw metadata again
        return _XReserveQuote(
            xreserve_contract=Web3.to_checksum_address(s["xReserveContract"]), token=Web3.to_checksum_address(s["tokenAddress"]),
            source_chain_id=int(s["sourceChainId"]), source_domain=meta.source_domain, remote_domain=int(s["remoteDomain"]),
            remote_token_bytes32=meta.remote_token_bytes32,
            remote_recipient_bytes32=bytes.fromhex(s["remoteRecipientBytes32"][2:]), amount_atomic=int(s["amountAtomic"]),
            max_fee_atomic=int(s["maxFeeAtomic"]), hook_data=bytes.fromhex(s["hookData"][2:]), balance_atomic=0, allowance_atomic=0,
            bridge_program=meta.bridge_program, wrapper_program=meta.wrapper_program)

    def _observed_owner(self, plan: Plan, receipt: Receipt | None) -> str:
        """Prefer the sender committed to the receipt or plan so read-only recovery never needs a signer."""
        Web3 = _web3().Web3
        saved = receipt.protocol_state.get("sourceSender") if receipt is not None else None
        for candidate in (saved, plan.sender, self.conn.address):
            if isinstance(candidate, str) and Web3.is_address(candidate):
                return Web3.to_checksum_address(candidate)
        raise ConfigurationError("Read-only EVM access requires the prepared sender address (plan.sender or protocol_state.sourceSender)")

    def _xreserve_source_status(self, route: Route, plan: Plan, receipt: Receipt, *,
                                detect_dropped: bool = True) -> Receipt:
        q = self._xreserve_quote_from_state(route, plan, receipt)
        self.assert_chain(route)
        owner = self._observed_owner(plan, receipt)
        source_tx_id = self._require_hash(receipt.source_tx_id, "xReserve source transaction id")
        observed = self.conn.get_receipt(source_tx_id)
        if observed is None:
            dropped = self._dropped_error(receipt, source_tx_id) if detect_dropped else None
            return receipt if dropped is None else self._dropped(receipt, dropped)
        if int(observed["status"]) == 0:
            return self._failed(receipt, "sourceError", f"EVM transaction reverted: {source_tx_id}")
        return self._confirmed_deposit_receipt(route, q, owner=owner,
                                               approval_tx_ids=list(receipt.protocol_state.get("approvalTxIds", [])),
                                               source_tx_id=source_tx_id, receipt=observed, mint_mode=plan.mint_mode,
                                               intended_recipient=plan.recipient)

    def source_status(self, plan: Plan, receipt: Receipt) -> Receipt:
        """One read-only refresh of an Ethereum source leg (brief §2.4 branches 1 and 3, plus xReserve SOURCE_CONFIRMING).

        ``SOURCE_APPROVAL_PENDING``: approval receipt → ``SOURCE_SUBMISSION_PENDING`` (or ``FAILED`` on revert).
        ``SOURCE_CONFIRMING``: Hyperlane → ``DELIVERY_PENDING`` with the ``DispatchId`` message id;
        xReserve → ``ATTESTATION_PENDING`` after re-verifying the ``DepositedToRemote`` event.
        An unmined transaction returns the receipt unchanged. Never signs.

        A transaction that can never mine — the node has forgotten it and the sender's account
        nonce has moved past the ``sourceNonce`` it was broadcast at, so something else took that
        nonce — becomes ``EXPIRED`` with ``protocol_state["sourceError"]`` and ``dropped: True``.
        This is the REFRESH answer for one known hash: it says that this transaction is dead, not
        that the transfer never happened. ``recover_source`` answers the other question — it scans
        source history first, so a dispatch that did land (including one of our own resends at the
        replacing nonce) wins, and only a proved-empty history turns a dropped hash into a
        resumable ``SOURCE_SUBMISSION_PENDING``.

        Deviation from veil (deliberate): veil raises for a reverted Hyperlane/xReserve *source*
        transaction but returns ``FAILED`` for a reverted approval. Here every reverted source
        transaction observed at this stage becomes ``FAILED`` with ``protocol_state["sourceError"]`` —
        one uniform rule that plan 4's ``get_status``/``wait`` can rely on without a protocol switch.
        ``send()`` still raises on revert.
        """
        route = self._route_for_plan(plan)
        if receipt.protocol != plan.protocol or receipt.protocol_state.get("routeId") != plan.route_id:
            raise CheckpointInvalidError("Receipt does not match the prepared route")
        if receipt.status == Status.SOURCE_APPROVAL_PENDING:
            approval_id = self._require_hash(receipt.id, "EVM approval transaction id")
            observed = self.conn.get_receipt(approval_id)
            if observed is None:
                return receipt
            if int(observed["status"]) == 0:
                return self._failed(receipt, "sourceError", f"EVM approval transaction reverted: {approval_id}")
            return receipt.replace(status=Status.SOURCE_SUBMISSION_PENDING)
        if receipt.status == Status.SOURCE_CONFIRMING:
            if route.protocol == "hyperlane":
                return self._hyperlane_source_status(route, plan, receipt)
            return self._xreserve_source_status(route, plan, receipt)
        raise BridgeError("source_status refreshes SOURCE_APPROVAL_PENDING and SOURCE_CONFIRMING receipts only; "
                          "use bridge.get_status for later stages")

    # -- recovery -----------------------------------------------------------------------------

    def _checkpoint_approvals(self, checkpoint: Checkpoint) -> list[str]:
        approvals = list((checkpoint.source or {}).get("approvalTransactionIds", []))
        if any(not isinstance(a, str) or not _HASH_RE.fullmatch(a) for a in approvals):
            raise CheckpointInvalidError("Bridge checkpoint contains an invalid approval transaction id")
        return approvals

    @staticmethod
    def _checkpoint_nonce(source: Mapping[str, Any]) -> str | None:
        """The checkpointed ``sourceNonce`` (decimal string), or ``None`` when absent or malformed.

        Version-1 checkpoints written before the nonce was recorded simply do not have it; recovery
        then behaves exactly as it did before, so a missing value is never an error.
        """
        value = source.get("sourceNonce")
        return value if isinstance(value, str) and value.isdigit() else None

    def _recovered_dropped(self, pending: Receipt, transaction_id: str, approvals: list[str]) -> Receipt:
        """The answer for a checkpointed hash that the history scan did not match.

        Unchanged (still ``SOURCE_CONFIRMING``) unless the hash can never mine. When it cannot and
        the scan actually ran — proving no dispatch/deposit of ours exists — the transfer is back at
        ``SOURCE_SUBMISSION_PENDING`` so ``resume`` may re-authorize the one remaining transaction;
        the reason travels in ``sourceError``. When the scan could not run, nothing proves the
        replacement was not our own dispatch, so the receipt is ``EXPIRED`` for an operator to look at.
        """
        message = self._dropped_error(pending, transaction_id)
        if message is None:
            return pending
        if not self._scan_can_run(approvals):
            return self._dropped(pending, message)
        return pending.replace(status=Status.SOURCE_SUBMISSION_PENDING,
                               protocol_state={**pending.protocol_state, "sourceError": message, "dropped": True})

    def _approval_scan_block(self, approvals: list[str]) -> int | None:
        """Highest block of a confirmed approval; an unresolved hash is skipped, a reverted one is an error."""
        block: int | None = None
        for approval in approvals:
            observed = self.conn.get_receipt(approval)
            if observed is None:
                continue
            if int(observed["status"]) == 0:
                raise BridgeError(f"EVM transaction reverted: {approval}")
            number = int(observed["blockNumber"])
            block = number if block is None or number > block else block
        return block

    def _head_block(self, address: str) -> int:
        errors = _provider_errors()
        try:
            return int(self.conn.w3.eth.block_number)
        except errors as exc:
            raise BridgeError(f"Could not read the current block number to bound a log scan of {address}: {exc}") from exc

    def _scan_logs(self, address: str, from_block: int) -> list[Any]:
        """Every log of *address* from *from_block* to the head, read in bounded ascending chunks.

        The head is read once so the scan terminates on a fixed range, and every request carries an
        explicit ``fromBlock``/``toBlock``: an unbounded filter is what public RPCs reject or truncate,
        and a truncated answer would silently read as "no dispatch/deposit was ever submitted".

        A load-balanced endpoint can answer ``eth_getLogs`` from a node behind the one that gave us
        that head, which rejects the range it just handed out (:data:`HEAD_RACE_RE`). Such a chunk is
        retried up to ``log_scan_head_race_retries`` times, pausing ``log_scan_head_race_sleep``
        seconds and re-reading the head each time; the scan's upper bound only ever moves DOWN to the
        answering node's head, so the range stays one the whole cluster can serve. Any other provider
        error, and a head race that outlives the retries, still raises ``BridgeError`` — a scan that
        cannot finish must never look like an empty history.
        """
        errors = _provider_errors()
        chunk = self.log_scan_chunk_blocks
        latest = self._head_block(address)
        logs: list[Any] = []
        start = from_block
        while start <= latest:
            end = min(start + chunk - 1, latest)
            for attempt in range(self.log_scan_head_race_retries + 1):
                try:
                    logs.extend(self.conn.w3.eth.get_logs({"address": address, "fromBlock": start, "toBlock": end}))
                    break
                except errors as exc:
                    head_race = HEAD_RACE_RE.search(str(exc)) is not None
                    if head_race and attempt < self.log_scan_head_race_retries:
                        self.sleep(self.log_scan_head_race_sleep)
                        latest = min(latest, self._head_block(address))
                        end = min(end, latest)
                        if end >= start:
                            continue                  # the answering node has caught up, or we followed it down
                    raise BridgeError(
                        f"eth_getLogs failed for blocks {start}-{end} of {from_block}-{latest} on {address}: {exc}. "
                        + (f"The endpoint's head lagged the range it reported for {self.log_scan_head_race_retries} "
                           f"retries. " if head_race else "")
                        + f"Use a dedicated RPC endpoint, or a smaller EthModule(log_scan_chunk_blocks=...) "
                        f"than the current {chunk}.") from exc
            start = end + 1
        return logs

    def _recover_hyperlane_from_history(self, route: Route, recipient32: bytes, receipt: Receipt, approvals: list[str],
                                        *, required: bool) -> Receipt | None:
        """Scan router ``SentTransferRemote`` logs after the last confirmed approval; sender and router must match."""
        from web3.exceptions import TransactionNotFound

        Web3 = _web3().Web3
        sender = receipt.protocol_state.get("sourceSender")
        if not isinstance(sender, str) or not Web3.is_address(sender):
            if required:
                raise BridgeError("Cannot safely resume Hyperlane without the source account used by the approval")
            return None
        from_block = self._approval_scan_block(approvals)
        if from_block is None:
            if required:
                raise BridgeError("Cannot safely resume Hyperlane because no confirmed approval block is available "
                                  "for source history verification")
            return None
        amount = int(receipt.protocol_state["amountAtomic"])
        meta = self._hyperlane_metadata(route)                # reuse the registry validator before touching the router
        destination = meta.destination_domain
        router = meta.router
        warp = self._contract(router, WARP_ROUTE_ABI)
        topic = Web3.keccak(text="SentTransferRemote(uint32,bytes32,uint256)")
        candidates: list[str] = []
        for log in self._scan_logs(router, from_block):
            if not log["topics"] or bytes(log["topics"][0]) != bytes(topic):
                continue
            args = warp.events.SentTransferRemote().process_log(log)["args"]
            tx_hash = Web3.to_hex(log["transactionHash"])
            if (int(args["destination"]) == destination and bytes(args["recipient"]) == recipient32
                    and int(args["amount"]) == amount and tx_hash not in candidates):
                candidates.append(tx_hash)
        matches: list[Receipt] = []
        for tx_hash in candidates:
            try:
                tx = self.conn.w3.eth.get_transaction(tx_hash)
            except TransactionNotFound:
                continue
            observed = self.conn.get_receipt(tx_hash)
            if (tx is None or observed is None or tx["to"] is None
                    or Web3.to_checksum_address(tx["from"]) != Web3.to_checksum_address(sender)
                    or Web3.to_checksum_address(tx["to"]) != router):
                continue
            if int(observed["status"]) == 0:
                # Deliberately asymmetric with the xReserve scan below, and identical to veil: a
                # reverted Hyperlane candidate raises (hyperlane/evm.ts) because sender+router+args
                # already identify it as ours, while xreserve/evmToAleo.ts swallows a rejected
                # candidate because the shared contract's logs are mostly other accounts' deposits.
                raise BridgeError(f"EVM transaction reverted: {tx_hash}")
            message_id = self._message_id_from_receipt(route, observed)
            state = dict(receipt.protocol_state)
            state.pop("sourceNonce", None)        # belongs to the checkpointed hash, not to this one
            if message_id is not None:
                state["messageId"] = message_id
            matches.append(receipt.replace(id=message_id or tx_hash, status=Status.DELIVERY_PENDING,
                                           source_tx_id=tx_hash, protocol_state=state))
        if len(matches) > 1:
            raise BridgeError("Multiple matching Hyperlane dispatches were found; recovery cannot safely choose one source transaction")
        return matches[0] if matches else None

    def _recover_hyperlane(self, route: Route, plan: Plan, checkpoint: Checkpoint, *, required: bool) -> Receipt:
        Web3 = _web3().Web3
        recipient32 = encoding.aleo_address_to_bytes32(plan.recipient)
        approvals = self._checkpoint_approvals(checkpoint)
        sender = Web3.to_checksum_address(plan.sender) if plan.sender and Web3.is_address(plan.sender) else None
        meta = self._hyperlane_metadata(route)
        source = checkpoint.source or {}
        state = self._hyperlane_protocol_state(route, recipient_bytes32=recipient32,
                                               destination_domain=meta.destination_domain,
                                               native_value_atomic=0, amount_atomic=plan.amount_atomic,
                                               approval_tx_ids=approvals, sender=sender,
                                               source_nonce=self._checkpoint_nonce(source))
        transaction_id = source.get("transactionId")
        if not transaction_id:
            if not approvals:
                raise CheckpointInvalidError("Bridge checkpoint contains no submitted transaction")
            pending = Receipt(id=approvals[-1], protocol="hyperlane", status=Status.SOURCE_APPROVAL_PENDING, protocol_state=state)
            observed = self.conn.get_receipt(approvals[-1])
            if observed is None:
                if required:
                    raise BridgeError("Cannot safely resume Hyperlane because no confirmed approval block is available "
                                      "for source history verification")
                return pending
            if int(observed["status"]) == 0:
                return self._failed(pending, "sourceError", f"EVM approval transaction reverted: {approvals[-1]}")
            recovered = self._recover_hyperlane_from_history(route, recipient32, pending, approvals, required=required)
            return recovered or pending.replace(status=Status.SOURCE_SUBMISSION_PENDING)
        transaction_id = self._require_hash(transaction_id, "source transaction id")
        pending = Receipt(id=transaction_id, protocol="hyperlane", status=Status.SOURCE_CONFIRMING,
                          source_tx_id=transaction_id, protocol_state=state)
        # History first: a dispatch that landed always beats the checkpointed hash, even when that
        # hash was dropped (its replacement may BE our own resent dispatch).
        observed = self._hyperlane_source_status(route, plan, pending, detect_dropped=False)
        if observed is not pending:
            return observed
        recovered = self._recover_hyperlane_from_history(route, recipient32, pending, approvals, required=False)
        if recovered is not None:
            return recovered
        return self._recovered_dropped(pending, transaction_id, approvals)

    def _recover_xreserve_from_history(self, route: Route, plan: Plan, q: _XReserveQuote, owner: str, approvals: list[str],
                                       *, required: bool) -> Receipt | None:
        """Scan xReserve logs after the last confirmed approval; a candidate matches only if every event field matches."""
        Web3 = _web3().Web3
        from_block = self._approval_scan_block(approvals)
        if from_block is None:
            if required:
                raise BridgeError("Cannot safely resume xReserve because no confirmed approval block is available "
                                  "for source history verification")
            return None
        hashes: list[str] = []
        for log in self._scan_logs(q.xreserve_contract, from_block):
            tx_hash = Web3.to_hex(log["transactionHash"])
            if tx_hash not in hashes:
                hashes.append(tx_hash)
        matches: list[Receipt] = []
        for tx_hash in hashes:
            observed = self.conn.get_receipt(tx_hash)
            if observed is None:
                continue
            try:
                matches.append(self._confirmed_deposit_receipt(route, q, owner=owner, approval_tx_ids=approvals, source_tx_id=tx_hash,
                                                               receipt=observed, mint_mode=plan.mint_mode, intended_recipient=plan.recipient))
            except BridgeError:
                continue                       # other accounts' deposits share the contract; unrelated unless every field matches
        if len(matches) > 1:
            raise BridgeError("Multiple matching xReserve deposits were found; recovery cannot safely choose one source transaction")
        return matches[0] if matches else None

    def _recover_xreserve(self, route: Route, plan: Plan, checkpoint: Checkpoint, *, required: bool) -> Receipt:
        Web3 = _web3().Web3
        owner = self._observed_owner(plan, None)
        approvals = self._checkpoint_approvals(checkpoint)
        source = checkpoint.source or {}
        stored_hook = source.get("hookData")
        if stored_hook is not None and (not isinstance(stored_hook, str) or not re.fullmatch(r"0x[0-9a-fA-F]{130}", stored_hook)):
            raise CheckpointInvalidError("Bridge checkpoint contains invalid xReserve hook data")
        hook = bytes.fromhex(stored_hook[2:]) if stored_hook else encoding.xreserve_hook_data(
            plan.mint_mode, plan.recipient, self.network, "0scalar")
        meta = self._xreserve_metadata(route)                 # reuse the registry validator rather than trusting raw metadata
        token = self.registry.asset(route.source_asset_id).locator
        if token is None or token.kind != "evm-contract":
            raise RouteUnavailableError(f"xReserve source token contract is missing: {route.id}")
        q = _XReserveQuote(
            xreserve_contract=meta.xreserve_contract, token=Web3.to_checksum_address(token.value),
            source_chain_id=meta.source_chain_id, source_domain=meta.source_domain, remote_domain=meta.remote_domain,
            remote_token_bytes32=meta.remote_token_bytes32,
            remote_recipient_bytes32=self._xreserve_recipient_bytes32(route, meta, plan.recipient, plan.mint_mode),
            amount_atomic=plan.amount_atomic, max_fee_atomic=meta.max_fee_atomic, hook_data=hook,
            balance_atomic=0, allowance_atomic=0, bridge_program=meta.bridge_program, wrapper_program=meta.wrapper_program)
        state = self._xreserve_protocol_state(route, q, approval_tx_ids=approvals, sender=owner, mint_mode=plan.mint_mode,
                                              intended_recipient=plan.recipient,
                                              source_nonce=self._checkpoint_nonce(source))
        transaction_id = source.get("transactionId")
        if not transaction_id:
            if not approvals:
                raise CheckpointInvalidError("Bridge checkpoint contains no submitted transaction")
            pending = Receipt(id=approvals[-1], protocol="xreserve", status=Status.SOURCE_APPROVAL_PENDING, protocol_state=state)
            observed = self.conn.get_receipt(approvals[-1])
            if observed is None:
                if required:
                    raise BridgeError("Cannot safely resume xReserve because no confirmed approval block is available "
                                      "for source history verification")
                return pending
            if int(observed["status"]) == 0:
                return self._failed(pending, "sourceError", f"EVM approval transaction reverted: {approvals[-1]}")
            recovered = self._recover_xreserve_from_history(route, plan, q, owner, approvals, required=required)
            return recovered or pending.replace(status=Status.SOURCE_SUBMISSION_PENDING)
        transaction_id = self._require_hash(transaction_id, "xReserve source transaction id")
        pending = Receipt(id=transaction_id, protocol="xreserve", status=Status.SOURCE_CONFIRMING,
                          source_tx_id=transaction_id, protocol_state=state)
        observed = self._xreserve_source_status(route, plan, pending, detect_dropped=False)   # history first
        if observed is not pending:
            return observed
        recovered = self._recover_xreserve_from_history(route, plan, q, owner, approvals, required=False)
        if recovered is not None:
            return recovered
        return self._recovered_dropped(pending, transaction_id, approvals)

    def recover_source(self, plan: Plan, checkpoint: Checkpoint, *, required: bool = False) -> Receipt:
        """Reconstruct an interrupted Ethereum source leg from a checkpoint without signing (brief §2.7, §3.1, §3.2).

        Approval-only checkpoints: observe the last approval; when confirmed, scan the router /
        xReserve logs from its block for a matching dispatch or deposit and stop at
        ``SOURCE_SUBMISSION_PENDING`` when none exists — recovery never moves funds. Checkpoints
        with a source transaction are observed through ``source_status``. ``required=True`` (plan
        4's resume-before-dispatch mode) demands the scan actually run — a known sender and a
        confirmed approval block — or raises, instead of quietly returning an approval-boundary
        receipt. ``required=True`` only makes the INABILITY to scan fatal: once the scan actually
        runs, a completed scan that matches zero dispatches/deposits is a valid answer ("nothing
        was submitted yet"), not an error, and returns ``SOURCE_SUBMISSION_PENDING`` so ``resume``
        may re-authorize the send.

        This is the history-first counterpart to ``source_status``, which refreshes one known hash
        and calls a dropped transaction ``EXPIRED``. Here the log scan runs FIRST even when the
        checkpointed hash was dropped, because the transaction that replaced it may be our own
        resent dispatch — a real dispatch in history always wins. Only a scan that ran and matched
        nothing turns a dropped hash into ``SOURCE_SUBMISSION_PENDING`` (``next == "resume"``,
        carrying ``sourceError``); when the scan could not run at all the receipt stays ``EXPIRED``
        rather than inviting a second send.
        """
        if checkpoint.version != 1 or checkpoint.intent.get("bridgeProtocol") != plan.protocol or checkpoint.route.get("id") != plan.route_id:
            raise CheckpointInvalidError("Bridge checkpoint does not match the prepared route")
        if checkpoint.route.get("registryVersion") != self.registry.version:
            raise RegistryVersionMismatchError(
                f"Checkpoint uses registry {checkpoint.route.get('registryVersion')}; this client has {self.registry.version}")
        if plan.protocol == "hyperlane" and checkpoint.destination:
            raise CheckpointInvalidError("Hyperlane checkpoints must not carry a destination leg")
        route = self._route_for_plan(plan)
        self.assert_chain(route)
        if route.protocol == "hyperlane":
            return self._recover_hyperlane(route, plan, checkpoint, required=required)
        if route.protocol == "xreserve":
            return self._recover_xreserve(route, plan, checkpoint, required=required)
        raise UnsupportedRouteError(f"No Ethereum recovery for protocol {route.protocol}")

    def _mailbox_address(self) -> str:
        """The Hyperlane Mailbox deployed on this chain.

        Prefer a route that ORIGINATES here and passes the full metadata validator: its
        ``mailboxAddress`` is the contract this chain's own dispatches go through, checksummed and
        checked. Only if no such route exists do we fall back to any route that merely touches this
        chain (whose ``mailboxAddress`` may be the remote one, and is unvalidated).
        """
        routes = list(self.registry.routes(protocol="hyperlane", include_unavailable=True,
                                           environment=self.bridge.environment))
        for route in routes:
            if self.registry.asset(route.source_asset_id).chain_id != self.chain.id:
                continue
            try:
                return self._hyperlane_metadata(route).mailbox
            except (ConfigurationError, RouteUnavailableError):
                continue
        for route in routes:
            chains = {self.registry.asset(route.source_asset_id).chain_id, self.registry.asset(route.destination_asset_id).chain_id}
            mailbox = route.metadata.get("mailboxAddress")
            if self.chain.id in chains and isinstance(mailbox, str):
                return mailbox
        raise UnsupportedRouteError(f"No Hyperlane Mailbox is configured for {self.chain.id} in registry {self.registry.version}")

    def is_delivered(self, message_id: str | bytes) -> bool:
        """``Mailbox.delivered(bytes32)`` on this chain — the canonical Aleo → Ethereum delivery signal."""
        raw = bytes.fromhex(message_id[2:]) if isinstance(message_id, str) and message_id.startswith("0x") else message_id
        if not isinstance(raw, (bytes, bytearray)) or len(raw) != 32:
            raise BridgeError("Hyperlane delivery requires a 32-byte message id")
        return bool(self._contract(self._mailbox_address(), MAILBOX_ABI).functions.delivered(bytes(raw)).call())

    def balance(self, asset: Any, *, address: str | None = None) -> int:
        """Atomic balance of ``asset`` (native via ``eth_getBalance``, ERC-20 via ``balanceOf``) for ``address`` or the connection's account."""
        target = self._asset(asset)
        owner = self._owner(address)
        if owner is None:
            raise ConfigurationError("balance() needs an address: pass address= or configure a signer")
        if target.locator is None:
            raise UnsupportedRouteError(f"{target.id} has no on-chain locator")
        if target.locator.kind == "native":
            return int(self.conn.w3.eth.get_balance(owner))
        if target.locator.kind == "evm-contract":
            return int(self._erc20(target.locator.value).functions.balanceOf(owner).call())
        raise UnsupportedRouteError(f"{target.id} is not an EVM asset")

    def _chain_assertion_route(self) -> Route:
        """Any route originating on this chain with a usable ``sourceChainId``, used only to bind
        ``assert_chain`` to the registry's notion of this chain (never touches contracts)."""
        routes = [r for r in self.registry.routes(include_unavailable=True, environment=self.bridge.environment)
                  if self.registry.asset(r.source_asset_id).chain_id == self.chain.id
                  and isinstance(r.metadata.get("sourceChainId"), int)]
        if not routes:
            raise UnsupportedRouteError(
                f"No Hyperlane or xReserve route with sourceChainId is configured for {self.chain.id}")
        return routes[0]

    def chain_status(self) -> ChainStatus:
        """Address, signing ability, and atomic balances of every registry asset on this chain (empty when read-only).

        Asserts the connected ``Web3``'s ``eth_chainId`` matches this chain's registry
        ``sourceChainId`` first, so a connection pointed at the wrong network raises
        ``ChainMismatchError`` instead of silently reading balances from the wrong chain.
        """
        self.assert_chain(self._chain_assertion_route())
        address = self.conn.address
        balances: dict[str, int] = {}
        if address is not None:
            for asset in self.registry.assets(chain=self.chain.id):
                if asset.locator is not None and asset.locator.kind in ("native", "evm-contract"):
                    balances[asset.id] = self.balance(asset, address=address)
        return ChainStatus(chain_id=self.chain.id, address=address, can_sign=self.conn.can_sign, balances=balances)


__all__ = ["Ethereum", "EthModule"]
