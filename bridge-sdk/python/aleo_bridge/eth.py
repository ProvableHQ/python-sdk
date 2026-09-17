"""Ethereum connection and the ``bridge.eth`` module (Hyperlane + xReserve, Ethereum origin).

``web3`` and ``eth_account`` are imported lazily so ``import aleo_bridge`` works
without the ``evm`` extra; the first call that needs them raises
``MissingExtraError("evm", ...)``.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Mapping

from . import encoding
from ._evm_abi import ERC20_ABI, EVM_CHAIN_BY_ENVIRONMENT, WARP_ROUTE_ABI, ZERO_ADDRESS
from .errors import (AmbiguousRouteError, BridgeError, ChainMismatchError, ConfigurationError, InvalidAmountError,
                     InvalidRecipientError, MissingExtraError, RegistryVersionMismatchError, RouteNotFoundError,
                     RouteUnavailableError)
from .registry import Asset, Chain, Registry, Route
from .types import EvmHyperlaneQuote, Fee, Plan, Step
from .units import format_decimal_amount, parse_decimal_amount, resolve_amount


def _web3():
    try:
        import web3
    except ImportError as exc:  # pragma: no cover - exercised by test_import_without_web3
        raise MissingExtraError("evm", "Ethereum connections") from exc
    return web3


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
                 private_key: str | None = None) -> None:
        if (rpc_url is None) == (w3 is None):
            raise ConfigurationError("Pass exactly one of rpc_url or w3 to Ethereum(...)")
        if signer is not None and private_key is not None:
            raise ConfigurationError("Pass at most one of signer or private_key to Ethereum(...)")
        if w3 is None:
            web3 = _web3()
            w3 = web3.Web3(web3.HTTPProvider(rpc_url))
        if private_key is not None:
            signer = _eth_account().from_key(private_key)
        self._w3 = w3
        self._signer = signer
        self._chain_id: int | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Ethereum | None":
        """``EVM_PRIVATE_KEY`` + ``ETHEREUM_RPC_URL`` (both or neither) → signing connection; neither → None."""
        env = os.environ if env is None else env
        key = env.get("EVM_PRIVATE_KEY")
        url = env.get("ETHEREUM_RPC_URL")
        if bool(key) != bool(url):
            raise ConfigurationError("Set both EVM_PRIVATE_KEY and ETHEREUM_RPC_URL or neither")
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
            return Web3.to_hex(self._w3.eth.send_transaction(tx))
        tx.setdefault("nonce", self._w3.eth.get_transaction_count(sender, "pending"))
        if "gas" not in tx:
            estimate_fields = {k: v for k, v in tx.items() if k in ("from", "to", "data", "value")}
            tx["gas"] = int(self._w3.eth.estimate_gas(estimate_fields)) * 12 // 10
        if "gasPrice" not in tx and "maxFeePerGas" not in tx:
            base_fee = self._w3.eth.get_block("latest").get("baseFeePerGas")
            if base_fee is None:
                tx["gasPrice"] = int(self._w3.eth.gas_price)
            else:
                tip = int(self._w3.eth.max_priority_fee)
                tx["maxPriorityFeePerGas"] = tip
                tx["maxFeePerGas"] = int(base_fee) * 2 + tip
        signed = self._signer.sign_transaction(tx)
        return Web3.to_hex(self._w3.eth.send_raw_transaction(signed.raw_transaction))

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


def _plan_for(registry: Registry, route: Route, *, amount_atomic: int, recipient: str, sender: str | None,
              mint_mode: str = "public") -> Plan:
    """Build the ``Plan`` for an Ethereum-origin route (mirrors veil ``prepare`` steps, brief §2.1).

    Plan 4's ``lifecycle.prepare`` is the Tier-1 entry point; this helper is the Tier-2
    path so ``bridge.eth.*`` calls carry a checkpointable plan without importing lifecycle.
    """
    source: Asset = registry.asset(route.source_asset_id)
    destination: Asset = registry.asset(route.destination_asset_id)
    if mint_mode not in ("public", "record", "private"):
        raise BridgeError(f"mint_mode must be public, record or private; got {mint_mode!r}")
    if mint_mode != "public" and route.protocol != "xreserve":
        raise BridgeError("mint_mode other than public applies only to xReserve deposits to Aleo")
    if amount_atomic <= 0:
        raise BridgeError("amount_atomic must be positive")
    if route.protocol == "xreserve":
        steps = (Step("source-approval", "approve", "evm-wallet", False),
                 Step("source-deposit", "deposit", "evm-wallet", True),
                 Step("deposit-attestation", "wait-attestation", "protocol", False),
                 Step("destination-mint", "mint", "aleo-wallet" if mint_mode == "private" else "protocol", False))
    else:
        steps = tuple([Step("source-approval", "approve", "evm-wallet", False)] if source.kind == "token" else []) + (
            Step("source-dispatch", "dispatch", "evm-wallet", True),
            Step("message-delivery", "wait-delivery", "protocol", False),
            Step("destination-confirmation", "confirm-delivery", "protocol", False))
    return Plan(route_id=route.id, registry_version=registry.version, protocol=route.protocol,
                environment=route.environment, source_asset_id=source.id, destination_asset_id=destination.id,
                amount=format_decimal_amount(amount_atomic, source.decimals), amount_atomic=amount_atomic,
                recipient=recipient, sender=sender, mint_mode=mint_mode, steps=steps)


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


class EthModule:
    """``bridge.eth`` — Ethereum-origin Hyperlane and xReserve actions (reads return values, writes return ``EvmCall``)."""

    def __init__(self, bridge: Any, conn: Ethereum) -> None:
        self.bridge = bridge
        self.conn = conn
        self.registry: Registry = bridge.registry
        self.network: str = bridge.network            # "mainnet" | "testnet" → aleo.<network> for encoders
        self.chain: Chain = self.registry.chain(EVM_CHAIN_BY_ENVIRONMENT[bridge.environment])

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
        return Fee(kind="network", chain_id=self.chain.id, asset_id=asset_id,
                   amount=format_decimal_amount(amount_wei, 18), estimated=True)

    # -- Hyperlane quote ----------------------------------------------------------------------

    def _quote_hyperlane(self, route: Route, recipient_bytes32: bytes, amount_atomic: int, owner: str | None) -> _HyperlaneQuote:
        """Brief §3.1: chain assert → quoteTransferRemote → native/collateral split → allowance."""
        self.assert_chain(route)
        Web3 = _web3().Web3
        meta = route.metadata
        router = Web3.to_checksum_address(str(meta["routerAddress"]))
        router_type = str(meta["routerType"])
        destination_domain = int(meta["destinationDomain"])
        quotes = self._contract(router, WARP_ROUTE_ABI).functions.quoteTransferRemote(
            destination_domain, recipient_bytes32, amount_atomic).call()
        native_value = sum(int(q[1]) for q in quotes if Web3.to_checksum_address(q[0]) == ZERO_ADDRESS)
        if router_type == "native":
            if native_value < amount_atomic:
                raise BridgeError("Native Hyperlane quote does not cover the transfer amount")
            return _HyperlaneQuote(router, "native", None, destination_domain, recipient_bytes32, amount_atomic,
                                   native_value, native_value - amount_atomic, 0, None, False)
        if router_type != "collateral":
            raise RouteUnavailableError(f"Hyperlane route has an invalid routerType {router_type!r}: {route.id}")
        token = Web3.to_checksum_address(str(meta["tokenAddress"]))
        token_amount = sum(int(q[1]) for q in quotes if Web3.to_checksum_address(q[0]) == token)
        if token_amount < amount_atomic:
            raise BridgeError("Collateral Hyperlane quote does not cover the transfer amount")
        allowance = int(self._erc20(token).functions.allowance(owner, router).call()) if owner else None
        return _HyperlaneQuote(router, "collateral", token, destination_domain, recipient_bytes32, amount_atomic,
                               native_value, native_value, token_amount, allowance,
                               meta.get("requiresApprovalReset") is True)

    def quote_transfer_remote(self, asset: Any, recipient: str, *, amount: Any = None, amount_atomic: int | None = None,
                              route: Route | None = None, sender: str | None = None) -> EvmHyperlaneQuote:
        """Quote an Ethereum → Aleo Hyperlane transfer without signing.

        Native routes (ETH): ``msg.value`` carries the asset and the relayer fee, so
        ``native_fee_atomic = native_value_atomic - amount``. Collateral routes (WBTC, USDT):
        ``msg.value`` is fee only and ``approval_required`` reflects the router's ERC-20
        allowance for ``sender`` (or the connection's account); it is ``None`` when no account is known.
        """
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


__all__ = ["Ethereum", "EthModule"]
