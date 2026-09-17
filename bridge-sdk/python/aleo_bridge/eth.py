"""Ethereum connection and the ``bridge.eth`` module (Hyperlane + xReserve, Ethereum origin).

``web3`` and ``eth_account`` are imported lazily so ``import aleo_bridge`` works
without the ``evm`` extra; the first call that needs them raises
``MissingExtraError("evm", ...)``.
"""
from __future__ import annotations

import os
from typing import Any, Mapping

from ._evm_abi import EVM_CHAIN_BY_ENVIRONMENT
from .errors import BridgeError, ConfigurationError, MissingExtraError
from .registry import Asset, Chain, Registry, Route
from .types import Plan, Step
from .units import format_decimal_amount


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


class EthModule:
    """``bridge.eth`` — Ethereum-origin Hyperlane and xReserve actions (reads return values, writes return ``EvmCall``)."""

    def __init__(self, bridge: Any, conn: Ethereum) -> None:
        self.bridge = bridge
        self.conn = conn
        self.registry: Registry = bridge.registry
        self.network: str = bridge.network            # "mainnet" | "testnet" → aleo.<network> for encoders
        self.chain: Chain = self.registry.chain(EVM_CHAIN_BY_ENVIRONMENT[bridge.environment])


__all__ = ["Ethereum", "EthModule"]
