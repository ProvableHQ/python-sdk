"""Privy server wallets as bridge signers (``pip install 'aleo-bridge-sdk[privy]'``).

An existing Privy wallet signs the Ethereum or Solana leg of a transfer without ever exporting
its key; the bridge keeps building, broadcasting, checkpointing and recovering exactly as with a
local key::

    from privy import PrivyClient
    from aleo_bridge import Bridge, Ethereum, Solana
    from aleo_bridge.privy import PrivyEvmSigner, PrivySolanaSigner

    privy = PrivyClient(app_id=..., app_secret=...)
    ethereum = Ethereum(rpc_url, signer=PrivyEvmSigner(privy, wallet_id=evm_wallet_id, address=evm_address))
    solana = Solana(rpc_url, signer=PrivySolanaSigner(privy, wallet_id=sol_wallet_id, address=sol_address))
    bridge = Bridge(aleo, ethereum=ethereum, solana=solana)

Construction contacts neither Privy nor a chain. Each ``send`` requests one signature through
Privy's wallet RPC (``eth_signTransaction`` / ``signTransaction``) and verifies it locally before
broadcasting through the connection's own RPC: an Ethereum transaction must recover to the
configured address, a Solana signature must verify over the exact message the bridge built, and a
Solana response whose message bytes changed is refused. Wallet provisioning, app secrets and
authorization keys stay with the application; the signers add no retries of their own.
"""
from __future__ import annotations

import base64
from typing import Any, Mapping, Sequence

from ._remote_signing import (RemoteSignedTransaction, checksum_address, signer_index, solana_pubkey,
                              unsigned_evm_fields, verified_evm_transaction, verified_solana_signature)
from .errors import BridgeError, ConfigurationError, MissingExtraError

PROVIDER = "Privy"


def _request_options(authorization_private_keys: Sequence[str] | None) -> Any:
    """Privy request options carrying the wallet authorization keys, or ``None`` when there are none."""
    if not authorization_private_keys:
        return None
    try:
        from privy.lib import AuthorizationContext, PrivyRequestOptions
    except ImportError as exc:
        raise MissingExtraError("privy", "Privy server wallets") from exc
    return PrivyRequestOptions(authorization_context=AuthorizationContext(
        authorization_private_keys=tuple(authorization_private_keys)))


def _services(client: Any, chain: str) -> tuple[Any, Any]:
    """``(client.wallets, client.wallets.<chain>)`` — the wallet lookup and the chain-specific signing
    service of a ``privy.PrivyClient``."""
    wallets = getattr(client, "wallets", None)
    service = getattr(wallets, chain, None)
    if service is None or not callable(getattr(service, "sign_transaction", None)) or not callable(getattr(wallets, "get", None)):
        raise ConfigurationError(
            "client must be a privy.PrivyClient (pip install 'aleo-bridge-sdk[privy]'); the low-level "
            "privy.PrivyAPI has no wallets.ethereum / wallets.solana signing services")
    return wallets, service


def _resolve(wallets: Any, wallet_id: str, *, chain_type: str, address: str, same: Any) -> str:
    """Fetch *wallet_id* from Privy and confirm it is a *chain_type* wallet at *address*."""
    wallet = wallets.get(wallet_id=wallet_id)
    found_type, found = getattr(wallet, "chain_type", None), getattr(wallet, "address", None)
    if found_type != chain_type or not isinstance(found, str) or not same(found):
        raise ConfigurationError(
            f"Privy wallet {wallet_id} is a {found_type or 'unknown'} wallet at {found or 'no address'}; "
            f"it does not match the configured {chain_type} address {address}")
    return address


def _wallet_id(wallet_id: Any) -> str:
    if not isinstance(wallet_id, str) or not wallet_id.strip():
        raise ConfigurationError("Privy wallet_id must not be empty")
    return wallet_id.strip()


class PrivyEvmSigner:
    """An existing Privy Ethereum wallet as the signer of an :class:`aleo_bridge.Ethereum` connection.

    ``client`` is an authenticated ``privy.PrivyClient`` owned by the caller; ``wallet_id`` and
    ``address`` MUST name the same wallet (the address is what every signature is checked against).
    ``authorization_private_keys`` are the wallet's owner keys when its policy requires them; they
    are passed to Privy per request and never stored anywhere else by the bridge.

    Signs standard EIP-1559 transactions (type 2), or legacy ones when the connection prepared a
    ``gasPrice``. Use it as ``Ethereum(rpc_url, signer=PrivyEvmSigner(...))``.
    """

    legacy_transactions_only = False

    def __init__(self, client: Any, *, wallet_id: str, address: str,
                 authorization_private_keys: Sequence[str] | None = None) -> None:
        self._wallets, self._service = _services(client, "ethereum")
        self.wallet_id = _wallet_id(wallet_id)
        self.address = checksum_address(address, what="Privy wallet address")
        self._request_options = _request_options(authorization_private_keys)

    def __repr__(self) -> str:
        return f"PrivyEvmSigner(wallet_id={self.wallet_id!r}, address={self.address!r})"

    def resolve(self) -> str:
        """Fetch the wallet from Privy and confirm it is an Ethereum wallet at :attr:`address`.

        Optional — construction never contacts Privy. Call it at startup to fail fast on a wallet
        id / address pair that does not name one wallet, instead of at the first signature.
        """
        return _resolve(self._wallets, self.wallet_id, chain_type="ethereum", address=self.address,
                        same=lambda found: checksum_address(found, what="Privy wallet address") == self.address)

    def sign_transaction(self, tx: Mapping[str, Any]) -> RemoteSignedTransaction:
        """One ``eth_signTransaction`` request for the prepared *tx*; the result is verified to
        recover to :attr:`address` before it is handed back for broadcast."""
        fields = unsigned_evm_fields(tx, provider=PROVIDER)
        sender = tx.get("from")
        if sender is not None and checksum_address(sender, what="transaction from") != self.address:
            raise ConfigurationError(f"Transaction sender {sender} does not match the Privy wallet {self.address}")
        transaction: dict[str, Any] = {
            "from": self.address, "to": fields["to"], "value": fields["value"], "chain_id": fields["chainId"],
            "nonce": fields["nonce"], "gas_limit": fields["gas"], "data": fields["data"],
        }
        if "gasPrice" in fields:
            transaction.update({"type": 0, "gas_price": fields["gasPrice"]})
        else:
            transaction.update({"type": 2, "max_fee_per_gas": fields["maxFeePerGas"],
                                "max_priority_fee_per_gas": fields["maxPriorityFeePerGas"]})
        kwargs: dict[str, Any] = {"params": {"transaction": transaction}, "address": self.address}
        if self._request_options is not None:
            kwargs["request_options"] = self._request_options
        response = self._service.sign_transaction(self.wallet_id, **kwargs)
        signed = getattr(response, "signed_transaction", None)
        if not isinstance(signed, str) or not signed.startswith("0x"):
            raise BridgeError("Privy returned no hex signed_transaction for eth_signTransaction")
        try:
            raw = bytes.fromhex(signed[2:])
        except ValueError as exc:
            raise BridgeError("Privy returned a malformed hex signed_transaction") from exc
        return verified_evm_transaction(raw, sender=self.address, provider=PROVIDER)


class PrivySolanaSigner:
    """An existing Privy Solana wallet as the signer of an :class:`aleo_bridge.Solana` connection.

    ``client`` is an authenticated ``privy.PrivyClient``; ``wallet_id`` and ``address`` MUST name the
    same wallet. The wallet is the transfer's fee payer, so it must be the connected sender.

    Privy signs whole transactions, so each request wraps the message the bridge compiled in a
    transaction whose signature slots are empty; the response must carry the identical message, and
    only this wallet's slot is read back (the bridge adds its own ephemeral signature itself). Use it
    as ``Solana(rpc_url, signer=PrivySolanaSigner(...))``.
    """

    def __init__(self, client: Any, *, wallet_id: str, address: str,
                 authorization_private_keys: Sequence[str] | None = None) -> None:
        self._wallets, self._service = _services(client, "solana")
        self.wallet_id = _wallet_id(wallet_id)
        self._pubkey = solana_pubkey(address, what="Privy wallet address")
        self.address = str(self._pubkey)
        self._request_options = _request_options(authorization_private_keys)

    def __repr__(self) -> str:
        return f"PrivySolanaSigner(wallet_id={self.wallet_id!r}, address={self.address!r})"

    def resolve(self) -> str:
        """Fetch the wallet from Privy and confirm it is a Solana wallet at :attr:`address` (optional,
        see :meth:`PrivyEvmSigner.resolve`)."""
        return _resolve(self._wallets, self.wallet_id, chain_type="solana", address=self.address,
                        same=lambda found: found == self.address)

    def pubkey(self) -> Any:
        return self._pubkey

    def sign_message(self, message: bytes) -> Any:
        """The wallet's signature over the versioned *message* bytes, verified before it is returned."""
        from solders.message import to_bytes_versioned

        from .sol import _libs

        libs = _libs()
        message_bytes = bytes(message)
        decoded, index = signer_index(message_bytes, self._pubkey, provider=PROVIDER)
        empty = [libs.Signature.default()] * int(decoded.header.num_required_signatures)
        wire = bytes(libs.VersionedTransaction.populate(decoded, empty))
        kwargs: dict[str, Any] = {"address": self.address}
        if self._request_options is not None:
            kwargs["request_options"] = self._request_options
        response = self._service.sign_transaction(self.wallet_id, wire, **kwargs)
        encoded = getattr(response, "signed_transaction", None)
        if getattr(response, "encoding", "base64") != "base64" or not isinstance(encoded, str):
            raise BridgeError("Privy returned no base64 signed_transaction for signTransaction")
        try:
            signed = libs.VersionedTransaction.from_bytes(base64.b64decode(encoded, validate=True))
        except Exception as exc:  # noqa: BLE001 — base64 or transaction decoding
            raise BridgeError("Privy returned a malformed signed Solana transaction") from exc
        if bytes(to_bytes_versioned(signed.message)) != message_bytes:
            raise BridgeError("Privy changed the Solana transaction message — refusing to broadcast")
        signatures = list(signed.signatures)
        if index >= len(signatures):
            raise BridgeError("Privy returned no signature slot for the Solana wallet")
        return verified_solana_signature(signatures[index], pubkey=self._pubkey, message_bytes=message_bytes,
                                         provider=PROVIDER)


__all__ = ["PrivyEvmSigner", "PrivySolanaSigner"]
