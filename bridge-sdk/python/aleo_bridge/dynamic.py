"""Dynamic server wallets as bridge signers (the ``dynamic-wallet-sdk`` SDK ships with ``aleo-bridge-sdk``
on Python 3.11 and newer, the oldest Python that SDK supports).

An existing Dynamic MPC wallet signs the Ethereum or Solana leg of a transfer; the bridge keeps
building, broadcasting, checkpointing and recovering exactly as with a local key::

    from dynamic_wallet_sdk import DynamicEvmWalletClient, DynamicSvmWalletClient
    from aleo_bridge import Bridge, Ethereum, Solana
    from aleo_bridge.dynamic import DynamicEvmSigner, DynamicSolanaSigner

    ethereum = Ethereum(rpc_url, signer=DynamicEvmSigner(
        DynamicEvmWalletClient(environment_id), address=evm_address, api_token=api_token, password=evm_password))
    solana = Solana(rpc_url, signer=DynamicSolanaSigner(
        DynamicSvmWalletClient(environment_id), address=sol_address, api_token=api_token, password=sol_password))
    bridge = Bridge(aleo, ethereum=ethereum, solana=solana)

The Dynamic Python SDK is ``async``-only; the signers run it on a private event-loop thread, so
pass ``api_token=`` and let the signer authenticate the client on that thread (a client you
authenticated on another loop may hold connections that loop has since closed). The first signature
request authenticates and resolves the wallet by address — its chain and (when given) ``wallet_id``
must match — and a mismatch is refused before anything is signed. The session token Dynamic issues
is short-lived, so a signer re-authenticates on a schedule (``reauthenticate_seconds``, an hour by
default) instead of retrying a failed signature. Every signature is
verified locally before broadcast: an Ethereum transaction must recover to the configured address, a
Solana signature must verify over the exact message the bridge built.

Dynamic's EVM signing supports legacy transactions only, so :class:`DynamicEvmSigner` advertises
``legacy_transactions_only`` and :class:`aleo_bridge.Ethereum` prepares a ``gasPrice`` transaction
for it instead of an EIP-1559 one. Passwords and key shares are forwarded to Dynamic per request and
never written by the bridge; the signers add no retries of their own.
"""
from __future__ import annotations

import time
from typing import Any, Awaitable, Mapping, cast

from ._remote_signing import (RemoteSignedTransaction, checksum_address, legacy_transaction_from_signature,
                              run_async, signer_index, solana_pubkey, unsigned_evm_fields,
                              verified_solana_signature)
from .errors import BridgeError, ConfigurationError

PROVIDER = "Dynamic"

DEFAULT_SIGNING_TIMEOUT_SECONDS = 120.0
"""How long one authenticate / lookup / MPC signing round may take before the signer gives up.
The transaction is then simply unsigned; nothing was broadcast."""

DEFAULT_REAUTHENTICATE_SECONDS = 3600.0
"""Age after which a signer built with ``api_token=`` exchanges the token again before signing.
Dynamic's session JWT (observed 2026-10-01) lives two hours and the SDK has no refresh path, so a
long-lived signer re-authenticates on a schedule rather than retrying a failed signature."""


class _DynamicSigner:
    """What the EVM and Solana signers share: lazy authentication, wallet lookup and identity checks."""

    chain_names: tuple[str, ...] = ()

    def __init__(self, client: Any, *, address: str, api_token: str | None, password: str | None,
                 key_shares: Any, wallet_id: str | None, timeout_seconds: float,
                 reauthenticate_seconds: float) -> None:
        for method in ("load_wallet", "sign_transaction", "authenticate_api_token"):
            if not callable(getattr(client, method, None)):
                raise ConfigurationError(
                    f"client must be a dynamic_wallet_sdk wallet client with {method}() — the dynamic-wallet-sdk "
                    "package aleo-bridge-sdk installs on Python 3.11 and newer")
        if api_token is not None and not api_token.strip():
            raise ConfigurationError("Dynamic api_token must not be empty when given")
        if wallet_id is not None and not wallet_id.strip():
            raise ConfigurationError("Dynamic wallet_id must not be empty when given")
        if not password and not key_shares:
            raise ConfigurationError(
                "Dynamic signing needs the wallet's key shares: pass password= for shares backed up to Dynamic, "
                "or key_shares= for caller-managed shares")
        self._client = client
        self.address = address
        self._api_token = api_token
        self._password = password
        self._key_shares = key_shares
        self.wallet_id = wallet_id
        self.timeout_seconds = float(timeout_seconds)
        self.reauthenticate_seconds = float(reauthenticate_seconds)
        self._ready = False
        self._authenticated_at: float | None = None

    def _same_address(self, found: str) -> bool:
        raise NotImplementedError

    async def _authenticate(self) -> None:
        await self._client.authenticate_api_token(self._api_token)
        self._authenticated_at = time.monotonic()

    async def _prepare(self) -> None:
        if self._api_token is not None:
            await self._authenticate()
        wallet = await self._client.load_wallet(self.address)
        chain = str(getattr(wallet, "chain_name", ""))
        found = str(getattr(wallet, "account_address", ""))
        found_id = str(getattr(wallet, "wallet_id", "") or "") or None
        if chain not in self.chain_names or not self._same_address(found):
            raise ConfigurationError(
                f"Dynamic resolved {self.address} to a {chain or 'unknown'} wallet at {found or 'no address'}; "
                "it does not match the configured chain and address")
        if self.wallet_id is not None and found_id != self.wallet_id:
            raise ConfigurationError(
                f"Dynamic wallet at {self.address} has id {found_id}, not the configured wallet_id {self.wallet_id}")
        self.wallet_id = found_id or self.wallet_id

    def _ensure_ready(self) -> None:
        """Authenticate (when an API token was given) and resolve the wallet once per signer; exchange
        the token again before signing once the session is older than ``reauthenticate_seconds``."""
        if not self._ready:
            run_async(self._prepare(), timeout=self.timeout_seconds)
            self._ready = True
        elif (self._api_token is not None and self._authenticated_at is not None
              and time.monotonic() - self._authenticated_at >= self.reauthenticate_seconds):
            run_async(self._authenticate(), timeout=self.timeout_seconds)

    def resolve(self) -> str | None:
        """Authenticate and resolve the wallet now, returning its Dynamic wallet id (``None`` when
        the provider reports none).

        Optional — the first signature does the same. Call it at startup to fail fast on a wrong
        token, a missing wallet, a wrong chain or a mismatched ``wallet_id`` instead of at the first
        transfer.
        """
        self._ensure_ready()
        return self.wallet_id

    def _sign_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self._password:
            kwargs["password"] = self._password
        if self._key_shares:
            kwargs["key_shares"] = self._key_shares
        return kwargs

    def close(self) -> None:
        """Close the provider client's HTTP connections on the signing thread (idempotent)."""
        closer = getattr(self._client, "close", None)
        if callable(closer):
            run_async(cast("Awaitable[Any]", closer()), timeout=self.timeout_seconds)


class DynamicEvmSigner(_DynamicSigner):
    """An existing Dynamic EVM wallet as the signer of an :class:`aleo_bridge.Ethereum` connection.

    ``client`` is a ``dynamic_wallet_sdk.DynamicEvmWalletClient``; ``address`` is the wallet to sign
    with; ``api_token`` authenticates the client on the signing thread; ``password`` recovers shares
    backed up to Dynamic, or ``key_shares`` supplies caller-managed ones; an optional ``wallet_id``
    is checked against what Dynamic resolves for the address.

    Dynamic returns a raw ``r ‖ s ‖ v`` signature over a legacy transaction, so the signer assembles
    the EIP-155 transaction itself and verifies that it recovers to ``address``. Use it as
    ``Ethereum(rpc_url, signer=DynamicEvmSigner(...))``.
    """

    chain_names = ("EVM",)
    legacy_transactions_only = True

    def __init__(self, client: Any, *, address: str, api_token: str | None = None, password: str | None = None,
                 key_shares: Any = None, wallet_id: str | None = None,
                 timeout_seconds: float = DEFAULT_SIGNING_TIMEOUT_SECONDS,
                 reauthenticate_seconds: float = DEFAULT_REAUTHENTICATE_SECONDS) -> None:
        super().__init__(client, address=checksum_address(address, what="Dynamic wallet address"),
                         api_token=api_token, password=password, key_shares=key_shares, wallet_id=wallet_id,
                         timeout_seconds=timeout_seconds, reauthenticate_seconds=reauthenticate_seconds)

    def __repr__(self) -> str:
        return f"DynamicEvmSigner(address={self.address!r})"

    def _same_address(self, found: str) -> bool:
        return found.lower() == self.address.lower()

    def sign_transaction(self, tx: Mapping[str, Any]) -> RemoteSignedTransaction:
        """One MPC signing round for the prepared legacy *tx*; the assembled transaction is verified
        to recover to :attr:`address` before it is handed back for broadcast."""
        fields = unsigned_evm_fields(tx, provider=PROVIDER)
        if "gasPrice" not in fields:
            raise ConfigurationError("Dynamic signs legacy transactions only: prepare gasPrice, not EIP-1559 fees")
        sender = tx.get("from")
        if sender is not None and checksum_address(sender, what="transaction from") != self.address:
            raise ConfigurationError(f"Transaction sender {sender} does not match the Dynamic wallet {self.address}")
        self._ensure_ready()
        unsigned = {"to": fields["to"], "value": fields["value"], "nonce": fields["nonce"], "gas": fields["gas"],
                    "gasPrice": fields["gasPrice"], "chainId": fields["chainId"], "data": fields["data"]}
        signature = run_async(self._client.sign_transaction(self.address, unsigned, **self._sign_kwargs()),
                              timeout=self.timeout_seconds)
        if not isinstance(signature, str) or not signature.startswith("0x"):
            raise BridgeError("Dynamic returned no hex signature for the Ethereum transaction")
        try:
            raw_signature = bytes.fromhex(signature[2:])
        except ValueError as exc:
            raise BridgeError("Dynamic returned a malformed hex Ethereum signature") from exc
        return legacy_transaction_from_signature(fields, raw_signature, sender=self.address, provider=PROVIDER)


class DynamicSolanaSigner(_DynamicSigner):
    """An existing Dynamic Solana wallet as the signer of an :class:`aleo_bridge.Solana` connection.

    ``client`` is a ``dynamic_wallet_sdk.DynamicSvmWalletClient``; the other arguments are as for
    :class:`DynamicEvmSigner`. Dynamic signs the message bytes directly; the returned signature is
    verified against the wallet's public key before it is attached. Use it as
    ``Solana(rpc_url, signer=DynamicSolanaSigner(...))``.
    """

    chain_names = ("SVM", "SOL")

    def __init__(self, client: Any, *, address: str, api_token: str | None = None, password: str | None = None,
                 key_shares: Any = None, wallet_id: str | None = None,
                 timeout_seconds: float = DEFAULT_SIGNING_TIMEOUT_SECONDS,
                 reauthenticate_seconds: float = DEFAULT_REAUTHENTICATE_SECONDS) -> None:
        self._pubkey = solana_pubkey(address, what="Dynamic wallet address")
        super().__init__(client, address=str(self._pubkey), api_token=api_token, password=password,
                         key_shares=key_shares, wallet_id=wallet_id, timeout_seconds=timeout_seconds,
                         reauthenticate_seconds=reauthenticate_seconds)

    def __repr__(self) -> str:
        return f"DynamicSolanaSigner(address={self.address!r})"

    def _same_address(self, found: str) -> bool:
        # Dynamic's address lookup can echo a base58 address case-folded; the Python SDK keeps the
        # caller's spelling as canonical, and the signature check below is what proves the key.
        return found == self.address or found.lower() == self.address.lower()

    def pubkey(self) -> Any:
        return self._pubkey

    def sign_message(self, message: bytes) -> Any:
        """The wallet's signature over the versioned *message* bytes, verified before it is returned."""
        message_bytes = bytes(message)
        signer_index(message_bytes, self._pubkey, provider=PROVIDER)      # a required signer, before any request
        self._ensure_ready()
        signature = run_async(self._client.sign_transaction(self.address, message_bytes, **self._sign_kwargs()),
                              timeout=self.timeout_seconds)
        if not isinstance(signature, str):
            raise BridgeError("Dynamic returned no hex signature for the Solana transaction")
        try:
            raw_signature = bytes.fromhex(signature[2:] if signature.startswith("0x") else signature)
        except ValueError as exc:
            raise BridgeError("Dynamic returned a malformed hex Solana signature") from exc
        return verified_solana_signature(raw_signature, pubkey=self._pubkey, message_bytes=message_bytes,
                                         provider=PROVIDER)


__all__ = ["DEFAULT_REAUTHENTICATE_SECONDS", "DEFAULT_SIGNING_TIMEOUT_SECONDS", "DynamicEvmSigner",
           "DynamicSolanaSigner"]
