"""Aleo — facade account module (F2).

Attached to the :class:`~aleo.facade.client.Aleo` client as ``aleo.account``.
Wraps the network module's cryptographic primitives behind a clean, Web3.py-
style interface.  All operations are purely local (no network I/O).

.. note::

    **No ``recover`` / signer-from-signature verb.**  Aleo is a privacy chain.
    Surfacing a "which address signed this?" affordance in the facade is a
    de-anonymisation vector.  The low-level ``Signature.to_address()`` primitive
    remains accessible directly; the facade deliberately does not expose it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Import only for type annotations; resolved at runtime via _net().
    pass


class AccountModule:
    """Namespaced account operations attached to an :class:`~aleo.facade.client.Aleo` client.

    Access via ``aleo.account``, not by direct construction.

    Parameters
    ----------
    client:
        The parent :class:`~aleo.facade.client.Aleo` instance.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    # ── Internal helper ────────────────────────────────────────────────────

    def _net(self) -> Any:
        """Return the network module (``aleo.mainnet`` or ``aleo.testnet``).

        Mirrors the pattern used by :pyattr:`Aleo.process` and
        :pyattr:`Aleo.network_id` in ``client.py``.
        """
        network: str = self._client._provider.network
        if network == "testnet":
            import aleo.testnet as _testnet  # type: ignore[attr-defined]
            return _testnet
        import aleo.mainnet as _mainnet
        return _mainnet

    # ── Account creation ───────────────────────────────────────────────────

    def create(self) -> Any:
        """Generate a fresh, random :class:`Account`.

        Returns
        -------
        Account
            A new account with a random private key.

        Example
        -------
        ::

            acct = aleo.account.create()
            print(acct.address)   # aleo1…
        """
        net = self._net()
        return net.Account.random()

    def from_private_key(self, private_key: str | Any) -> Any:
        """Derive an :class:`Account` from an existing private key.

        Parameters
        ----------
        private_key:
            An Aleo private-key string (``"APrivateKey1…"``) or a
            :class:`PrivateKey` object.

        Returns
        -------
        Account
            The account derived from *private_key*.

        Raises
        ------
        ValueError
            If *private_key* is a string that cannot be parsed.
        """
        net = self._net()
        if isinstance(private_key, str):
            pk: Any = net.PrivateKey.from_string(private_key)
        else:
            pk = private_key
        return net.Account.from_private_key(pk)

    def from_seed(self, seed: str | Any) -> Any:
        """Derive an :class:`Account` from a seed :class:`Field` element.

        Parameters
        ----------
        seed:
            A :class:`Field` object, or a field-element string accepted by
            ``Field.from_string`` (e.g. ``"123field"``).

        Returns
        -------
        Account
            The account whose private key is derived from *seed*.

        Raises
        ------
        ValueError
            If *seed* is a string that cannot be parsed as a ``Field``.
        """
        net = self._net()
        if isinstance(seed, str):
            field: Any = net.Field.from_string(seed)
        else:
            field = seed
        pk: Any = net.PrivateKey.from_seed(field)
        return net.Account.from_private_key(pk)

    # ── Encryption ─────────────────────────────────────────────────────────

    def export_encrypted(self, account: Any, secret: str) -> Any:
        """Encrypt *account*'s private key with *secret*.

        Uses :class:`~aleo.mainnet.PrivateKeyCiphertext` (the Rust-backed
        symmetric encryption primitive) to produce a portable ciphertext.

        Parameters
        ----------
        account:
            An :class:`Account` instance whose private key is to be encrypted.
        secret:
            A passphrase string.

        Returns
        -------
        PrivateKeyCiphertext
            An opaque ciphertext object.  ``str(ct)`` yields the serialised
            form suitable for storage.
        """
        return account.private_key.to_ciphertext(secret)

    def import_encrypted(self, ciphertext: str | Any, secret: str) -> Any:
        """Decrypt a :class:`PrivateKeyCiphertext` and return the :class:`Account`.

        Parameters
        ----------
        ciphertext:
            A :class:`PrivateKeyCiphertext` object, or its serialised string
            representation.
        secret:
            The passphrase used to encrypt.

        Returns
        -------
        Account
            The account whose private key was encrypted in *ciphertext*.

        Raises
        ------
        ValueError
            If *ciphertext* is malformed or *secret* is incorrect.
        """
        net = self._net()
        if isinstance(ciphertext, str):
            ct: Any = net.PrivateKeyCiphertext.from_string(ciphertext)
        else:
            ct = ciphertext
        pk: Any = net.PrivateKey.from_private_key_ciphertext(ct, secret)
        return net.Account.from_private_key(pk)

    # ── Signing ────────────────────────────────────────────────────────────

    def sign(self, message: bytes, account: Any = None) -> Any:
        """Sign *message* with *account*'s private key.

        When *account* is omitted the client's
        :pyattr:`~aleo.facade.client.Aleo.default_account` is used.

        Parameters
        ----------
        message:
            Raw bytes to sign.
        account:
            An :class:`Account` to sign with.  Defaults to
            ``aleo.default_account`` when ``None``.

        Returns
        -------
        Signature
            An Aleo :class:`Signature` object.  ``str(sig)`` yields the
            serialised form (``"sign1…"``).

        Raises
        ------
        ValueError
            If both *account* and ``aleo.default_account`` are ``None``.
        """
        acct = self._resolve_account(account)
        return acct.sign(message)

    def verify(self, address: str | Any, message: bytes, signature: str | Any) -> bool:
        """Verify that *signature* over *message* was produced by *address*.

        Parameters
        ----------
        address:
            An Aleo address string (``"aleo1…"``) or :class:`Address` object.
        message:
            The raw bytes that were signed.
        signature:
            A :class:`Signature` object or serialised string (``"sign1…"``).

        Returns
        -------
        bool
            ``True`` if the signature is valid; ``False`` otherwise.
        """
        net = self._net()
        addr: Any = (
            net.Address.from_string(address)
            if isinstance(address, str)
            else address
        )
        sig: Any = (
            net.Signature.from_string(signature)
            if isinstance(signature, str)
            else signature
        )
        return bool(sig.verify(addr, message))

    # ── Structured (Value) signing ─────────────────────────────────────────

    def sign_value(self, value: str, account: Any = None) -> Any:
        """Sign a structured Aleo *value* string.

        The ``sign_typed_data`` analog for Aleo.  Wraps
        ``PrivateKey.sign_value`` which signs an Aleo ``Value``
        (e.g. ``"100u64"``, ``"true"``, a record literal).

        When *account* is omitted the client's
        :pyattr:`~aleo.facade.client.Aleo.default_account` is used.

        Parameters
        ----------
        value:
            An Aleo ``Value`` string (e.g. ``"100u64"``).
        account:
            An :class:`Account` to sign with.  Defaults to
            ``aleo.default_account`` when ``None``.

        Returns
        -------
        Signature
            An Aleo :class:`Signature` object.

        Raises
        ------
        ValueError
            If both *account* and ``aleo.default_account`` are ``None``, or if
            *value* is not a valid Aleo ``Value`` string.
        """
        acct = self._resolve_account(account)
        return acct.private_key.sign_value(value)

    def verify_value(
        self,
        address: str | Any,
        value: str,
        signature: str | Any,
    ) -> bool:
        """Verify that *signature* over *value* was produced by *address*.

        Parameters
        ----------
        address:
            An Aleo address string (``"aleo1…"``) or :class:`Address` object.
        value:
            The Aleo ``Value`` string that was signed (e.g. ``"100u64"``).
        signature:
            A :class:`Signature` object or serialised string (``"sign1…"``).

        Returns
        -------
        bool
            ``True`` if the signature is valid; ``False`` otherwise.
        """
        net = self._net()
        addr: Any = (
            net.Address.from_string(address)
            if isinstance(address, str)
            else address
        )
        sig: Any = (
            net.Signature.from_string(signature)
            if isinstance(signature, str)
            else signature
        )
        return bool(sig.verify_value(addr, value))

    # ── Internal ───────────────────────────────────────────────────────────

    def _resolve_account(self, account: Any) -> Any:
        """Return *account* if provided, else ``client.default_account``.

        Raises :exc:`ValueError` when both are ``None``.
        """
        if account is not None:
            return account
        default: Any = self._client.default_account
        if default is None:
            raise ValueError(
                "No account provided and aleo.default_account is not set. "
                "Pass an account explicitly or set aleo.default_account first."
            )
        return default


__all__ = ["AccountModule"]
