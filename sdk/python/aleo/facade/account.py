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

from ..keystore._mnemonic import derive_seed, generate_phrase

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

    def __repr__(self) -> str:
        return f"AccountModule(network={self._client._provider.network!r})"

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

        Sets ``aleo.default_account`` when no default is configured. Later
        imports leave an existing default unchanged. Does not contact the network.

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
        account = net.Account.from_private_key(pk)
        if self._client.default_account is None:
            self._client.default_account = account
        return account

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

    def from_mnemonic(
        self,
        mnemonic: str,
        *,
        index: int = 0,
        derivation: str = "standard",
    ) -> Any:
        """Derive an :class:`Account` from a recovery phrase.

        Recovers an account created by a wallet from its BIP-39 phrase, using
        the derivation the Shield wallet and the other Aleo SDKs use, so the
        same phrase gives the same account everywhere. One phrase holds many
        accounts; *index* selects among them.

        The phrase is checked first, so a mistyped word is reported instead of
        silently producing a different account. Works offline and stores
        nothing: the caller keeps the phrase, which is the only way to derive
        the account again. Does not change ``aleo.default_account``.

        Parameters
        ----------
        mnemonic:
            A BIP-39 English phrase of 12, 15, 18, 21 or 24 lower-case words
            separated by single spaces.
        index:
            Which account of the phrase to derive, from 0 through
            ``2**31 - 1``. Defaults to 0, the first account a wallet shows.
        derivation:
            ``"standard"`` (the default) follows the registered Aleo path
            ``m/44'/683'/<index>'/0'``. ``"legacy"`` follows
            ``m/44'/0'/<index>'/0'``; choose it only to recover an account
            from a wallet that predates the registration.

        Returns
        -------
        Account
            The account at *index* of the phrase.

        Raises
        ------
        aleo.keystore.KeystoreError
            With ``code == "INVALID_MNEMONIC"`` when a word is not in the
            BIP-39 English word list, the word count is wrong, or the checksum
            does not match. The message never repeats a word of the phrase.
        ValueError
            If *index* or *derivation* is outside the accepted values.

        Example
        -------
        ::

            first = aleo.account.from_mnemonic(phrase)
            second = aleo.account.from_mnemonic(phrase, index=1)
        """
        net = self._net()
        seed = derive_seed(mnemonic, index=index, derivation=derivation)
        pk: Any = net.PrivateKey.from_seed_unchecked(list(seed))
        return net.Account.from_private_key(pk)

    def generate_mnemonic(self, strength: int = 128) -> str:
        """Create a new recovery phrase.

        Draws the randomness from the operating system and encodes it as a
        BIP-39 English phrase. Works offline and stores nothing.

        The phrase is the root secret of every account derived from it: whoever
        holds it controls them all. Show it to the person who will keep it and
        nowhere else: do not log it, write it to a file left on disk, or send
        it to a service.

        Parameters
        ----------
        strength:
            Bits of randomness: 128 (the default) gives 12 words, 256 gives 24.

        Returns
        -------
        str
            The phrase, as lower-case words separated by single spaces. Pass it
            to :meth:`from_mnemonic` to derive its accounts.

        Raises
        ------
        ValueError
            If *strength* is neither 128 nor 256.

        Example
        -------
        ::

            phrase = aleo.account.generate_mnemonic()
            account = aleo.account.from_mnemonic(phrase)
        """
        return generate_phrase(strength)

    def create_with_mnemonic(
        self,
        *,
        strength: int = 128,
        index: int = 0,
        derivation: str = "standard",
    ) -> "tuple[str, Any]":
        """Create a new recovery phrase and derive its account in one call.

        Use this for a new account that a wallet must be able to import later:
        the wallet derives the same account from the phrase. :meth:`create`
        makes an account from a random key instead, which has no phrase.

        Works offline and stores nothing. The caller must give the phrase to
        the person who will keep it; it is the only way to derive the account
        again, and it controls every account of the phrase. Does not change
        ``aleo.default_account``.

        Parameters
        ----------
        strength:
            Bits of randomness: 128 (the default) gives 12 words, 256 gives 24.
        index:
            Which account of the phrase to derive, from 0 through
            ``2**31 - 1``. Defaults to 0, the first account a wallet shows.
        derivation:
            ``"standard"`` (the default) or ``"legacy"``, as in
            :meth:`from_mnemonic`.

        Returns
        -------
        tuple[str, Account]
            The new phrase and the account derived from it.

        Raises
        ------
        ValueError
            If *strength*, *index* or *derivation* is outside the accepted
            values.

        Example
        -------
        ::

            phrase, account = aleo.account.create_with_mnemonic()
            # show `phrase` to the user once; `account.address` is ready to use
        """
        phrase = generate_phrase(strength)
        return phrase, self.from_mnemonic(phrase, index=index, derivation=derivation)

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

        Notes
        -----
        Aleo private keys are not network-parametrised, so the produced
        ciphertext is network-agnostic and interoperates with
        :meth:`import_encrypted` on either network.  The caller is
        responsible for supplying an *account* they intend to encrypt.
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
