"""Bind a participant's home directory to one Aleo account.

One profile per home directory (``$SHIELD_SWAP_HOME`` or ``~/.shield-swap``).
``profile.json`` records which Aleo account the profile trades with: its
address and the name of its account file. The private key is held by the Aleo
account keystore (``aleo.keystore``, by default
``~/.aleo/accounts/<name>.json``). Every Aleo tool reads that keystore, so an
account set up once serves them all. Credentials (JWT, delegated-proving keys)
are stored separately so they can be refreshed without touching the account.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional, cast

DEFAULT_ENDPOINT = "https://edge.provable.com/api"
_PROFILE = "profile.json"
_CREDENTIALS = "credentials.json"

_log = logging.getLogger(__name__)

# Profile files already reported as holding their own key. The notice is given
# once per process, however many Profile objects load the file.
_legacy_notices: set[str] = set()
_legacy_notices_lock = threading.Lock()


def _write_private(path: Path, payload: dict[str, Any]) -> None:
    """Owner-only file, written atomically (no umask window, no torn reads)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(payload, indent=1))
    os.replace(tmp, path)


def _selected_account_name(account: Optional[str]) -> str:
    """Return the account name a lookup selects when the profile records none.

    Applies the account keystore's own order: the *account* argument, then
    ``ALEO_ACCOUNT`` (an empty value counts as unset), then ``default``.
    """
    return account if account is not None else (os.environ.get("ALEO_ACCOUNT") or "default")


def _report_legacy_key(path: Path, address: str) -> None:
    """Log, once per process and profile file, that the file holds the key itself.

    The warning names the command that copies the key into an account file.
    """
    with _legacy_notices_lock:
        if str(path) in _legacy_notices:
            return
        _legacy_notices.add(str(path))
    _log.warning(
        "The private key of %s is still stored in %s; `python -m aleo.keystore migrate --from %s` "
        "copies it into an account file shared by the Aleo tools (add --remove-legacy-key to take "
        "it out of profile.json).",
        address, path, path)


class Profile:
    """A participant's persistent identity and credentials.

    Holds exactly one Aleo address, fixed when the profile is created and
    reused every session after.  For several addresses, create a profile per
    address — each needs its own home directory (``SHIELD_SWAP_HOME``, or
    *home* on :meth:`load_or_create`), since the journal and credentials are
    per-profile too — and give each its own account name.

    Load with :meth:`load_or_create`.  The same call serves the first run and
    every later one, so callers never branch on whether a profile exists::

        from aleo_shield_swap import Profile

        # First run: uses the stored Aleo account, or creates and stores one.
        profile = Profile.load_or_create()
        print(profile.address)                  # aleo1…

        # Every run after: the same call returns that account, unchanged.
        assert Profile.load_or_create().address == profile.address

        # A second address needs its own home and its own account name.
        other = Profile.load_or_create("~/.shield-swap-alt", account="alt")

    The private key is not in the profile.  It is found through the Aleo
    account keystore: the ``ALEO_PRIVATE_KEY`` or ``ALEO_PRIVATE_KEY_FILE``
    variables for a key supplied per run, otherwise the account file the
    profile names.  A key from the environment is used for the process and
    never written to disk.  Whichever source supplies it, the key must belong
    to the profile's address.
    """

    def __init__(self, home: Path, data: dict[str, Any]) -> None:
        self.home = home
        self._data = data
        # How to find the key of an existing profile, as given to load_or_create().
        self._account: Optional[str] = None
        self._accounts_dir: "Path | str | None" = None
        self._password: Optional[str] = None
        # The key, once resolved; one lookup serves the object's lifetime.
        self._private_key: Optional[str] = None
        self._key_lock = threading.Lock()

    def __repr__(self) -> str:
        return f"Profile({self.address!r}, home={str(self.home)!r})"

    # ── Construction ────────────────────────────────────────────────────────

    @staticmethod
    def default_home() -> Path:
        """Where profiles live by default: ``$SHIELD_SWAP_HOME`` or ``~/.shield-swap``.

        Local only — returns the path whether or not it exists.  A ``~`` in
        ``SHIELD_SWAP_HOME`` is expanded, so a value set from a config file that
        does not shell-expand still resolves to the home directory rather than a
        literal ``~`` folder in the working directory.
        """
        env = os.environ.get("SHIELD_SWAP_HOME")
        return Path(env).expanduser() if env else Path.home() / ".shield-swap"

    @classmethod
    def load_or_create(cls, home: "Path | str | None" = None, *,
                       network: str = "testnet",
                       endpoint: str = DEFAULT_ENDPOINT,
                       create: bool = True,
                       storage: str = "plaintext",
                       account: Optional[str] = None,
                       accounts_dir: "Path | str | None" = None,
                       password: Optional[str] = None,
                       save_password: bool = False) -> "Profile":
        """Load the profile at *home*, binding one to an Aleo account if absent.

        A new profile takes the account the Aleo account keystore resolves: a
        key in ``ALEO_PRIVATE_KEY`` or the file named by
        ``ALEO_PRIVATE_KEY_FILE`` (``SHIELD_SWAP_PRIVATE_KEY`` and
        ``SHIELD_SWAP_PRIVATE_KEY_FILE`` are aliases), otherwise the account
        file selected by *account*.  A user with an existing account supplies
        it one of these ways rather than pasting a key into a chat.  When
        neither exists, a new account is created and stored in the keystore
        with the chosen *storage*; pass ``create=False`` to raise instead.  A
        key from the environment is used for this process and never written
        to disk, so it has to be supplied again on every later run.

        ``profile.json`` then records the address, the account name, the
        network and the endpoint, at mode 600.  It holds no private key.

        An existing profile is returned as recorded: *network*, *endpoint*,
        *create*, *storage* and *save_password* apply only when creating.  Its
        key is looked up on the first read of :attr:`private_key`, using
        *account*, *accounts_dir* and *password* as given here.  A profile
        whose ``profile.json`` holds a ``private_key`` keeps working: that key
        is used when no other source supplies one.

        Reads the environment and the account file.  Creating also writes the
        account file and the profile, and for ``"keychain"`` or
        ``"encrypted"`` storage writes the macOS Keychain or asks for a
        password on an attached terminal.  Does not contact a network.

        Args:
            home: Profile directory; defaults to :meth:`default_home`.
            network: Network to bind a NEW profile to.
            endpoint: API endpoint to record on a NEW profile.
            create: Whether to create and store an account when a NEW profile
                finds none.  Defaults to True.
            storage: Where the keystore holds a newly created key.
                ``"plaintext"`` (the default) writes it to a file only its
                owner can read, so an unattended process starts without a
                password.  ``"keychain"`` holds it in the macOS login
                keychain.  ``"encrypted"`` holds it under a password on every
                system, and each later run needs that password.  ``"none"``
                is rejected, because nothing would hold the new key once the
                process exits.
            account: Account name in the keystore.  Defaults to
                ``ALEO_ACCOUNT``, then the name the profile records, then
                ``default``.
            accounts_dir: Accounts directory.  Defaults to
                ``ALEO_ACCOUNTS_DIR``, then ``~/.aleo/accounts``.  The profile
                does not record it; pass it again on later runs.
            password: Password of an ``"encrypted"`` account.  Defaults to
                ``ALEO_ACCOUNT_PASSWORD``, a password saved in the macOS
                Keychain, then a hidden prompt on an attached terminal.
            save_password: Also save the password of a newly created
                ``"encrypted"`` account in the macOS Keychain, so later runs
                need no password.  Defaults to False.

        Returns:
            The loaded or newly created profile.

        Raises:
            aleo.keystore.KeystoreError: Read ``code``.  For a NEW profile:
                ``ACCOUNT_NOT_FOUND`` when no account resolves and *create* is
                False; ``UNSUPPORTED_STORAGE`` when an account would be
                created with ``"none"`` or an unknown *storage*;
                ``EMPTY_KEY_SOURCE``, ``KEY_CONFLICT`` or ``INVALID_KEY`` for
                unusable key variables; ``ADDRESS_MISMATCH`` when a key from
                the environment belongs to another address than the selected
                account file; ``KEY_NOT_STORED`` when that file records an
                address only and no key was supplied; ``PASSWORD_REQUIRED`` or
                ``WRONG_PASSWORD`` for an encrypted account;
                ``KEYCHAIN_UNAVAILABLE`` for ``"keychain"`` outside macOS;
                ``INVALID_NAME`` for an unusable *account*.
        """
        # expanduser: a "~/..." string is otherwise taken literally, creating a
        # directory named ~ in the cwd and writing the profile there.
        home = Path(home).expanduser() if home is not None else cls.default_home()
        path = home / _PROFILE
        if path.exists():
            path.chmod(0o600)             # heal a pre-existing loose mode
            profile = cls(home, json.loads(path.read_text()))
            # The key is looked up when it is first read, with these options.
            profile._account = account
            profile._accounts_dir = accounts_dir
            profile._password = password
        else:
            import aleo
            from aleo import keystore
            net = getattr(aleo, network)
            try:
                loaded = keystore.load(name=account, dir=accounts_dir, password=password)
                private_key, address = loaded.private_key, loaded.address
                # A key from the environment needs no account file; the name
                # the lookup selected is recorded all the same.
                name = loaded.stored.name if loaded.stored is not None else _selected_account_name(account)
            except keystore.KeystoreError as error:
                if error.code != "ACCOUNT_NOT_FOUND" or not create:
                    raise
                if storage == "none":
                    raise keystore.KeystoreError(
                        "UNSUPPORTED_STORAGE",
                        'A new account cannot be created with storage "none": nothing would hold its key '
                        'after this process exits. Choose "plaintext", "keychain" or "encrypted".',
                    ) from None
                private_key = str(net.PrivateKey.random())
                stored = keystore.save(private_key, name=account, dir=accounts_dir,
                                       storage=cast(Any, storage),   # save() rejects an unknown value
                                       password=password, save_password=save_password)
                address, name = stored.address, stored.name
            home.mkdir(parents=True, exist_ok=True)
            data = {"address": address, "account": name,
                    "network": network, "endpoint": endpoint}
            _write_private(path, data)
            profile = cls(home, data)
            profile._private_key = private_key
        return profile

    # ── Identity ─────────────────────────────────────────────────────────────

    @property
    def address(self) -> str:
        """The profile's Aleo address (``aleo1…``) — the public half of its key."""
        return self._data["address"]

    @property
    def private_key(self) -> str:
        """The private key of the profile's account.

        Whoever holds this controls the account and can decrypt its records — do
        not log it, echo it into a conversation, or send it to a service.

        The key is looked up on first use and kept for the lifetime of this
        object: from ``ALEO_PRIVATE_KEY`` or ``ALEO_PRIVATE_KEY_FILE`` when
        set, otherwise from the account file named by the *account* argument
        of :meth:`load_or_create`, then ``ALEO_ACCOUNT``, then the name the
        profile records, then ``default``.  A ``private_key`` held in
        ``profile.json`` is used when no other source has one, and a warning
        is logged once with the command that moves it into an account file.
        Whichever source supplies the key, it must belong to :attr:`address`.

        The first read may run the macOS ``security`` tool or, for an
        encrypted account, stretch a password (about 200 ms) and ask for it
        on an attached terminal.

        Raises:
            aleo.keystore.KeystoreError: Read ``code``.  ``ADDRESS_MISMATCH``
                when the key found belongs to another address; the message
                names both.  ``ACCOUNT_NOT_FOUND`` when no source has a key,
                such as a profile created from a key in the environment that
                is no longer set.  ``PASSWORD_REQUIRED`` when the account is
                encrypted and no password was passed, set in
                ``ALEO_ACCOUNT_PASSWORD``, saved in the Keychain or typed at
                a terminal; ``WRONG_PASSWORD`` when it does not unlock the
                key.  ``EMPTY_KEY_SOURCE`` or ``KEY_CONFLICT`` for key
                variables that are empty or disagree.
        """
        if self._private_key is None:
            with self._key_lock:
                if self._private_key is None:
                    self._private_key = self._resolve_private_key()
        return self._private_key

    def _resolve_private_key(self) -> str:
        """Find the key of an existing profile through the account keystore."""
        from aleo import keystore

        # The keystore lets its name argument override ALEO_ACCOUNT, so the
        # recorded name is passed only when neither the caller nor the
        # variable selects an account.
        name = self._account
        if name is None and not os.environ.get("ALEO_ACCOUNT"):
            name = self._data.get("account")
        loaded = keystore.load(
            name=name, dir=self._accounts_dir,
            # A profile file may hold the key itself; it is the last source.
            legacy_key=self._data.get("private_key") or None,
            expected_address=self.address,
            password=self._password)
        if loaded.source == "legacy":
            _report_legacy_key(self.home / _PROFILE, self.address)
        self._password = None             # the key is held now; the password is not needed again
        return loaded.private_key

    @property
    def network(self) -> str:
        """Network this profile is bound to; fixed when the profile was created.

        Key derivations are network-scoped, so reusing a profile against the other
        network yields keys that do not match anything on chain.
        """
        return self._data["network"]

    @property
    def endpoint(self) -> str:
        """API endpoint recorded for this profile, or the package default.

        Older profiles predate the stored field and fall back to the default.
        """
        return self._data.get("endpoint", DEFAULT_ENDPOINT)

    # ── Credentials ──────────────────────────────────────────────────────────

    @property
    def credentials(self) -> dict[str, str]:
        """DEX credentials saved for this profile — API tokens and session data.

        Re-read from disk on every access, so a value written by another process
        is picked up. A missing file reads as an empty dict rather than raising.
        """
        path = self.home / _CREDENTIALS
        return json.loads(path.read_text()) if path.exists() else {}

    def save_credentials(self, **kv: Optional[str]) -> None:
        """Merge non-None values into ``credentials.json`` (mode 600)."""
        merged = {**self.credentials, **{k: v for k, v in kv.items() if v}}
        _write_private(self.home / _CREDENTIALS, merged)

    def forget_credentials(self, *keys: str) -> None:
        """Drop *keys* from ``credentials.json``; absent keys are ignored."""
        merged = {k: v for k, v in self.credentials.items() if k not in keys}
        _write_private(self.home / _CREDENTIALS, merged)

    @property
    def journal_path(self) -> Path:
        """Where this profile's :class:`~aleo_shield_swap.journal.Journal` lives.

        Returns the path whether or not the file exists — the journal creates it
        on first append.
        """
        return self.home / "journal.jsonl"
