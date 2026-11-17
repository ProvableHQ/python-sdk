"""Keep a trading account and its saved work available between runs.

A profile ties one Aleo address to a directory containing API credentials
and a swap journal. Reuse that directory to continue trading and collect
pending swaps. The default is ``SHIELD_SWAP_HOME``, then ``~/.shield-swap``.

New profiles store the address and account name in ``profile.json``. The
private key is loaded through ``aleo.keystore`` or supplied through the
environment. Use macOS Keychain or encrypted account storage to protect a
saved key; see :meth:`Profile.load_or_create`.
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

# Avoid repeating the migration warning when several objects load one profile.
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
    """Choose a name to record when no account file supplies one.

    Use the same defaults as the keystore: ``account``, then a nonempty
    ``ALEO_ACCOUNT``, then ``default``.
    """
    return account if account is not None else (os.environ.get("ALEO_ACCOUNT") or "default")


def _same_directory(first: Path, second: Path) -> bool:
    """Report whether two paths name one directory, following symbolic links."""
    try:
        return os.path.samefile(first, second)
    except OSError:
        # A directory that does not exist yet is compared by its resolved path.
        return os.path.normcase(os.path.realpath(first)) == os.path.normcase(os.path.realpath(second))


def _reject_account_in_profile_files(home: Path, account: Optional[str],
                                     accounts_dir: "Path | str | None") -> None:
    """Refuse an account file that is also one of the profile's own files.

    The keystore stores an account as ``<name>.json``. When the accounts
    directory is the profile home, the names ``profile`` and ``credentials``
    select ``profile.json`` and ``credentials.json``, and writing either
    file would replace the stored key. Reads the environment and resolves
    paths; writes nothing.

    Raises:
        ValueError: The selected account file is ``profile.json`` or
            ``credentials.json`` in ``home``.
    """
    name = _selected_account_name(account)
    if f"{name}.json" not in (_PROFILE, _CREDENTIALS):
        return
    # Resolve the directory as the keystore does: argument, variable, then default.
    directory = Path(accounts_dir or os.environ.get("ALEO_ACCOUNTS_DIR")
                     or Path.home() / ".aleo" / "accounts").expanduser()
    if _same_directory(directory, home):
        raise ValueError(
            f'The account "{name}" in {directory} would be stored as {home / (name + ".json")}, '
            "a file this profile writes its own data to. Choose another account name, or an "
            "accounts directory other than the profile home.")


def _report_legacy_key(path: Path, address: str) -> None:
    """Point callers to the migration command for a key still in profile.json.

    Report each file once per process without including the key.
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
    """Reuse an account and its trading history across sessions.

    Load the same profile on each run to keep its address, API credentials,
    and swap journal together. A profile belongs to one address and one
    network; give a separate account or network its own home directory.

    Use :meth:`load_or_create` to open the profile. If an account already
    exists, the profile uses it. Otherwise it generates a key. For a new
    key, choose password encryption or macOS Keychain storage::

        from aleo_shield_swap import Profile

        profile = Profile.load_or_create(storage="encrypted")
        address = profile.address

    The first run asks for an encryption password unless one was supplied.
    Later runs need that password to unlock the key. On macOS,
    ``storage="keychain"`` keeps the key in Keychain instead of the account
    file. The default, ``"plaintext"``, leaves the key readable to anyone
    with access to the file. Passing ``storage`` does not convert an
    existing account; use ``python -m aleo.keystore protect`` for that.

    A second home directory alone does not create a separate account:
    profiles selecting the same account name share its key. To trade from
    another address, use a separate home and ``account`` name, and clear
    any key variables that still select the first account.
    """

    def __init__(self, home: Path, data: dict[str, Any]) -> None:
        self.home = home
        self._data = data
        # Retain these options so inspecting a profile does not require unlocking it.
        self._account: Optional[str] = None
        self._accounts_dir: "Path | str | None" = None
        self._password: Optional[str] = None
        # Reuse the unlocked key for this object instead of prompting on every read.
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
        """Open a trading profile, creating one if this is its first run.

        To keep an existing address and its funds, import its key with
        ``python -m aleo.keystore import --storage encrypted`` first, then
        load the profile with ``create=False``. On macOS, ``--storage keychain``
        keeps the key in Keychain. Import keys in a terminal, never in a chat.
        ``create=False`` permits a new profile but requires an existing key.

        To start with a separate address, choose an unused account name and
        profile home. Use ``storage="keychain"`` on macOS or
        ``storage="encrypted"`` to protect a newly generated key. The default
        is plaintext, which leaves the private key readable in the account
        file. ``storage`` applies only when generating a key; it does not
        change an existing account. ``python -m aleo.keystore protect`` can
        change storage while keeping the address and funds.

        The profile records its address, account name, network, and endpoint
        in ``profile.json``. Creating it does not contact the network or move
        funds. An existing profile keeps its recorded settings and can be
        opened without a password; its key is unlocked on the first read of
        :attr:`private_key`.

        Keys supplied through ``ALEO_PRIVATE_KEY`` or ``ALEO_PRIVATE_KEY_FILE``
        take precedence over saved keys and are never copied to disk. Supply
        the key again on each run, or import it once to save it.
        ``SHIELD_SWAP_PRIVATE_KEY`` and ``SHIELD_SWAP_PRIVATE_KEY_FILE`` are
        also accepted. Any supplied key must match the profile's address
        and, if present, the selected account file.

        Args:
            home: Directory for the profile, credentials, and swap journal.
                Defaults to ``SHIELD_SWAP_HOME``, then ``~/.shield-swap``.
            network: ``"testnet"`` (default) or ``"mainnet"`` for a new profile.
                An existing profile keeps its network; use another home to
                trade on a different network.
            endpoint: API service root for a new profile. Defaults to
                ``https://edge.provable.com/api``; ignored for an existing profile.
            create: Generate and save a key if none is found for a new profile.
                Defaults to True. Use False to require an existing account.
            storage: Protection for a generated key. ``"encrypted"`` encrypts
                the key in the account file and requires a password to unlock
                it. ``"keychain"`` stores the key in macOS Keychain, outside
                the file. ``"plaintext"`` (default) restricts file permissions
                but does not encrypt the key or protect a copied file. The
                address remains readable in every mode. ``"none"`` is rejected
                because the generated key would be lost when the process exits.
            account: Name of the saved account. Defaults to ``ALEO_ACCOUNT``,
                then the profile's saved name, then ``default``. Use a distinct
                name as well as a distinct home for each separate account.
            accounts_dir: Directory containing account files. Defaults to
                ``ALEO_ACCOUNTS_DIR``, then ``~/.aleo/accounts``. This location
                is not saved in the profile; supply it on each run.
            password: Password to unlock an encrypted account or protect a new
                key. If omitted, uses ``ALEO_ACCOUNT_PASSWORD``, then a password
                saved in macOS Keychain, then a hidden terminal prompt.
            save_password: Save a new encrypted account's password in macOS
                Keychain so later runs can unlock it without a prompt.
                Defaults to False; applies only when generating a key.

        Returns:
            The profile for the recorded address. For an existing profile,
            opening it does not check whether its key is available; read
            :attr:`private_key` when it is needed for signing.

        Raises:
            aleo.keystore.KeystoreError: A new profile cannot open or save its
                account. Check ``code``: ``ACCOUNT_NOT_FOUND`` with
                ``create=False`` means the account must be imported or supplied;
                ``PASSWORD_REQUIRED`` or ``WRONG_PASSWORD`` requires a valid
                password; ``ADDRESS_MISMATCH`` means the supplied key belongs
                to another account. Other codes identify invalid key sources,
                names, or unavailable storage. An existing profile reports key
                errors when :attr:`private_key` is read.
            ValueError: The account file would be ``profile.json`` or
                ``credentials.json`` in ``home``: the account is named
                ``profile`` or ``credentials`` and the accounts directory is
                the profile home. Nothing is written. Choose another name or
                another accounts directory.
        """
        # Treat ~ as the user's home, including paths passed directly from Python.
        home = Path(home).expanduser() if home is not None else cls.default_home()
        _reject_account_in_profile_files(home, account, accounts_dir)
        path = home / _PROFILE
        if path.exists():
            path.chmod(0o600)             # heal a pre-existing loose mode
            profile = cls(home, json.loads(path.read_text()))
            # Allow address and credential reads before the account is unlocked.
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
                # Remember the account name even when the key is supplied per run.
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
                                       storage=cast(Any, storage),   # save() validates storage.
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
        """Unlock the account's private key for signing.

        The key gives control of the account's funds and can decrypt its
        records. Keep it out of logs, conversations, and service requests.

        For an existing profile, the first read loads the key and checks
        that it belongs to :attr:`address`. An encrypted account may ask
        for its password. Later reads reuse the unlocked key for this
        object's lifetime, so changes to key variables or files take effect
        only after loading another profile object.

        A key in ``ALEO_PRIVATE_KEY`` or ``ALEO_PRIVATE_KEY_FILE`` is used
        before the saved account. Account selection follows the ``account``
        argument, ``ALEO_ACCOUNT``, the profile's saved name, then ``default``.
        An older profile containing a private key still works when no other
        key source or account file exists. Using that key logs a migration
        command once per profile file per process; it does not move or delete
        the key.

        Raises:
            aleo.keystore.KeystoreError: Check ``code`` to decide what to supply.
                ``ACCOUNT_NOT_FOUND`` means the key is missing;
                ``KEY_NOT_STORED`` means the account file contains only an
                address. Import or supply the original key in either case.
                ``PASSWORD_REQUIRED`` or ``WRONG_PASSWORD`` means an encrypted
                account needs its password. ``ADDRESS_MISMATCH`` means the key
                belongs to another address; the message names both addresses.
                ``EMPTY_KEY_SOURCE`` or ``KEY_CONFLICT`` identifies empty or
                conflicting key variables.
        """
        if self._private_key is None:
            with self._key_lock:
                if self._private_key is None:
                    self._private_key = self._resolve_private_key()
        return self._private_key

    def _resolve_private_key(self) -> str:
        """Load a key that matches the saved address before allowing it to sign."""
        from aleo import keystore

        # Passing the saved name would override ALEO_ACCOUNT in keystore.load().
        # Use it only when neither the caller nor the environment selects a name.
        name = self._account
        if name is None and not os.environ.get("ALEO_ACCOUNT"):
            name = self._data.get("account")
        loaded = keystore.load(
            name=name, dir=self._accounts_dir,
            # Keep older profiles usable until their keys are migrated.
            legacy_key=self._data.get("private_key") or None,
            expected_address=self.address,
            password=self._password)
        if loaded.source == "legacy":
            _report_legacy_key(self.home / _PROFILE, self.address)
        self._password = None             # Release the password reference after unlocking.
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
