"""Store an Aleo account on disk and find it again on later runs.

An account file records an address and, depending on the storage chosen, the
private key itself, a pointer to it in the macOS Keychain, or the key
encrypted under a password. :func:`load` then resolves the signing key the
same way in every tool: a key passed by the caller, the ``ALEO_PRIVATE_KEY``
or ``ALEO_PRIVATE_KEY_FILE`` variables, and finally the account file.

Creating a key and storing it are separate steps. Make or import an account
with ``aleo.account.create()``, ``from_private_key()`` or ``from_mnemonic()``
on a client, then pass it to :func:`save`. Nothing here contacts a network,
and nothing is written unless :func:`save`, :func:`protect` or
:func:`migrate` is called.

The file is network-neutral: one account serves testnet and mainnet, and the
functions take and return keys and addresses as strings. Named accounts keep
separate keys apart for callers who want that.

Choose the storage by what must be protected:

- ``"plaintext"`` (the default) keeps the key in a file only its owner can
  read. Anything running as the same user can read it, and so can a backup or
  a synced folder.
- ``"keychain"`` keeps the key in the macOS login keychain, so a copied or
  backed-up account file holds nothing secret. macOS only.
- ``"encrypted"`` encrypts the key under a password, on every system. Each
  load needs the password from ``ALEO_ACCOUNT_PASSWORD``, from the macOS
  Keychain when it was saved there, or from a hidden terminal prompt.
- ``"none"`` records the address only. The key is supplied through the
  environment on every run, and the file confirms it is the expected account.

Keychain and encrypted storage protect a key at rest. They do not stop a
process running as the same user from reading it.

Example::

    from aleo import Aleo, HTTPProvider, keystore

    aleo = Aleo(HTTPProvider("https://api.provable.com/v2", network="testnet"))

    # First run: create an account and store it.
    stored = keystore.save(aleo.account.create(), name="testnet")

    # Later runs: resolve the key and hand it to the client.
    loaded = keystore.load(name="testnet")
    account = aleo.account.from_private_key(loaded.private_key)

The same operations are available from a terminal through
``python -m aleo.keystore``.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, NamedTuple, Optional, Sequence, Tuple, Union, cast

from ._encrypted import EncryptedKey, encrypt_key, parse_key
from ._errors import KeystoreError
from ._file import (
    STORAGE_TYPES,
    AccountFile,
    Location,
    loads_strict,
    read_account_file,
    resolve_location,
    write_account_file,
    write_file_atomic,
)
from ._keychain import (
    KEY_SERVICE,
    PASSWORD_SERVICE,
    delete_keychain,
    keychain_available,
    read_keychain,
    require_keychain,
    write_keychain,
)
from ._password import (
    PasswordSubject,
    choose_new_password,
    password_entry_account,
    password_to_keychain_value,
    saved_password_entry,
    unlock,
)
from ._sources import (
    KEY_FILE_VARIABLES,
    KEY_VARIABLES,
    PathLike,
    Resolved,
    error_code,
    key_from_environment,
)

Storage = Literal["none", "plaintext", "keychain", "encrypted"]
"""How an account file holds the private key; see the module description."""

# Top-level fields a legacy state file may keep its key in, in order of
# precedence.
_LEGACY_KEY_FIELDS = ("privateKey", "private_key")

_LONE_SURROGATE = re.compile("[\ud800-\udfff]")


@dataclass(frozen=True)
class StoredAccount:
    """Describe an account file without its key.

    Safe to print or log: it never holds the private key or a password.

    Attributes:
        name: Account name. Pass it as ``name`` to select the same account
            again.
        path: Absolute path of the account file.
        address: Aleo address recorded in the file.
        storage: How the file holds the private key: ``"none"``,
            ``"plaintext"``, ``"keychain"`` or ``"encrypted"``.
    """

    name: str
    path: Path
    address: str
    storage: Storage


@dataclass(frozen=True)
class LoadedAccount:
    """The account resolved by :func:`load`.

    Pass ``private_key`` to ``aleo.account.from_private_key()`` on a client to
    get an account object for signing. The key is left out of ``repr()`` so a
    logged result does not leak it.

    Attributes:
        address: Aleo address the key derives.
        private_key: The private key (``APrivateKey1...``). Keep it out of
            logs and files.
        source: Where the key came from: ``"argument"`` (the ``private_key``
            argument), ``"env"`` (``ALEO_PRIVATE_KEY`` or its alias),
            ``"env-file"`` (the file named by ``ALEO_PRIVATE_KEY_FILE`` or its
            alias), ``"account-file"`` (the selected account file) or
            ``"legacy"`` (the ``legacy_key`` argument). A caller that passed
            ``legacy_key`` reports the pending migration when this is
            ``"legacy"``.
        stored: The selected account file when one exists, even when the key
            itself came from the environment; otherwise ``None``.
    """

    address: str
    private_key: str = field(repr=False)
    source: str
    stored: Optional[StoredAccount] = None


class _Key(NamedTuple):
    """A parsed private key and the address it derives."""

    private_key: str
    address: str


class _Legacy(NamedTuple):
    """A legacy state file, the fields that hold its key, and the parsed key."""

    text: str
    content: Dict[str, Any]
    fields: Tuple[str, ...]
    account: "_Key"


def _describe(location: Location, file: AccountFile) -> StoredAccount:
    """Build the public description of an account file."""
    return StoredAccount(
        name=location.name,
        path=Path(location.path),
        address=file["address"],
        storage=cast(Storage, file["key"]["type"]),
    )


def _assert_storage(storage: object, allowed: Sequence[str]) -> None:
    """Check a storage value supplied at run time.

    Raises:
        KeystoreError: ``UNSUPPORTED_STORAGE`` for a value outside *allowed*.
    """
    if not isinstance(storage, str) or storage not in allowed:
        raise KeystoreError(
            "UNSUPPORTED_STORAGE",
            f'Unsupported storage type "{storage}"; expected one of {", ".join(allowed)}.',
        )


def _mismatch(origin: str, actual: str, requirement: str, expected: str) -> KeystoreError:
    """Build the error for a key that derives a different address than required."""
    return KeystoreError("ADDRESS_MISMATCH", f"{origin} derives {actual}, but {requirement} {expected}.")


def _account_key(account: object) -> Tuple[object, Optional[str]]:
    """Take the key, and the address when one is stated, from a ``save`` argument.

    Accepts a key string, a private-key object (its text form is the key) or
    an object with a ``private_key`` attribute, such as an account of either
    network or a :class:`LoadedAccount`.
    """
    if isinstance(account, str):
        return account, None
    holder: Any = getattr(account, "private_key", None)
    if holder is None:
        # A private-key object of either network.
        return _text_of(account), None
    stated: Any = getattr(account, "address", None)
    key: object = holder if isinstance(holder, str) else _text_of(holder)
    return key, None if stated is None else _text_of(stated)


def _text_of(value: object) -> Optional[str]:
    """Return the text form of a key or address object, or ``None`` when it has none."""
    try:
        return str(value)
    except Exception:
        return None


def _read_stored_key(
    file: AccountFile,
    location: Location,
    password: Union[str, bytes, None],
    prompt: bool,
) -> Tuple[str, Optional[bytes]]:
    """Read the private key an account file holds or points at.

    Does not consult the key variables. May run the ``security`` tool, stretch
    a password and prompt on the terminal, depending on the storage type.

    Returns:
        The private key, and for an encrypted file the normalised password
        that unlocked it.

    Raises:
        KeystoreError: ``KEY_NOT_STORED``, ``INVALID_KEY``,
            ``ADDRESS_MISMATCH``, ``KEYCHAIN_UNAVAILABLE``,
            ``KEYCHAIN_ITEM_NOT_FOUND``, ``KEYCHAIN_FAILED``,
            ``PASSWORD_REQUIRED`` or ``WRONG_PASSWORD``.
    """
    key: Dict[str, Any] = file["key"]
    address: str = file["address"]
    kind: str = key["type"]
    if kind == "none":
        raise KeystoreError(
            "KEY_NOT_STORED",
            f"Account file {location.path} records the address {address} without a key; "
            "the key of this account is supplied through ALEO_PRIVATE_KEY or ALEO_PRIVATE_KEY_FILE.",
        )
    if kind == "plaintext":
        origin = f"The key in account file {location.path}"
        parsed = _Key(*parse_key(key["privateKey"], origin))
        if parsed.address != address:
            raise _mismatch(origin, parsed.address, "the file records", address)
        return parsed.private_key, None
    if kind == "keychain":
        require_keychain(
            f"Account file {location.path} keeps its key in the Keychain of the Mac it was created on. "
            "Export the key on that Mac and import it here."
        )
        value = read_keychain(KEY_SERVICE, address)
        if value is None:
            raise KeystoreError(
                "KEYCHAIN_ITEM_NOT_FOUND",
                f"Account file {location.path} points at the Keychain entry {KEY_SERVICE} for {address}, "
                "which does not exist.",
            )
        origin = f"The Keychain entry {KEY_SERVICE} for {address}"
        parsed = _Key(*parse_key(value, origin))
        if parsed.address != address:
            raise _mismatch(origin, parsed.address, f"account file {location.path} records", address)
        return parsed.private_key, None
    private_key, used = unlock(
        PasswordSubject(location.name, address), cast(EncryptedKey, key), password, prompt
    )
    return private_key, used


class _KeychainEntry(NamedTuple):
    """One Keychain entry: its service and its account within the service."""

    service: str
    account: str


def _remove_entries(entries: Sequence[_KeychainEntry]) -> None:
    """Remove Keychain entries, ignoring failures; used when undoing a failed step."""
    for entry in entries:
        try:
            delete_keychain(entry.service, entry.account)
        except KeystoreError:
            pass


def _neighbour_uses(location: Location, matches: Callable[[Dict[str, Any]], bool]) -> bool:
    """Report whether another account file in the same directory matches.

    Keychain entries can be shared: every account file that holds one key in
    the Keychain points at the same entry, and a copied encrypted file shares
    its saved password. Before an entry is removed, the neighbouring files are
    checked so that one still in use stays. Files that cannot be read or
    parsed are skipped. Account files in other directories are not seen.

    Args:
        location: The account file being changed; it is not counted itself.
        matches: Decides whether a parsed neighbouring file uses the entry.
    """
    try:
        names = os.listdir(location.dir)
    except OSError:
        return False
    for name in names:
        if not name.endswith(".json") or name == f"{location.name}.json":
            continue
        try:
            with open(os.path.join(location.dir, name), "rb") as handle:
                parsed = json.loads(handle.read().decode("utf-8"))
        except (OSError, ValueError, RecursionError):
            # Not an account file, or not readable: it holds no claim on the entry.
            continue
        if isinstance(parsed, dict) and matches(cast(Dict[str, Any], parsed)):
            return True
    return False


def _key_of(file: Dict[str, Any]) -> Dict[str, Any]:
    """Read the ``key`` object of a loosely parsed account file."""
    key = file.get("key")
    return cast(Dict[str, Any], key) if isinstance(key, dict) else {}


def _write_storage(
    location: Location,
    account: _Key,
    storage: str,
    password: Optional[bytes],
    save_password: bool,
    exclusive: bool = False,
    created: Optional[List[_KeychainEntry]] = None,
) -> AccountFile:
    """Write an account under the given storage.

    Secrets held outside the file (the Keychain key entry, a saved password)
    are written first, so a failure there leaves no file pointing at a missing
    entry.

    Args:
        location: Resolved account location.
        account: Parsed key and the address it derives.
        storage: Storage type to write.
        password: Normalised password bytes; required for ``"encrypted"``.
        save_password: Whether to save the password in the macOS Keychain.
            Applies to ``"encrypted"`` on macOS and is ignored otherwise.
        exclusive: Whether the account file must not exist yet. By default
            any file at the path is replaced.
        created: Receives each Keychain entry this call creates, so a
            caller can remove them when a later step fails. An entry that
            existed before the call is not listed: another account file may
            rely on it.

    Returns:
        The content written to the account file.

    Raises:
        FileExistsError: *exclusive* is set and the account file already
            exists.
    """
    private_key, address = account
    key: Dict[str, Any]
    if storage == "none":
        key = {"type": "none"}
    elif storage == "plaintext":
        key = {"type": "plaintext", "privateKey": private_key}
    elif storage == "keychain":
        if write_keychain(KEY_SERVICE, address, private_key, location.name) and created is not None:
            created.append(_KeychainEntry(KEY_SERVICE, address))
        key = {"type": "keychain", "service": KEY_SERVICE, "account": address}
    else:
        if not password:
            raise KeystoreError(
                "PASSWORD_REQUIRED", f"Encrypting the account {address} needs a password."
            )
        key = dict(encrypt_key(private_key, password))
        if save_password and keychain_available():
            # The entry is named after this file's salt, so it cannot collide
            # with the saved password of another file for the same address.
            entry = password_entry_account(address, key["kdfParams"]["salt"])
            if (
                write_keychain(PASSWORD_SERVICE, entry, password_to_keychain_value(password), location.name)
                and created is not None
            ):
                created.append(_KeychainEntry(PASSWORD_SERVICE, entry))
    file: AccountFile = {"version": 1, "address": address, "key": key}
    write_account_file(location, file, exclusive)
    return file


def _confirm_stored(
    location: Location,
    account: _Key,
    storage: str,
    password: Union[str, bytes, None],
    prompt: bool,
) -> None:
    """Read an account file back and confirm it returns the expected key.

    Raises:
        KeystoreError: ``INVALID_ACCOUNT_FILE`` when the file is absent or
            holds another account or storage, ``ADDRESS_MISMATCH`` when it
            returns another key, plus the failures of reading the stored key.
    """
    written = read_account_file(location.path)
    if (
        written is None
        or written.file["address"] != account.address
        or written.file["key"]["type"] != storage
    ):
        raise KeystoreError(
            "INVALID_ACCOUNT_FILE",
            f"Account file {location.path} does not hold {account.address} with {storage} storage.",
        )
    private_key, _ = _read_stored_key(written.file, location, password, prompt)
    if private_key != account.private_key:
        raise KeystoreError(
            "ADDRESS_MISMATCH",
            f"Account file {location.path} does not return the key of {account.address} when read back.",
        )


def _store_account(
    account: _Key,
    location: Location,
    storage: str,
    password: Optional[str],
    save_password: bool,
) -> Tuple[StoredAccount, Optional[bytes]]:
    """Store an account unless its file already exists.

    Shared by :func:`save` and :func:`migrate`.

    Returns:
        The stored account and, when a new encrypted file was written, the
        password it was written with.

    Raises:
        KeystoreError: ``ACCOUNT_EXISTS`` when the file exists with another
            address or another storage type, plus the failures of writing the
            storage.
    """
    # Checked before anything is written or asked for.
    if storage == "keychain":
        require_keychain('Keychain storage works on macOS only; use "plaintext" or "encrypted" here.')

    existing = read_account_file(location.path)
    if existing is not None:
        held: str = existing.file["address"]
        current: str = existing.file["key"]["type"]
        if held != account.address:
            raise KeystoreError(
                "ACCOUNT_EXISTS",
                f"Account file {location.path} already holds {held}; refusing to replace it with "
                f"{account.address}. Choose another account name.",
            )
        if current != storage:
            raise KeystoreError(
                "ACCOUNT_EXISTS",
                f"Account file {location.path} already holds {account.address} with {current} storage. "
                "Use protect to change how the key is stored.",
            )
        # Same account, same storage: nothing to write.
        return _describe(location, existing.file), None

    chosen: Optional[bytes] = None
    if storage == "encrypted":
        chosen = choose_new_password(
            PasswordSubject(location.name, account.address),
            explicit=[password],
            environment=True,
            prompt=True,
        )
    # The file is created exclusively: another process may have stored an
    # account under this name since the check above, and it must not be
    # replaced.
    # Only Keychain entries this call creates are undone on failure: an entry
    # that was already there may be the only stored key of another account
    # file.
    created: List[_KeychainEntry] = []
    try:
        file = _write_storage(location, account, storage, chosen, save_password, True, created)
    except FileExistsError:
        now = read_account_file(location.path)
        same = now is not None and now.file["address"] == account.address and now.file["key"]["type"] == storage
        # The same account in the same Keychain storage is what was asked for,
        # and the key entry just written is the one that file points at.
        if now is not None and same and storage != "encrypted":
            return _describe(location, now.file), None
        _remove_entries(created)
        if now is not None and same:
            return _describe(location, now.file), None
        raise KeystoreError(
            "ACCOUNT_EXISTS",
            f"Account file {location.path} was created by another process while {account.address} was "
            "being stored; it was not replaced. Choose another account name.",
        ) from None
    except BaseException:
        # No account file was created, so nothing points at the entries made here.
        _remove_entries(created)
        raise
    return _describe(location, file), chosen


def save(
    account: object,
    *,
    name: Optional[str] = None,
    dir: Optional[PathLike] = None,
    storage: Storage = "plaintext",
    password: Optional[str] = None,
    save_password: bool = False,
) -> StoredAccount:
    """Store an account in an account file so later runs can load it.

    Writes ``<dir>/<name>.json`` with owner-only access (directory mode 0700,
    file mode 0600) through a temporary file and a rename. Depending on
    ``storage`` it also writes the macOS Keychain or stretches a password, and
    it prompts for a password when ``"encrypted"`` is requested without one
    and a terminal is attached. Does not contact a network.

    An existing file is never replaced. Saving the same account with the same
    storage again returns without rewriting; use :func:`protect` to change how
    an account is stored.

    Args:
        account: The account to store: an ``Account`` or ``PrivateKey`` from
            ``aleo.mainnet`` or ``aleo.testnet``, a private-key string, or a
            :class:`LoadedAccount`. When the object states an address, the key
            must derive it.
        name: Account name: 1 to 64 characters of ``a-z``, ``0-9``, ``_`` or
            ``-``, starting with a letter or digit. Defaults to
            ``ALEO_ACCOUNT``, then ``default``.
        dir: Accounts directory; a leading ``~`` expands to the home
            directory. Defaults to ``ALEO_ACCOUNTS_DIR``, then
            ``~/.aleo/accounts``. Pass it to keep accounts inside a project or
            a test directory.
        storage: How to hold the key. Defaults to ``"plaintext"``. Use
            ``"keychain"`` (macOS) or ``"encrypted"`` to protect a copied or
            backed-up file, and ``"none"`` to record the address only; the
            caller then keeps the key it already holds.
        password: Password for ``"encrypted"``. Defaults to
            ``ALEO_ACCOUNT_PASSWORD``, then a hidden prompt that asks twice.
            Must not be empty.
        save_password: Also save the password in the macOS Keychain, so later
            loads need no password. Defaults to False. Applies to
            ``"encrypted"`` on macOS and is ignored otherwise.

    Returns:
        The stored account, without its key.

    Raises:
        KeystoreError: ``ACCOUNT_EXISTS`` when the file holds another address,
            or the same address with another storage; ``INVALID_KEY`` or
            ``ADDRESS_MISMATCH`` when the key is unusable or does not derive
            the stated address; ``INVALID_NAME``; ``UNSUPPORTED_STORAGE``;
            ``KEYCHAIN_UNAVAILABLE`` for ``"keychain"`` outside macOS, before
            anything is written; ``KEYCHAIN_FAILED``; ``PASSWORD_REQUIRED``
            when ``"encrypted"`` has no password source.
        OSError: The directory or file could not be written.
    """
    _assert_storage(storage, STORAGE_TYPES)
    location = resolve_location(name, dir)
    origin = "The private key of the account"
    key_text, stated = _account_key(account)
    parsed = _Key(*parse_key(key_text, origin))
    if stated is not None and stated != parsed.address:
        raise _mismatch(origin, parsed.address, "the account states", stated)
    stored, _ = _store_account(parsed, location, storage, password, bool(save_password))
    return stored


def load(
    *,
    name: Optional[str] = None,
    dir: Optional[PathLike] = None,
    private_key: Optional[str] = None,
    legacy_key: Optional[str] = None,
    expected_address: Optional[str] = None,
    password: Optional[str] = None,
    prompt: bool = True,
) -> LoadedAccount:
    """Resolve the account to sign with, following the lookup order.

    Sources, first match wins:

    1. The ``private_key`` argument. The key variables are then not read.
    2. ``ALEO_PRIVATE_KEY``, then its alias ``SHIELD_SWAP_PRIVATE_KEY``.
    3. The file named by ``ALEO_PRIVATE_KEY_FILE``, then its alias
       ``SHIELD_SWAP_PRIVATE_KEY_FILE``: a text file holding only the key.
    4. The selected account file.
    5. The ``legacy_key`` argument, only when no account file exists.

    A variable that is set but empty is an error, and so are two variables
    that hold different keys. When the selected account file exists, a key
    from sources 1 to 3 must derive the address the file records. Nothing here
    generates a key or writes one to disk, so a key supplied through the
    environment is used for this process only.

    Reads the environment and the account file, tightens a file readable by
    others to mode 0600, and for a ``"keychain"`` or ``"encrypted"`` file runs
    the macOS ``security`` tool or stretches a password (about 200 ms). The
    password of an encrypted file comes from the ``password`` argument, then
    ``ALEO_ACCOUNT_PASSWORD``, then a password saved in the macOS Keychain,
    then a hidden terminal prompt with three attempts. A wrong password from
    any source but the prompt fails at once. Does not contact a network.

    Args:
        name: Account name. Defaults to ``ALEO_ACCOUNT``, then ``default``.
        dir: Accounts directory. Defaults to ``ALEO_ACCOUNTS_DIR``, then
            ``~/.aleo/accounts``.
        private_key: Key supplied by the caller, taking precedence over every
            other source.
        legacy_key: Key from the calling tool's own earlier state. Used only
            when no other source and no account file exists; the result then
            reports ``source == "legacy"``. When an account file exists, this
            key must derive the file's address.
        expected_address: Address the resolved key must derive, such as the
            address recorded in the calling tool's state.
        password: Password for an encrypted account file.
        prompt: Whether a password prompt may be shown when a terminal is
            attached. Defaults to True. Pass False for unattended jobs.

    Returns:
        The address, the private key, the source of the key and the selected
        account file when one exists.

    Raises:
        KeystoreError: ``ACCOUNT_NOT_FOUND`` when no source produced a key;
            ``KEY_NOT_STORED`` when the file records an address only and no
            key came from sources 1 to 3; ``EMPTY_KEY_SOURCE``;
            ``KEY_CONFLICT``; ``INVALID_KEY``; ``ADDRESS_MISMATCH``;
            ``INVALID_NAME``; ``INVALID_ACCOUNT_FILE``,
            ``UNSUPPORTED_VERSION`` or ``UNSUPPORTED_STORAGE`` for a file this
            reader cannot use; ``KEYCHAIN_UNAVAILABLE``,
            ``KEYCHAIN_ITEM_NOT_FOUND`` or ``KEYCHAIN_FAILED``;
            ``PASSWORD_REQUIRED`` or ``WRONG_PASSWORD``.
        OSError: The account file exists but could not be read.
    """
    location = resolve_location(name, dir)

    # Sources 1 to 3: a key handed over by the caller or the environment.
    resolved: Optional[Resolved]
    if private_key is not None:
        origin = "The private_key argument"
        resolved = Resolved(*parse_key(private_key, origin), "argument", origin)
    else:
        resolved = key_from_environment()

    # The account file is read even when a key is already in hand: it records
    # which account this name stands for, and an unusable file stops the lookup.
    existing = read_account_file(location.path)
    legacy_origin = "The legacy_key argument"
    if existing is not None and legacy_key is not None:
        legacy = _Key(*parse_key(legacy_key, legacy_origin))
        if legacy.address != existing.file["address"]:
            raise _mismatch(
                legacy_origin,
                legacy.address,
                f"account file {location.path} records",
                existing.file["address"],
            )

    if resolved is not None:
        if existing is not None and resolved.address != existing.file["address"]:
            raise _mismatch(
                resolved.origin,
                resolved.address,
                f"account file {location.path} records",
                existing.file["address"],
            )
    elif existing is not None:
        # Source 4: the account file itself.
        stored_key, _ = _read_stored_key(existing.file, location, password, bool(prompt))
        resolved = Resolved(
            stored_key, existing.file["address"], "account-file", f"Account file {location.path}"
        )
    elif legacy_key is not None:
        # Source 5: the calling tool's earlier state.
        resolved = Resolved(*parse_key(legacy_key, legacy_origin), "legacy", legacy_origin)
    else:
        raise KeystoreError(
            "ACCOUNT_NOT_FOUND",
            f"No Aleo account found: {KEY_VARIABLES[0]} and {KEY_FILE_VARIABLES[0]} are not set and "
            f"there is no account file at {location.path}.",
        )

    if expected_address is not None and resolved.address != expected_address:
        raise _mismatch(resolved.origin, resolved.address, "the expected address is", expected_address)

    return LoadedAccount(
        address=resolved.address,
        private_key=resolved.private_key,
        source=resolved.source,
        stored=_describe(location, existing.file) if existing is not None else None,
    )


def info(*, name: Optional[str] = None, dir: Optional[PathLike] = None) -> Optional[StoredAccount]:
    """Describe the selected account file without reading its key.

    Reads only the account file: no key variable, Keychain entry or password
    is touched, so the result is safe to show. Tightens a file readable by
    others to mode 0600.

    Args:
        name: Account name. Defaults to ``ALEO_ACCOUNT``, then ``default``.
        dir: Accounts directory. Defaults to ``ALEO_ACCOUNTS_DIR``, then
            ``~/.aleo/accounts``.

    Returns:
        The account's name, path, address and storage type, or ``None`` when
        no account file exists.

    Raises:
        KeystoreError: ``INVALID_NAME``, or ``INVALID_ACCOUNT_FILE``,
            ``UNSUPPORTED_VERSION`` or ``UNSUPPORTED_STORAGE`` for a file this
            reader cannot use.
        OSError: The account file exists but could not be read.
    """
    location = resolve_location(name, dir)
    existing = read_account_file(location.path)
    return _describe(location, existing.file) if existing is not None else None


def protect(
    *,
    storage: Literal["plaintext", "keychain", "encrypted"],
    name: Optional[str] = None,
    dir: Optional[PathLike] = None,
    password: Optional[str] = None,
    new_password: Optional[str] = None,
    save_password: bool = False,
    prompt: bool = True,
) -> StoredAccount:
    """Change how an existing account file holds its key.

    Reads the current key from the account file, writes the new storage, reads
    it back and checks the address, and only then removes the old secret: the
    Keychain key entry, or a saved password that no longer applies. When the
    check fails the original file is put back. The key variables are not read.

    Requesting ``"encrypted"`` for an account that is already encrypted sets a
    new password and a new salt. Requesting the storage a plaintext or
    Keychain account already has returns without rewriting. An account cannot
    be moved to ``"none"``, which would delete the only stored copy of the
    key.

    Moving to ``"keychain"`` or ``"encrypted"`` protects this account file
    only. A copy of the key in another tool's own state file stays readable
    until it is removed; see :func:`migrate`.

    Args:
        storage: Storage to move to: ``"plaintext"``, ``"keychain"`` (macOS)
            or ``"encrypted"``.
        name: Account name. Defaults to ``ALEO_ACCOUNT``, then ``default``.
        dir: Accounts directory. Defaults to ``ALEO_ACCOUNTS_DIR``, then
            ``~/.aleo/accounts``.
        password: Current password when the account is encrypted; defaults to
            ``ALEO_ACCOUNT_PASSWORD``, a password saved in the Keychain, then
            a prompt. When the account is not yet encrypted and
            ``new_password`` is not given, this is the password to set.
        new_password: Password to encrypt with. When the account is not yet
            encrypted it defaults to ``password``, then
            ``ALEO_ACCOUNT_PASSWORD``, then a prompt that asks twice. When
            replacing an existing password it defaults to the prompt only,
            because the variable then holds the current password.
        save_password: Save the new password in the macOS Keychain. Defaults
            to False, which also removes a previously saved password. Applies
            to ``"encrypted"`` on macOS and is ignored otherwise.
        prompt: Whether password prompts may be shown when a terminal is
            attached. Defaults to True.

    Returns:
        The account as stored after the change.

    Raises:
        KeystoreError: ``ACCOUNT_NOT_FOUND`` when no account file exists;
            ``KEY_NOT_STORED`` when the file records an address only;
            ``UNSUPPORTED_STORAGE`` for ``"none"`` or an unknown storage;
            ``KEYCHAIN_UNAVAILABLE`` outside macOS, before anything is
            written; ``KEYCHAIN_ITEM_NOT_FOUND`` or ``KEYCHAIN_FAILED``;
            ``PASSWORD_REQUIRED`` or ``WRONG_PASSWORD``; ``ADDRESS_MISMATCH``
            when a stored key does not derive the file's address.
        OSError: The account file could not be read or written.
    """
    _assert_storage(storage, ("plaintext", "keychain", "encrypted"))
    location = resolve_location(name, dir)
    if storage == "keychain":
        require_keychain('Keychain storage works on macOS only; use "plaintext" or "encrypted" here.')

    existing = read_account_file(location.path)
    if existing is None:
        raise KeystoreError("ACCOUNT_NOT_FOUND", f"There is no account file at {location.path}.")
    current: str = existing.file["key"]["type"]
    address: str = existing.file["address"]
    subject = PasswordSubject(location.name, address)

    # A plaintext or Keychain account already stored as requested needs nothing.
    if current == storage and storage != "encrypted":
        return _describe(location, existing.file)

    # Reading the current key also proves the stored copy is intact before
    # anything is replaced. An address-only file fails here with KEY_NOT_STORED.
    private_key, _ = _read_stored_key(existing.file, location, password, bool(prompt))
    account = _Key(private_key, address)

    chosen: Optional[bytes] = None
    if storage == "encrypted":
        if current == "encrypted":
            # The variable and ``password`` hold the current password here, so
            # the new one comes from its own argument or the person at the
            # terminal.
            chosen = choose_new_password(
                subject, explicit=[new_password], environment=False, prompt=bool(prompt)
            )
        else:
            chosen = choose_new_password(
                subject, explicit=[new_password, password], environment=True, prompt=bool(prompt)
            )
    saving = storage == "encrypted" and bool(save_password) and keychain_available()

    # Every step below is undone together if any of them fails: the original
    # file goes back, and the Keychain entries this call created are removed.
    # The old secret is the last thing touched, and it is touched only once
    # everything else has succeeded, so the original file works again as soon
    # as it is back in place.
    created: List[_KeychainEntry] = []
    try:
        # The new file's saved password has its own entry, named after the new
        # salt, so the entry of the file being replaced is not disturbed.
        _write_storage(location, account, storage, chosen, saving, False, created)
        # Read the new storage back the way a later load would, then compare.
        _confirm_stored(location, account, storage, chosen, False)

        # Only now is the old secret redundant. An entry a neighbouring
        # account file still uses is left in place.
        if current == "keychain" and storage != "keychain":
            if not _neighbour_uses(
                location, lambda file: file.get("address") == address and _key_of(file).get("type") == "keychain"
            ):
                delete_keychain(KEY_SERVICE, address)
        previous = saved_password_entry(existing.file)
        if previous is not None and keychain_available():
            salt = existing.file["key"]["kdfParams"]["salt"]

            def same_ciphertext(file: Dict[str, Any]) -> bool:
                key = _key_of(file)
                params = key.get("kdfParams")
                return (
                    file.get("address") == address
                    and key.get("type") == "encrypted"
                    and isinstance(params, dict)
                    and cast(Dict[str, Any], params).get("salt") == salt
                )

            if not _neighbour_uses(location, same_ciphertext):
                delete_keychain(PASSWORD_SERVICE, previous)
    except BaseException:
        write_file_atomic(location.path, existing.data)
        _remove_entries(created)
        raise

    return StoredAccount(name=location.name, path=Path(location.path), address=address, storage=storage)


def export(
    *,
    name: Optional[str] = None,
    dir: Optional[PathLike] = None,
    password: Optional[str] = None,
    prompt: bool = True,
) -> str:
    """Return the private key held by the selected account file.

    Use it to back a key up or to move an account to another machine. Reads
    only the account file and what it points at (the Keychain entry, or the
    password sources of an encrypted file); the key variables are never
    consulted. The result is the secret itself: a caller that prints it is
    responsible for doing so only for a person at a terminal.

    Args:
        name: Account name. Defaults to ``ALEO_ACCOUNT``, then ``default``.
        dir: Accounts directory. Defaults to ``ALEO_ACCOUNTS_DIR``, then
            ``~/.aleo/accounts``.
        password: Password for an encrypted account file. Defaults to
            ``ALEO_ACCOUNT_PASSWORD``, a password saved in the Keychain, then
            a prompt.
        prompt: Whether a password prompt may be shown when a terminal is
            attached. Defaults to True.

    Returns:
        The private key (``APrivateKey1...``).

    Raises:
        KeystoreError: ``ACCOUNT_NOT_FOUND`` when no account file exists;
            ``KEY_NOT_STORED`` when the file records an address only;
            ``INVALID_KEY`` or ``ADDRESS_MISMATCH`` when the stored key is
            unusable; ``KEYCHAIN_UNAVAILABLE``, ``KEYCHAIN_ITEM_NOT_FOUND`` or
            ``KEYCHAIN_FAILED``; ``PASSWORD_REQUIRED`` or ``WRONG_PASSWORD``.
        OSError: The account file exists but could not be read.
    """
    location = resolve_location(name, dir)
    existing = read_account_file(location.path)
    if existing is None:
        raise KeystoreError("ACCOUNT_NOT_FOUND", f"There is no account file at {location.path}.")
    private_key, _ = _read_stored_key(existing.file, location, password, bool(prompt))
    return private_key


def _read_legacy_key(path: str) -> _Legacy:
    """Read the private key from a legacy state file.

    Args:
        path: Path of a JSON file with a top-level ``privateKey`` or
            ``private_key`` string.

    Raises:
        KeystoreError: ``LEGACY_KEY_NOT_FOUND`` when the file cannot be read,
            is not a JSON object or has neither field; ``INVALID_KEY`` when a
            field holds text that is not a key; ``KEY_CONFLICT`` when the two
            fields hold different keys, since it is then unclear which account
            the file belongs to.
    """

    def not_found(problem: str) -> KeystoreError:
        return KeystoreError("LEGACY_KEY_NOT_FOUND", f"No key found in {path}: {problem}.")

    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError as error:
        raise not_found(f"the file could not be read ({error_code(error)})") from error

    # The parser's message can quote the file, which holds a key, so neither
    # failure below carries the underlying error.
    text: Optional[str]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = None
    if text is None:
        raise not_found("the file is not UTF-8 text")
    content: Any = None
    is_json = True
    try:
        content = loads_strict(text)
    except (ValueError, RecursionError):
        is_json = False
    if not is_json:
        raise not_found("the file is not JSON")
    if not isinstance(content, dict):
        raise not_found("the file does not hold a JSON object")
    record = cast(Dict[str, Any], content)
    fields = tuple(
        name for name in _LEGACY_KEY_FIELDS if isinstance(record.get(name), str) and record.get(name) != ""
    )
    if not fields:
        raise not_found('it has no "privateKey" or "private_key" field')
    first = fields[0]
    account = _Key(*parse_key(record[first], f'The "{first}" field of {path}'))
    if len(fields) > 1:
        second = fields[1]
        other = _Key(*parse_key(record[second], f'The "{second}" field of {path}'))
        if other.address != account.address:
            raise KeystoreError(
                "KEY_CONFLICT",
                f'{path} holds two different keys: "{first}" belongs to {account.address} and "{second}" to '
                f"{other.address}. Remove the one that is not wanted, then migrate again.",
            )
    return _Legacy(text, record, fields, account)


def _without_key_fields(legacy: _Legacy) -> str:
    """Render a legacy state file without its key fields.

    Keeps the other fields in their order, the indentation of the original and
    whether it ended with a newline.
    """
    rest = {name: value for name, value in legacy.content.items() if name not in legacy.fields}
    # The indentation is whatever precedes the first field on its own line; a
    # file written on one line stays on one line.
    layout = re.match(r"\s*\{\r?\n([ \t]+)\S", legacy.text)
    if layout is None:
        body = json.dumps(rest, ensure_ascii=False, separators=(",", ":"))
    else:
        # At most ten characters of indentation, the limit of JSON.stringify,
        # so every SDK rewrites a file the same way.
        body = json.dumps(rest, ensure_ascii=False, indent=layout.group(1)[:10])
    # A lone surrogate cannot be written as UTF-8; it keeps its escaped form.
    body = _LONE_SURROGATE.sub(lambda found: f"\\u{ord(found.group()):04x}", body)
    return f"{body}\n" if legacy.text.endswith("\n") else body


def migrate(
    source: PathLike,
    *,
    name: Optional[str] = None,
    dir: Optional[PathLike] = None,
    storage: Storage = "plaintext",
    password: Optional[str] = None,
    save_password: bool = False,
    remove_legacy_key: bool = False,
) -> StoredAccount:
    """Copy the private key from a legacy state file into an account file.

    Reads a JSON file that keeps a key in a top-level ``privateKey`` or
    ``private_key`` field, such as a Shield Swap CLI ``state.json`` or a
    Shield Swap Python ``profile.json``, and stores that key exactly as
    :func:`save` would. The legacy file is left untouched unless
    ``remove_legacy_key`` is set, so by default a plaintext copy of the key
    remains there.

    With ``remove_legacy_key``, the new account is read back and its key
    confirmed first; the legacy file is then rewritten without the key field,
    keeping its other fields, their order and its indentation, at mode 0600.

    Args:
        source: Path of the legacy JSON file.
        name: Account name. Defaults to ``ALEO_ACCOUNT``, then ``default``.
        dir: Accounts directory. Defaults to ``ALEO_ACCOUNTS_DIR``, then
            ``~/.aleo/accounts``.
        storage: How to hold the key. Defaults to ``"plaintext"``.
        password: Password for ``"encrypted"``. Defaults to
            ``ALEO_ACCOUNT_PASSWORD``, then a hidden prompt that asks twice.
        save_password: Also save the password in the macOS Keychain. Defaults
            to False. Applies to ``"encrypted"`` on macOS.
        remove_legacy_key: Strip the key from the legacy file once the new
            account file is confirmed. Defaults to False. Not allowed with
            ``storage="none"``, which would leave no stored copy.

    Returns:
        The stored account, without its key. When ``remove_legacy_key`` was
        set and the call returns, the legacy file no longer holds the key.

    Raises:
        KeystoreError: ``LEGACY_KEY_NOT_FOUND`` when the file holds no key;
            ``INVALID_KEY``; ``KEY_NOT_STORED`` when ``remove_legacy_key`` is
            combined with ``"none"``; plus every failure of :func:`save`,
            including ``ACCOUNT_EXISTS``.
        OSError: The account file or the legacy file could not be written.
    """
    _assert_storage(storage, STORAGE_TYPES)
    removing = bool(remove_legacy_key)
    if storage == "none" and removing:
        raise KeystoreError(
            "KEY_NOT_STORED",
            "An account stored without its key cannot replace the legacy key; "
            "choose another storage or keep the legacy key.",
        )
    location = resolve_location(name, dir)

    origin_path = os.fspath(source)
    legacy = _read_legacy_key(origin_path)
    account = legacy.account
    stored, chosen = _store_account(account, location, storage, password, bool(save_password))

    if not removing:
        return stored

    # The legacy copy goes only after the new one has been read back in full.
    _confirm_stored(location, account, storage, chosen if chosen is not None else password, True)
    write_file_atomic(origin_path, _without_key_fields(legacy))
    return stored


__all__ = [
    "KeystoreError",
    "LoadedAccount",
    "Storage",
    "StoredAccount",
    "export",
    "info",
    "load",
    "migrate",
    "protect",
    "save",
]
