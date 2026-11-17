"""Where the password of an encrypted account comes from.

Reading an account tries the supplied password, the ``ALEO_ACCOUNT_PASSWORD``
variable, a password saved in the macOS Keychain, and then a hidden prompt.
Setting a password never reads the Keychain and asks twice at the prompt.
"""
from __future__ import annotations

import base64
import os
import re
from typing import Any, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

from ._encrypted import EncryptedKey, decrypt_key, encode_password, normalize_password
from ._errors import KeystoreError
from ._keychain import PASSWORD_SERVICE, delete_keychain, keychain_available, read_keychain
from ._runtime import runtime

# Variable that supplies the password of an encrypted account.
PASSWORD_VARIABLE = "ALEO_ACCOUNT_PASSWORD"

# Number of tries a person gets at a prompt before the call fails.
_PROMPT_ATTEMPTS = 3

_SAVED_PASSWORD = re.compile(r"(?:[0-9a-f]{2})+")


class PasswordSubject(NamedTuple):
    """Identifies the account a password belongs to.

    Attributes:
        name: Account name, shown in prompts.
        address: Aleo address, shown in prompts and used as the Keychain
            account of a saved password.
    """

    name: str
    address: str


def _password_from_environment() -> Optional[str]:
    """Read the password variable, treating an empty value as unset."""
    return os.environ.get(PASSWORD_VARIABLE) or None


def password_entry_account(address: str, salt: str) -> str:
    """Name the Keychain entry that holds the saved password of one encrypted file.

    The entry belongs to a file, not to an address: two encrypted files can
    hold the same account under different passwords, and each needs its own
    saved password. The salt is unique to each encryption, so it identifies
    the file, and it changes whenever the file is encrypted again.

    Args:
        address: Aleo address of the account.
        salt: The file's scrypt salt, as standard base64.

    Returns:
        The entry's account: the address, ``-``, and the salt in hex.
    """
    return f"{address}-{base64.b64decode(salt).hex()}"


def saved_password_entry(file: Mapping[str, Any]) -> Optional[str]:
    """Name the saved-password entry of a parsed account file.

    Returns:
        The entry's account in the Keychain, or None when the file is not
        encrypted, since only an encrypted file has a saved password.
    """
    key = file["key"]
    if key["type"] != "encrypted":
        return None
    return password_entry_account(file["address"], key["kdfParams"]["salt"])


def forget_saved_password(file: Mapping[str, Any]) -> None:
    """Remove the saved password of a parsed account file from the Keychain.

    Does nothing for a file that is not encrypted. An entry that is already
    absent counts as removed.
    """
    entry = saved_password_entry(file)
    if entry is not None:
        delete_keychain(PASSWORD_SERVICE, entry)


def password_to_keychain_value(password: bytes) -> str:
    """Convert normalised password bytes to the form stored in the Keychain.

    Returns:
        The bytes as lower-case hex, which needs no quoting.
    """
    return password.hex()


def unlock(
    subject: PasswordSubject,
    key: EncryptedKey,
    password: Union[str, bytes, None] = None,
    prompt: bool = True,
) -> Tuple[str, bytes]:
    """Find the password of an encrypted account and decrypt its key.

    Sources, in order: the explicit password, ``ALEO_ACCOUNT_PASSWORD``, a
    password saved in the macOS Keychain, then a hidden prompt when a terminal
    is attached. The first source that supplies a password decides: a wrong
    one is an error and the later sources are not tried. Only the prompt
    retries, up to three times. The Keychain lookup runs the ``security`` tool
    on macOS.

    Args:
        subject: Account the password belongs to.
        key: The account file's encrypted ``key`` object.
        password: Explicit password, as text or as already normalised bytes.
        prompt: Whether a prompt may be shown when a terminal is attached.

    Returns:
        The private key and the normalised password that unlocked it.

    Raises:
        KeystoreError: ``WRONG_PASSWORD`` when the password does not decrypt
            the key, ``PASSWORD_REQUIRED`` when no source supplies one, and
            ``KEYCHAIN_FAILED`` when the saved-password lookup fails.
    """

    def attempt(candidate: bytes) -> Tuple[str, bytes]:
        return decrypt_key(key, subject.address, candidate), candidate

    if password is not None:
        return attempt(password if isinstance(password, bytes) else normalize_password(password))
    from_environment = _password_from_environment()
    if from_environment is not None:
        return attempt(normalize_password(from_environment))

    if keychain_available():
        saved = read_keychain(PASSWORD_SERVICE, password_entry_account(subject.address, key["kdfParams"]["salt"]))
        if saved is not None:
            # A saved password is the hex of its bytes; anything else was not
            # written by the keystore.
            if not _SAVED_PASSWORD.fullmatch(saved):
                raise KeystoreError(
                    "KEYCHAIN_FAILED",
                    f"The Keychain entry {PASSWORD_SERVICE} for {subject.address} does not hold a saved password.",
                )
            return attempt(bytes.fromhex(saved))

    if prompt and runtime.terminal_attached():
        for _ in range(_PROMPT_ATTEMPTS):
            typed = runtime.prompt_hidden(
                f"Password for Aleo account {subject.name} ({subject.address}): "
            )
            try:
                return attempt(normalize_password(typed))
            except KeystoreError as error:
                if error.code == "PASSWORD_REQUIRED":
                    runtime.notify("The password must not be empty.")
                elif error.code == "WRONG_PASSWORD":
                    runtime.notify("Wrong password.")
                else:
                    raise
        raise KeystoreError(
            "WRONG_PASSWORD",
            f"No password entered in {_PROMPT_ATTEMPTS} attempts unlocks the account {subject.address}.",
        )

    raise KeystoreError(
        "PASSWORD_REQUIRED",
        f"The account {subject.address} is encrypted and no password is available. "
        f"Set {PASSWORD_VARIABLE}, pass a password, or run with a terminal attached.",
    )


def choose_new_password(
    subject: PasswordSubject,
    explicit: Sequence[Optional[str]],
    environment: bool,
    prompt: bool,
) -> bytes:
    """Obtain the password a new encrypted account file is written with.

    Sources, in order: the explicit candidates, ``ALEO_ACCOUNT_PASSWORD`` when
    allowed, then a hidden prompt that asks twice when a terminal is attached.
    The prompt repeats up to three times while the entries are empty or
    differ.

    Args:
        subject: Account the password will protect.
        explicit: Passwords supplied by the caller, most specific first; the
            first one that is not ``None`` is used.
        environment: Whether ``ALEO_ACCOUNT_PASSWORD`` may supply the
            password. False when replacing an existing password, where the
            variable holds the old one.
        prompt: Whether a prompt may be shown when a terminal is attached.

    Returns:
        The normalised password bytes.

    Raises:
        KeystoreError: ``PASSWORD_REQUIRED`` when a supplied password is
            empty, when no source supplies one, or when the prompt is used up.
    """
    for candidate in explicit:
        if candidate is not None:
            return normalize_password(candidate)
    if environment:
        from_environment = _password_from_environment()
        if from_environment is not None:
            return normalize_password(from_environment)

    if prompt and runtime.terminal_attached():
        for _ in range(_PROMPT_ATTEMPTS):
            first = runtime.prompt_hidden(
                f"New password for Aleo account {subject.name} ({subject.address}): "
            )
            second = runtime.prompt_hidden("Repeat the new password: ")
            password = encode_password(first)
            if not password:
                runtime.notify("The password must not be empty.")
                continue
            # Compared after normalisation, which is the form that is stretched.
            if password == encode_password(second):
                return password
            runtime.notify("The passwords do not match.")
        raise KeystoreError(
            "PASSWORD_REQUIRED",
            f"No password was set for the account {subject.address} in {_PROMPT_ATTEMPTS} attempts.",
        )

    raise KeystoreError(
        "PASSWORD_REQUIRED",
        f"Encrypting the account {subject.address} needs a password. Pass one"
        f"{f', set {PASSWORD_VARIABLE},' if environment else ''} or run with a terminal attached.",
    )
