"""The error type raised by account storage and recovery-phrase handling."""
from __future__ import annotations

from .._client_common import AleoError


class KeystoreError(AleoError):
    """An account could not be stored, found, or unlocked.

    Branch on :attr:`code`, which is stable across releases and shared with
    the other Aleo SDKs; the message is for people and may change. Messages
    name addresses and file paths and never contain a private key, a recovery
    phrase, or a password, so they are safe to log.

    A file system failure on the account file or its directory other than a
    missing file, such as a denied permission, is not wrapped: it is raised as
    the ``OSError`` the operating system reported.

    Codes
    -----
    ``ACCOUNT_NOT_FOUND``
        No account file exists and no key was supplied another way.
    ``KEY_NOT_STORED``
        The account file records only an address, and no key was supplied.
    ``ACCOUNT_EXISTS``
        Saving would replace an account file that is already there.
    ``ACCOUNT_BUSY``
        Another process is changing the account file, or changed it while
        this operation was in progress. Nothing was changed; try again.
    ``ADDRESS_MISMATCH``
        A key belongs to a different address than the one expected.
    ``KEY_CONFLICT``
        Two environment variables supply different keys, or a legacy state
        file holds two different keys.
    ``EMPTY_KEY_SOURCE``
        A key variable or key file is set but holds no key.
    ``INVALID_KEY``
        A private key string cannot be parsed.
    ``INVALID_MNEMONIC``
        A recovery phrase has a wrong word, word count, or checksum.
    ``INVALID_NAME``
        The account name is not lower-case letters, digits, ``_`` and ``-``.
    ``INVALID_ACCOUNT_FILE``
        The account file is not valid JSON or breaks the file format.
    ``UNSUPPORTED_VERSION``
        The account file was written in a newer format.
    ``UNSUPPORTED_STORAGE``
        The storage type is unknown, or not allowed for this operation.
    ``KEYCHAIN_UNAVAILABLE``
        Keychain storage was requested, or found, on a system other than macOS.
    ``KEYCHAIN_ITEM_NOT_FOUND``
        The account file points at a Keychain entry that no longer exists.
    ``KEYCHAIN_FAILED``
        The macOS ``security`` tool failed for another reason.
    ``PASSWORD_REQUIRED``
        A password is needed and none was supplied.
    ``WRONG_PASSWORD``
        The password did not unlock the key.
    ``TERMINAL_REQUIRED``
        A command that shows a key was run without a terminal.
    ``LEGACY_KEY_NOT_FOUND``
        The file given to ``migrate`` holds no key.
    """

    def __init__(self, code: str, message: str) -> None:
        """Build the error for one failure.

        Args:
            code: Stable identifier of the failure, one of the codes above.
            message: Detail for a person. Must not contain a key, a recovery
                phrase or a password.
        """
        super().__init__(message)
        self.code = code


__all__ = ["KeystoreError"]
