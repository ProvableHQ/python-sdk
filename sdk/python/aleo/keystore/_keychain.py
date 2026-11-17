"""macOS Keychain access through the system ``security`` tool.

Entries are generic passwords in the login keychain, one per address. A value
is written through the tool's standard input so that it never appears in the
process arguments.
"""
from __future__ import annotations

import re
from typing import Optional

from ._errors import KeystoreError
from ._runtime import runtime

# Keychain service holding private keys; the entry's account is the address.
KEY_SERVICE = "aleo-account-key"

# Keychain service holding saved passwords; the entry's account is the address.
PASSWORD_SERVICE = "aleo-account-password"

# Exit status ``security`` returns when the requested entry does not exist.
_ITEM_NOT_FOUND_STATUS = 44

# Text that can be placed on a ``security -i`` command line without quoting.
# Addresses, private keys and hex strings all satisfy it.
_UNQUOTED = re.compile(r"[A-Za-z0-9]+")

# Account names as they may appear, unquoted, in an entry's label.
_LABEL_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


def keychain_available() -> bool:
    """Report whether the macOS Keychain exists on this system."""
    return runtime.platform == "darwin"


def require_keychain(detail: str) -> None:
    """Stop an operation that needs the macOS Keychain on another system.

    Args:
        detail: Sentence naming what needed the Keychain and what to do
            instead.

    Raises:
        KeystoreError: ``KEYCHAIN_UNAVAILABLE`` outside macOS.
    """
    if not keychain_available():
        raise KeystoreError(
            "KEYCHAIN_UNAVAILABLE",
            f"The macOS Keychain is not available on this system. {detail}",
        )


def _failed(action: str, service: str, address: str, reason: str) -> KeystoreError:
    """Build the error for a Keychain failure without echoing the tool's output."""
    return KeystoreError(
        "KEYCHAIN_FAILED",
        f"The security tool {reason} while {action} the Keychain entry {service} for {address}.",
    )


def _exit_reason(status: Optional[int]) -> str:
    """Describe a non-zero or missing exit status of the ``security`` tool."""
    if status is None:
        return "could not be started or did not finish within 15 seconds"
    return f"exited with status {status}"


def read_keychain(service: str, address: str) -> Optional[str]:
    """Read one generic password from the macOS login keychain.

    Returns:
        The stored value, or ``None`` when the entry does not exist.

    Raises:
        KeystoreError: ``KEYCHAIN_FAILED`` when the tool fails for any reason
            other than a missing entry.
    """
    status, stdout = runtime.security(
        ["find-generic-password", "-s", service, "-a", address, "-w"], None
    )
    if status == _ITEM_NOT_FOUND_STATUS:
        return None
    if status != 0:
        raise _failed("reading", service, address, _exit_reason(status))
    # The tool prints the value followed by one line ending.
    if stdout.endswith("\r\n"):
        return stdout[:-2]
    if stdout.endswith("\n"):
        return stdout[:-1]
    return stdout


def write_keychain(service: str, address: str, value: str, name: str) -> None:
    """Create or replace one generic password in the macOS login keychain.

    The command travels on the tool's standard input, so the value never
    appears in the process arguments. The entry is read back afterwards to
    confirm it holds the value.

    The entry is found by service and address. Its label, the name Keychain
    Access shows, is ``<service>:<name>``, so entries of several accounts can
    be told apart; the label takes no part in finding the entry.

    Args:
        service: Keychain service name.
        address: Aleo address, used as the entry's account.
        value: Secret to store. Must be ASCII letters and digits only.
        name: Account name, shown in the entry's label.

    Raises:
        KeystoreError: ``KEYCHAIN_FAILED`` when the tool fails or the entry
            does not hold the value afterwards.
    """
    # Nothing here is quoted, so anything able to split or extend the command
    # line is refused before the tool runs.
    if (
        not _UNQUOTED.fullmatch(address)
        or not _UNQUOTED.fullmatch(value)
        or not _LABEL_NAME.fullmatch(name)
    ):
        raise _failed(
            "writing", service, address, "was not run, because the value cannot be passed unquoted,"
        )
    command = f"add-generic-password -U -s {service} -a {address} -l {service}:{name} -w {value}\n"
    status, _ = runtime.security(["-i"], command)
    if status != 0:
        raise _failed("writing", service, address, _exit_reason(status))
    if read_keychain(service, address) != value:
        raise _failed("verifying", service, address, "did not store the value")


def delete_keychain(service: str, address: str) -> None:
    """Delete one generic password from the macOS login keychain.

    An entry that is already absent counts as deleted.

    Raises:
        KeystoreError: ``KEYCHAIN_FAILED`` when the tool fails for any reason
            other than a missing entry.
    """
    status, _ = runtime.security(["delete-generic-password", "-s", service, "-a", address], None)
    if status != 0 and status != _ITEM_NOT_FOUND_STATUS:
        raise _failed("deleting", service, address, _exit_reason(status))
