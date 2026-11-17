"""The private key a caller supplies through the environment.

Two variables hold a key and two name a text file that holds one. Every
variable that is set must supply a key, and all of them must supply the same
one, so a stale variable is reported instead of silently selecting another
account.
"""
from __future__ import annotations

import errno
import os
from typing import List, NamedTuple, Optional, Union

from ._encrypted import parse_key
from ._errors import KeystoreError

PathLike = Union[str, "os.PathLike[str]"]

# Variables holding a private key, in order of precedence.
KEY_VARIABLES = ("ALEO_PRIVATE_KEY", "SHIELD_SWAP_PRIVATE_KEY")

# Variables naming a text file that holds only a private key, in order of
# precedence.
KEY_FILE_VARIABLES = ("ALEO_PRIVATE_KEY_FILE", "SHIELD_SWAP_PRIVATE_KEY_FILE")

# The characters removed from both ends of a key variable or key file: the
# white space and line terminators of ECMAScript's ``String.prototype.trim``,
# so every SDK accepts the same files, including one that starts with a
# byte-order mark.
TRIMMED = (
    "\t\n\x0b\x0c\r \xa0\u1680"
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000\ufeff"
)


class Resolved(NamedTuple):
    """A key found by the lookup, with the source and a phrase naming it."""

    private_key: str
    address: str
    source: str
    origin: str


def error_code(error: OSError) -> str:
    """Name a file system failure by its symbolic code, such as ``ENOENT``."""
    return errno.errorcode.get(error.errno or 0, "unknown error")


def key_from_environment() -> Optional[Resolved]:
    """Collect the private key supplied through the environment.

    Reads the two key variables and the files named by the two key-file
    variables. Every variable that is set must supply a key, and all of them
    must supply the same one.

    Returns:
        The key with the highest precedence, or ``None`` when none of the
        variables is set.

    Raises:
        KeystoreError: ``EMPTY_KEY_SOURCE`` when a variable is set but empty
            or its file is empty or unreadable, ``INVALID_KEY`` when a value
            is not a key, and ``KEY_CONFLICT`` when two variables supply
            different keys.
    """
    found: List[Resolved] = []

    for variable in KEY_VARIABLES:
        value = os.environ.get(variable)
        if value is None:
            continue
        text = value.strip(TRIMMED)
        if not text:
            raise KeystoreError("EMPTY_KEY_SOURCE", f"{variable} is set but empty.")
        found.append(Resolved(*parse_key(text, variable), "env", variable))

    for variable in KEY_FILE_VARIABLES:
        path = os.environ.get(variable)
        if path is None:
            continue
        if not path.strip(TRIMMED):
            raise KeystoreError("EMPTY_KEY_SOURCE", f"{variable} is set but empty.")
        origin = f"The key file {path} named by {variable}"
        text = read_key_text(path, origin)
        found.append(Resolved(*parse_key(text, origin), "env-file", origin))

    # The first entry is the one with the highest precedence. The others only
    # have to agree with it.
    if not found:
        return None
    first = found[0]
    for other in found[1:]:
        if other.address != first.address:
            raise KeystoreError(
                "KEY_CONFLICT",
                f"{first.origin} and {other.origin} hold different keys "
                f"({first.address} and {other.address}). Unset one of them.",
            )
    return first


def read_key_text(path: PathLike, origin: str) -> str:
    """Read a text file that holds only a key or a recovery phrase.

    Returns:
        The content with surrounding white space removed.

    Raises:
        KeystoreError: ``EMPTY_KEY_SOURCE`` when the file cannot be read or
            holds nothing.
    """
    try:
        with open(path, "rb") as handle:
            content = handle.read().decode("utf-8", errors="replace")
    except OSError as error:
        raise KeystoreError(
            "EMPTY_KEY_SOURCE", f"{origin} could not be read ({error_code(error)})."
        ) from error
    text = content.strip(TRIMMED)
    if not text:
        raise KeystoreError("EMPTY_KEY_SOURCE", f"{origin} is empty.")
    return text
