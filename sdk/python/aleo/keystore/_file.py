"""The account file: where it lives, what it holds and how it is written.

Every writer of the format produces the same bytes for the same account, and
every reader rejects a file that departs from it, so an account written by one
SDK or tool is read unchanged by the others.
"""
from __future__ import annotations

import base64
import binascii
import json
import os
import re
import stat
from typing import Any, Dict, List, NamedTuple, NoReturn, Optional, Sequence, Tuple, Union, cast

from ._encrypted import SALT_BYTES, SCRYPT_MAX_N, SCRYPT_MIN_N, SCRYPT_P, SCRYPT_R
from ._errors import KeystoreError
from ._keychain import KEY_SERVICE
from ._runtime import runtime

# Format version written to, and accepted from, an account file.
FILE_VERSION = 1

# Account names: lower case so they are distinct on case-insensitive disks.
_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")

# Standard base64 with padding, the only salt encoding accepted.
_BASE64_PATTERN = re.compile(r"(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?")

# The four ways an account file can hold, or point at, the private key.
STORAGE_TYPES: Tuple[str, ...] = ("none", "plaintext", "keychain", "encrypted")

# The parsed content of an account file: ``version``, ``address`` and ``key``.
AccountFile = Dict[str, Any]


class Location(NamedTuple):
    """A resolved account file location.

    Attributes:
        name: Account name.
        dir: Absolute path of the accounts directory.
        path: Absolute path of the account file, ``<dir>/<name>.json``.
    """

    name: str
    dir: str
    path: str


class ExistingFile(NamedTuple):
    """An account file as found on disk.

    Attributes:
        file: The validated content.
        data: The raw bytes, kept so the file can be put back unchanged.
    """

    file: AccountFile
    data: bytes


def resolve_location(
    name: Optional[str] = None, dir: Union[str, "os.PathLike[str]", None] = None
) -> Location:
    """Resolve which account file a call refers to.

    The name comes from the argument, then ``ALEO_ACCOUNT``, then ``default``.
    The directory comes from the argument, then ``ALEO_ACCOUNTS_DIR``, then
    ``<home>/.aleo/accounts``; a leading ``~`` expands to the home directory.
    An empty variable or an empty ``dir`` counts as absent. Reads the
    environment and does not touch the disk.

    Raises:
        KeystoreError: ``INVALID_NAME`` when the name does not match
            ``^[a-z0-9][a-z0-9_-]{0,63}$``.
    """
    # Checked at run time as well: an untyped caller can pass anything.
    chosen = cast(object, name if name is not None else (os.environ.get("ALEO_ACCOUNT") or "default"))
    if not isinstance(chosen, str) or not _NAME_PATTERN.fullmatch(chosen):
        raise KeystoreError(
            "INVALID_NAME",
            'The account name must be 1 to 64 characters of a-z, 0-9, "_" or "-", '
            "starting with a letter or digit.",
        )
    home = runtime.home()
    directory = os.fspath(dir) if dir is not None else ""
    directory = directory or os.environ.get("ALEO_ACCOUNTS_DIR") or os.path.join(home, ".aleo", "accounts")
    # Only "~" and "~/..." expand; "~name" is left as an ordinary directory name.
    if directory == "~":
        directory = home
    elif directory.startswith("~/") or (runtime.platform == "win32" and directory.startswith("~\\")):
        directory = os.path.join(home, directory[2:])
    directory = os.path.abspath(directory)
    return Location(chosen, directory, os.path.join(directory, f"{chosen}.json"))


def _integer(value: object) -> Optional[int]:
    """Return a JSON number as an integer, or ``None`` when it is not one.

    A number written with a zero fraction, such as ``1.0``, is the integer it
    equals, as in every JSON reader that has a single number type.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _reject_constant(_: str) -> NoReturn:
    raise ValueError("not a JSON value")


def loads_strict(text: str) -> Any:
    """Parse JSON text, refusing the ``NaN`` and ``Infinity`` extensions."""
    return json.loads(text, parse_constant=_reject_constant)


def parse_account_file(text: str, path: str) -> AccountFile:
    """Parse and validate the text of an account file.

    Accepts exactly the fields of the format: no field may be missing and none
    may be added. An unknown version or storage type is reported as such
    before the remaining fields are examined, since a later format may carry
    others.

    Args:
        text: File content.
        path: File path, named in error messages.

    Returns:
        The validated content, with fields in the order of the format.

    Raises:
        KeystoreError: ``INVALID_ACCOUNT_FILE``, ``UNSUPPORTED_VERSION`` or
            ``UNSUPPORTED_STORAGE``.
    """

    def invalid(problem: str) -> KeystoreError:
        return KeystoreError("INVALID_ACCOUNT_FILE", f"Account file {path} is not valid: {problem}.")

    def exact(found: Dict[str, Any], fields: Sequence[str], where: str) -> None:
        # Requires an object to hold exactly the named fields.
        for field in fields:
            if field not in found:
                raise invalid(f'{where} is missing "{field}"')
        for field in found:
            if field not in fields:
                raise invalid(f'{where} has the unknown field "{field}"')

    def text_of(value: object, field: str) -> str:
        if not isinstance(value, str):
            raise invalid(f'"{field}" must be a string')
        return value

    parsed: Any = None
    is_json = True
    try:
        parsed = loads_strict(text)
    except (ValueError, RecursionError):
        is_json = False
    if not is_json:
        # The parser's message can quote the file, which may hold a key, so the
        # error is raised without it.
        raise invalid("it is not JSON")
    if not isinstance(parsed, dict):
        raise invalid("it does not hold a JSON object")
    raw = cast(Dict[str, Any], parsed)

    version = _integer(raw.get("version"))
    if version is None:
        raise invalid('"version" must be an integer')
    if version != FILE_VERSION:
        raise KeystoreError(
            "UNSUPPORTED_VERSION",
            f"Account file {path} has version {version}; this reader supports version {FILE_VERSION}.",
        )
    exact(raw, ["version", "address", "key"], "the file")
    address = text_of(raw["address"], "address")
    if not isinstance(raw["key"], dict):
        raise invalid('"key" must be an object')
    key = cast(Dict[str, Any], raw["key"])
    kind = text_of(key.get("type"), "key.type")

    if kind == "none":
        exact(key, ["type"], '"key"')
        return {"version": FILE_VERSION, "address": address, "key": {"type": kind}}
    if kind == "plaintext":
        exact(key, ["type", "privateKey"], '"key"')
        private_key = text_of(key["privateKey"], "key.privateKey")
        return {"version": FILE_VERSION, "address": address, "key": {"type": kind, "privateKey": private_key}}
    if kind == "keychain":
        exact(key, ["type", "service", "account"], '"key"')
        service = text_of(key["service"], "key.service")
        account = text_of(key["account"], "key.account")
        # Pinned so a crafted file cannot direct the reader to another entry.
        if service != KEY_SERVICE:
            raise invalid(f'"key.service" must be "{KEY_SERVICE}"')
        if account != address:
            raise invalid('"key.account" must equal "address"')
        return {
            "version": FILE_VERSION,
            "address": address,
            "key": {"type": kind, "service": service, "account": account},
        }
    if kind == "encrypted":
        exact(key, ["type", "kdf", "kdfParams", "ciphertext"], '"key"')
        if key["kdf"] != "scrypt":
            raise invalid('"key.kdf" must be "scrypt"')
        if not isinstance(key["kdfParams"], dict):
            raise invalid('"key.kdfParams" must be an object')
        params = cast(Dict[str, Any], key["kdfParams"])
        exact(params, ["N", "r", "p", "salt"], '"key.kdfParams"')
        cost = _integer(params["N"])
        # The upper bound stops a crafted file from demanding unbounded memory.
        if cost is None or cost < SCRYPT_MIN_N or cost > SCRYPT_MAX_N or cost & (cost - 1) != 0:
            raise invalid(f'"key.kdfParams.N" must be a power of two from {SCRYPT_MIN_N} to {SCRYPT_MAX_N}')
        if _integer(params["r"]) != SCRYPT_R:
            raise invalid(f'"key.kdfParams.r" must be {SCRYPT_R}')
        if _integer(params["p"]) != SCRYPT_P:
            raise invalid(f'"key.kdfParams.p" must be {SCRYPT_P}')
        salt = text_of(params["salt"], "key.kdfParams.salt")
        if not _BASE64_PATTERN.fullmatch(salt) or _decoded_length(salt) < SALT_BYTES:
            raise invalid(f'"key.kdfParams.salt" must be base64 of at least {SALT_BYTES} bytes')
        ciphertext = text_of(key["ciphertext"], "key.ciphertext")
        return {
            "version": FILE_VERSION,
            "address": address,
            "key": {
                "type": kind,
                "kdf": "scrypt",
                "kdfParams": {"N": cost, "r": SCRYPT_R, "p": SCRYPT_P, "salt": salt},
                "ciphertext": ciphertext,
            },
        }
    raise KeystoreError(
        "UNSUPPORTED_STORAGE",
        f'Account file {path} uses the storage type "{kind}", which this reader does not support.',
    )


def _decoded_length(salt: str) -> int:
    try:
        return len(base64.b64decode(salt, validate=True))
    except (binascii.Error, ValueError):
        return 0


def serialize_account_file(file: AccountFile) -> str:
    """Render an account file to its exact text.

    Fields appear in the fixed order of the format, with two-space
    indentation, LF line endings and one trailing newline, so every writer of
    the format produces the same bytes for the same account.
    """
    key: Dict[str, Any] = file["key"]
    kind: str = key["type"]
    # Each variant is rebuilt field by field so the order never depends on how
    # the caller assembled the mapping.
    entry: Dict[str, Any]
    if kind == "none":
        entry = {"type": "none"}
    elif kind == "plaintext":
        entry = {"type": "plaintext", "privateKey": key["privateKey"]}
    elif kind == "keychain":
        entry = {"type": "keychain", "service": key["service"], "account": key["account"]}
    else:
        params: Dict[str, Any] = key["kdfParams"]
        entry = {
            "type": "encrypted",
            "kdf": key["kdf"],
            "kdfParams": {"N": params["N"], "r": params["r"], "p": params["p"], "salt": params["salt"]},
            "ciphertext": key["ciphertext"],
        }
    content = {"version": file["version"], "address": file["address"], "key": entry}
    return json.dumps(content, indent=2, ensure_ascii=False) + "\n"


def read_account_file(path: str) -> Optional[ExistingFile]:
    """Read and validate an account file.

    Tightens the file to owner-only access (mode 0600) when it is wider,
    except on Windows, where file modes are not enforced.

    Returns:
        The validated content and the raw bytes, or ``None`` when no file
        exists at the path.

    Raises:
        KeystoreError: The file breaks the format.
        OSError: Any file system failure other than a missing file.
    """
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    except FileNotFoundError:
        return None
    # Windows has no fchmod and does not enforce these modes.
    tighten = getattr(os, "fchmod", None)
    with os.fdopen(descriptor, "rb") as handle:
        if runtime.platform != "win32" and tighten is not None:
            status = os.fstat(handle.fileno())
            # Only a regular file is re-moded: a directory in place of the
            # file fails below and keeps its mode.
            if stat.S_ISREG(status.st_mode) and stat.S_IMODE(status.st_mode) & ~0o600:
                # Best effort: a file the user cannot re-mode is still readable.
                try:
                    tighten(handle.fileno(), 0o600)
                except OSError:
                    pass
        data = handle.read()
    text: Optional[str]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = None
    if text is None:
        raise KeystoreError(
            "INVALID_ACCOUNT_FILE", f"Account file {path} is not valid: it is not UTF-8 text."
        )
    return ExistingFile(parse_account_file(text, path), data)


def write_file_atomic(path: str, data: Union[str, bytes]) -> None:
    """Replace a file's content through a temporary file and a rename.

    The temporary file is created in the same directory with mode 0600 before
    any content is written, so the content is never readable by other users
    and a reader never sees a partial file.

    Args:
        path: Destination path. Its directory must exist.
        data: Complete new content; text is written as UTF-8.
    """
    payload = data.encode("utf-8") if isinstance(data, str) else data
    temporary = os.path.join(os.path.dirname(path), f".{os.urandom(8).hex()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def write_file_exclusive(path: str, data: Union[str, bytes]) -> None:
    """Create a file that must not exist yet, through a temporary file and a hard link.

    A rename replaces whatever is at the destination; a hard link fails when a
    file is already there. Two callers creating the same path at the same
    moment therefore cannot both succeed, and neither can replace the other's
    file. The temporary file is created in the same directory with mode 0600
    before any content is written, so a reader never sees a partial file. On a
    file system without hard links the destination is created directly, still
    refusing an existing file.

    Args:
        path: Destination path. Its directory must exist.
        data: Complete content; text is written as UTF-8.

    Raises:
        FileExistsError: A file already exists at the path. Other file system
            errors propagate unchanged.
    """
    payload = data.encode("utf-8") if isinstance(data, str) else data
    temporary = os.path.join(os.path.dirname(path), f".{os.urandom(8).hex()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise
        except (OSError, NotImplementedError):
            # No hard links here: create the destination itself, exclusively.
            target = os.open(path, flags, 0o600)
            with os.fdopen(target, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
    finally:
        try:
            os.unlink(temporary)
        except OSError:
            pass


def write_account_file(location: Location, file: AccountFile, exclusive: bool = False) -> None:
    """Write an account file, creating the accounts directory when needed.

    A directory created here gets mode 0700; missing parents get the default
    mode, and an existing directory keeps its mode. The file is written with
    mode 0600. It replaces any file already at the path, unless *exclusive* is
    set.

    Args:
        location: Resolved account location.
        file: Account content.
        exclusive: Whether the file must not exist yet. Used when storing a
            new account, so that two callers storing different accounts under
            one name cannot replace each other's file.

    Raises:
        FileExistsError: *exclusive* is set and a file already exists at the
            path.
    """
    os.makedirs(os.path.dirname(location.dir), exist_ok=True)
    try:
        os.mkdir(location.dir, 0o700)
    except FileExistsError:
        pass
    else:
        # mkdir applies the umask; the directory is owner-only regardless of it.
        if runtime.platform != "win32":
            os.chmod(location.dir, 0o700)
    text = serialize_account_file(file)
    if exclusive:
        write_file_exclusive(location.path, text)
    else:
        write_file_atomic(location.path, text)


__all__: List[str] = [
    "AccountFile",
    "ExistingFile",
    "FILE_VERSION",
    "Location",
    "STORAGE_TYPES",
    "loads_strict",
    "parse_account_file",
    "read_account_file",
    "resolve_location",
    "serialize_account_file",
    "write_account_file",
    "write_file_atomic",
    "write_file_exclusive",
]
