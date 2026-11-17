"""The account commands behind ``python -m aleo.keystore``.

Six commands set up and inspect a stored account from a terminal:

- ``create`` generates a key and stores it.
- ``import`` stores an existing key or the account of a recovery phrase.
- ``show`` prints the address, name and storage type, never the key.
- ``protect`` changes how the key is stored.
- ``export`` prints the key for a person at a terminal.
- ``migrate`` copies a key out of another tool's state file.

None of them contacts a network. A password is never taken from the command
line: it comes from ``ALEO_ACCOUNT_PASSWORD``, the macOS Keychain or a hidden
prompt.

Exit status: 0 when done, 1 when the command failed, 2 when it needs a person
or a decision (no account, no stored key, no password, no terminal), and 64
for a usage error.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable, Dict, List, NoReturn, Optional, Sequence

from ..mainnet import PrivateKey
from . import StoredAccount
from . import export as export_key
from . import info, migrate, protect, save
from ._errors import KeystoreError
from ._file import resolve_location
from ._keychain import PASSWORD_SERVICE, delete_keychain, keychain_available, require_keychain
from ._mnemonic import clean_phrase, derive_seed, generate_phrase
from ._runtime import runtime
from ._sources import TRIMMED, key_from_environment, read_key_text

EXIT_DONE = 0
EXIT_FAILED = 1
EXIT_NEEDS_PERSON = 2
EXIT_USAGE = 64

# Failures a person has to resolve: an agent or a script stops and asks.
_NEEDS_PERSON = frozenset({"ACCOUNT_NOT_FOUND", "KEY_NOT_STORED", "PASSWORD_REQUIRED", "TERMINAL_REQUIRED"})

_PROGRAM = "python -m aleo.keystore"
_KEY_PREFIX = "APrivateKey1"
_MAX_INDEX = 2**31 - 1


class _Parser(argparse.ArgumentParser):
    """An argument parser whose usage errors exit with status 64."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def _account_index(text: str) -> int:
    """Parse ``--index``: a whole number from 0 through 2**31 - 1."""
    try:
        value = int(text, 10)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a whole number") from None
    if not 0 <= value <= _MAX_INDEX:
        raise argparse.ArgumentTypeError("must be from 0 through 2147483647")
    return value


def _build_parser() -> _Parser:
    parser = _Parser(
        prog=_PROGRAM,
        description="Store an Aleo account on this machine and inspect it.",
        allow_abbrev=False,
    )
    _add_common(parser, top_level=True)
    commands = parser.add_subparsers(dest="command", metavar="<command>", parser_class=_Parser)

    def command(name: str, summary: str) -> argparse.ArgumentParser:
        added = commands.add_parser(name, help=summary, description=summary, allow_abbrev=False)
        _add_common(added, top_level=False)
        return added

    def storage_flags(target: argparse.ArgumentParser, choices: Sequence[str], required: bool) -> None:
        target.add_argument(
            "--storage",
            choices=choices,
            default=None if required else "plaintext",
            help="how to hold the key" + ("" if required else " (default: plaintext)"),
        )
        target.add_argument(
            "--save-password",
            action="store_true",
            help="save the password in the macOS Keychain (encrypted storage on macOS)",
        )

    create = command("create", "Generate a key and store it.")
    storage_flags(create, ("plaintext", "keychain", "encrypted", "none"), required=False)
    create.add_argument(
        "--mnemonic",
        action="store_true",
        help="derive the key from a new recovery phrase, shown once and stored nowhere, "
        "so a wallet can import the account later; requires a terminal",
    )

    imported = command("import", "Store an existing key, or the account of a recovery phrase.")
    storage_flags(imported, ("plaintext", "keychain", "encrypted", "none"), required=False)
    imported.add_argument("--key-file", metavar="<path>", help="text file holding the private key")
    imported.add_argument("--mnemonic-file", metavar="<path>", help="text file holding a recovery phrase")
    imported.add_argument(
        "--index", metavar="<n>", type=_account_index, default=0,
        help="account index for a recovery phrase (default: 0)",
    )
    imported.add_argument(
        "--derivation", choices=("standard", "legacy"), default="standard",
        help="derivation path for a recovery phrase (default: standard)",
    )

    command("show", "Print the address, account name and storage type. Never the key.")

    protected = command("protect", "Change how the key is stored.")
    storage_flags(protected, ("plaintext", "keychain", "encrypted"), required=True)
    protected.add_argument(
        "--forget-password", action="store_true",
        help="remove a saved password from the macOS Keychain",
    )

    command("export", "Print the private key. Requires a terminal.")

    migrated = command("migrate", "Copy a key from a legacy profile.json or state.json into an account file.")
    storage_flags(migrated, ("plaintext", "keychain", "encrypted"), required=False)
    migrated.add_argument(
        "--from", dest="source", metavar="<path>", required=True,
        help="the legacy profile.json or state.json",
    )
    migrated.add_argument(
        "--remove-legacy-key", action="store_true",
        help="strip the key from the legacy file after checking the new account",
    )
    return parser


def _add_common(parser: argparse.ArgumentParser, *, top_level: bool) -> None:
    """Add the flags every command accepts, before or after the command name."""
    # A flag left out after the command name must not reset one given before it.
    unset: Dict[str, Any] = {} if top_level else {"default": argparse.SUPPRESS}
    parser.add_argument("--name", metavar="<name>", help="account name (default: ALEO_ACCOUNT, then default)", **unset)
    parser.add_argument(
        "--dir", metavar="<path>",
        help="accounts directory (default: ALEO_ACCOUNTS_DIR, then ~/.aleo/accounts)", **unset,
    )
    parser.add_argument(
        "--json", action="store_true", help="machine-readable output on standard output", **unset
    )


def _out(text: str) -> None:
    sys.stdout.write(f"{text}\n")
    sys.stdout.flush()


def _report(stored: StoredAccount, as_json: bool, sentence: str) -> None:
    """Print the outcome of a command that leaves an account file in place."""
    if as_json:
        _out(
            json.dumps(
                {
                    "name": stored.name,
                    "address": stored.address,
                    "storage": stored.storage,
                    "path": str(stored.path),
                },
                ensure_ascii=False,
            )
        )
    else:
        _out(sentence)


def _summary(stored: StoredAccount, verb: str) -> str:
    return f"{verb} Aleo account {stored.name} ({stored.address}) with {stored.storage} storage in {stored.path}."


def _require_terminal(what: str, as_json: bool, secret: str = "the private key") -> None:
    """Stop a command that shows a secret unless a person is at a terminal.

    Args:
        what: The command, as the person typed it.
        as_json: Whether ``--json`` was given.
        secret: What the command shows, named in the message.

    Raises:
        KeystoreError: ``TERMINAL_REQUIRED`` when ``--json`` was given, or
            when standard input, output or error is not a terminal.
    """
    if as_json:
        raise KeystoreError(
            "TERMINAL_REQUIRED",
            f"{what} shows {secret} to a person at a terminal and has no --json output.",
        )
    if not (runtime.terminal_attached() and runtime.output_is_terminal()):
        raise KeystoreError(
            "TERMINAL_REQUIRED",
            f"{what} shows {secret} and runs only with a terminal attached to its input and output.",
        )


def _storage_notes(stored: StoredAccount, save_password: bool) -> None:
    """Tell the person what the chosen storage does not do on this system."""
    if save_password and not (stored.storage == "encrypted" and keychain_available()):
        runtime.notify(
            "--save-password applies to encrypted storage on macOS only; no password was saved."
        )
    if stored.storage == "plaintext" and runtime.platform == "win32":
        runtime.notify(
            f"Windows does not enforce owner-only file modes: {stored.path} is protected only by the "
            "access rules of its folder."
        )


def _key_of_phrase(text: str, index: int, derivation: str) -> str:
    """Derive the private key of a recovery phrase after tidying it."""
    seed = derive_seed(clean_phrase(text), index=index, derivation=derivation)
    return str(PrivateKey.from_seed_unchecked(list(seed)))


def _create(args: argparse.Namespace) -> int:
    storage: str = args.storage
    with_phrase: bool = args.mnemonic
    # A phrase, or a key that is stored nowhere, is shown once, so a person
    # must be there to receive it. Checked before anything is generated or
    # written.
    if with_phrase:
        _require_terminal("create --mnemonic", args.json, "the recovery phrase")
    if storage == "none":
        _require_terminal("create --storage none", args.json)
    phrase = generate_phrase() if with_phrase else None
    key = _key_of_phrase(phrase, 0, "standard") if phrase is not None else str(PrivateKey.random())
    stored = save(
        key, name=args.name, dir=args.dir, storage=args.storage, save_password=args.save_password
    )
    _report(stored, args.json, _summary(stored, "Created"))
    _storage_notes(stored, args.save_password)
    if phrase is not None:
        _out(f"Recovery phrase: {phrase}")
        runtime.notify(
            "This phrase is shown once and stored nowhere. Write it down and keep it offline. "
            "A wallet imports this account from it, and whoever holds it controls every account "
            "derived from it."
        )
    if storage == "none":
        _out(f"Private key: {key}")
        runtime.notify(
            "This key is stored nowhere. Keep it safe and supply it through ALEO_PRIVATE_KEY or "
            "ALEO_PRIVATE_KEY_FILE on every run."
        )
    elif storage == "keychain" and phrase is None:
        runtime.notify(
            "The key of this account exists only in the macOS Keychain of this Mac. "
            f"Back it up with: {_PROGRAM} export"
        )
    return EXIT_DONE


def _key_to_import(args: argparse.Namespace) -> str:
    """Find the key to import: a file flag, the key variables, then a prompt."""
    if args.key_file is not None:
        return read_key_text(args.key_file, f"The key file {args.key_file}")
    if args.mnemonic_file is not None:
        phrase = read_key_text(args.mnemonic_file, f"The recovery phrase file {args.mnemonic_file}")
        return _key_of_phrase(phrase, args.index, args.derivation)
    from_environment = key_from_environment()
    if from_environment is not None:
        return from_environment.private_key
    if not runtime.terminal_attached():
        raise KeystoreError(
            "ACCOUNT_NOT_FOUND",
            "No key to import: pass --key-file or --mnemonic-file, set ALEO_PRIVATE_KEY or "
            "ALEO_PRIVATE_KEY_FILE, or run with a terminal attached.",
        )
    typed = ""
    try:
        typed = runtime.prompt_hidden("Private key or recovery phrase to import: ").strip(TRIMMED)
    except KeystoreError as error:
        if error.code != "PASSWORD_REQUIRED":
            raise
    if not typed:
        raise KeystoreError("EMPTY_KEY_SOURCE", "No key or recovery phrase was entered.")
    if typed.startswith(_KEY_PREFIX):
        return typed
    return _key_of_phrase(typed, args.index, args.derivation)


def _import(args: argparse.Namespace) -> int:
    key = _key_to_import(args)
    stored = save(
        key, name=args.name, dir=args.dir, storage=args.storage, save_password=args.save_password
    )
    _report(stored, args.json, _summary(stored, "Imported"))
    _storage_notes(stored, args.save_password)
    return EXIT_DONE


def _existing(args: argparse.Namespace) -> StoredAccount:
    """Describe the selected account file, failing when there is none."""
    stored = info(name=args.name, dir=args.dir)
    if stored is None:
        path = resolve_location(args.name, args.dir).path
        raise KeystoreError("ACCOUNT_NOT_FOUND", f"There is no account file at {path}.")
    return stored


def _show(args: argparse.Namespace) -> int:
    stored = _existing(args)
    _report(
        stored,
        args.json,
        f"name:    {stored.name}\naddress: {stored.address}\nstorage: {stored.storage}\npath:    {stored.path}",
    )
    return EXIT_DONE


def _protect(args: argparse.Namespace) -> int:
    if args.storage is None:
        # Only the saved password goes; the account file stays as it is.
        stored = _existing(args)
        require_keychain("A saved password exists only in the macOS Keychain.")
        delete_keychain(PASSWORD_SERVICE, stored.address)
        _report(
            stored,
            args.json,
            f"Removed the saved password of Aleo account {stored.name} ({stored.address}) from the Keychain.",
        )
        return EXIT_DONE
    stored = protect(storage=args.storage, name=args.name, dir=args.dir, save_password=args.save_password)
    if args.forget_password and keychain_available():
        delete_keychain(PASSWORD_SERVICE, stored.address)
    _report(stored, args.json, _summary(stored, "Stored"))
    _storage_notes(stored, args.save_password)
    return EXIT_DONE


def _export(args: argparse.Namespace) -> int:
    _require_terminal("export", args.json)
    stored = _existing(args)
    answer = runtime.prompt_visible(
        f"Show the private key of Aleo account {stored.name} ({stored.address}) on this terminal? [y/N] "
    )
    if answer.strip().lower() not in ("y", "yes"):
        runtime.notify("Not confirmed. The key was not shown.")
        return EXIT_FAILED
    _out(export_key(name=args.name, dir=args.dir))
    return EXIT_DONE


def _migrate(args: argparse.Namespace) -> int:
    stored = migrate(
        args.source,
        name=args.name,
        dir=args.dir,
        storage=args.storage,
        save_password=args.save_password,
        remove_legacy_key=args.remove_legacy_key,
    )
    _report(stored, args.json, _summary(stored, "Migrated"))
    _storage_notes(stored, args.save_password)
    if args.remove_legacy_key:
        runtime.notify(f"The key was removed from {args.source}.")
    else:
        runtime.notify(
            f"A plaintext copy of the key remains in {args.source}. "
            "Run migrate again with --remove-legacy-key to strip it."
        )
    return EXIT_DONE


_COMMANDS: Dict[str, Callable[[argparse.Namespace], int]] = {
    "create": _create,
    "import": _import,
    "show": _show,
    "protect": _protect,
    "export": _export,
    "migrate": _migrate,
}


def main(argv: Optional[List[str]] = None) -> int:
    """Run one account command and report its outcome.

    Results go to standard output; notes, prompts and errors go to standard
    error as ``<CODE>: <message>``. With ``--json``, a result is one JSON
    object with ``name``, ``address``, ``storage`` and ``path``, and an error
    is ``{"error": {"code": ..., "message": ...}}`` on standard output. The
    private key appears only in the output of ``export`` and of
    ``create --storage none``, and never with ``--json``.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        The exit status: 0 when done, 1 when the command failed, 2 when it
        needs a person (``ACCOUNT_NOT_FOUND``, ``KEY_NOT_STORED``,
        ``PASSWORD_REQUIRED`` or ``TERMINAL_REQUIRED``).

    Raises:
        SystemExit: With status 64 for a usage error, after printing the
            usage, and with status 0 after ``--help``.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.error("a command is required: create, import, show, protect, export or migrate")
    if args.command == "import" and args.key_file is not None and args.mnemonic_file is not None:
        parser.error("--key-file and --mnemonic-file cannot be combined")
    if args.command == "protect":
        if args.storage is None and not args.forget_password:
            parser.error("protect needs --storage, or --forget-password to only remove a saved password")
        if args.save_password and args.forget_password:
            parser.error("--save-password and --forget-password cannot be combined")
    try:
        return _COMMANDS[args.command](args)
    except KeystoreError as error:
        if args.json:
            _out(json.dumps({"error": {"code": error.code, "message": str(error)}}, ensure_ascii=False))
        else:
            runtime.notify(f"{error.code}: {error}")
        return EXIT_NEEDS_PERSON if error.code in _NEEDS_PERSON else EXIT_FAILED
    except OSError as error:
        runtime.notify(f"aleo.keystore: {error}")
        return EXIT_FAILED


__all__ = ["main"]
