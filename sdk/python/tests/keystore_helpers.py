"""Shared scaffolding for the account keystore tests.

Every test that uses the ``sandbox`` fixture runs in a fresh temporary
directory, with the key, account and password variables cleared and the home
directory, the Keychain tool and the terminal replaced, so nothing reads or
writes the developer's own accounts or Keychain.
"""
from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pytest

from aleo.keystore import KeystoreError
from aleo.keystore._runtime import KeystoreRuntime, runtime
from aleo.mainnet import Account

VECTORS_DIR = Path(__file__).parent / "fixtures" / "keystore-vectors"
VECTORS: Dict[str, Any] = json.loads((VECTORS_DIR / "vectors.json").read_text(encoding="utf-8"))

# The public sample account. Never funded.
SAMPLE = SimpleNamespace(
    address=VECTORS["account"]["address"], private_key=VECTORS["account"]["privateKey"]
)

# Variables a developer's shell could leak into a test.
VARIABLES = (
    "ALEO_PRIVATE_KEY",
    "ALEO_PRIVATE_KEY_FILE",
    "ALEO_ACCOUNT",
    "ALEO_ACCOUNTS_DIR",
    "ALEO_ACCOUNT_PASSWORD",
    "SHIELD_SWAP_PRIVATE_KEY",
    "SHIELD_SWAP_PRIVATE_KEY_FILE",
)

# Test-only passwords. Never used for a funded account.
PASSWORD = "correct horse battery staple"
OTHER_PASSWORD = "another test password"

POSIX = os.name != "nt"

SecurityResult = Tuple[Optional[int], str]


class FakeSecurity:
    """An in-memory stand-in for the macOS ``security`` tool."""

    def __init__(self) -> None:
        self.entries: Dict[str, str] = {}
        # The label each entry was last written with.
        self.labels: Dict[str, str] = {}
        self.calls: List[Dict[str, Any]] = []
        self._override: Optional[Callable[[Sequence[str]], Optional[SecurityResult]]] = None

    def force(self, override: Optional[Callable[[Sequence[str]], Optional[SecurityResult]]]) -> None:
        """Force the outcome of matching invocations."""
        self._override = override

    def run(self, args: Sequence[str], stdin: Optional[str] = None) -> SecurityResult:
        args = list(args)
        self.calls.append({"args": args, "stdin": stdin})
        forced = self._override(args) if self._override is not None else None
        if forced is not None:
            return forced
        if args == ["-i"]:
            match = re.fullmatch(r"add-generic-password -U -s (\S+) -a (\S+) -l (\S+) -w (\S+)\n", stdin or "")
            if match is None:
                return 1, ""
            self.entries[f"{match.group(1)} {match.group(2)}"] = match.group(4)
            self.labels[f"{match.group(1)} {match.group(2)}"] = match.group(3)
            return 0, ""
        if len(args) == 6 and args[0] == "find-generic-password" and args[1] == "-s" and args[3] == "-a" and args[5] == "-w":
            value = self.entries.get(f"{args[2]} {args[4]}")
            return (44, "") if value is None else (0, f"{value}\n")
        if len(args) == 5 and args[0] == "delete-generic-password" and args[1] == "-s" and args[3] == "-a":
            if self.entries.pop(f"{args[2]} {args[4]}", None) is None:
                return 44, ""
            return 0, 'keychain: "login.keychain-db"\n'
        return 1, ""


class Sandbox:
    """The replaced surroundings of one test.

    Attributes:
        root: Temporary directory of the test.
        home: Stand-in home directory, inside ``root``.
        dir: Accounts directory the test passes explicitly. Not created.
        security: The in-memory Keychain tool.
        notes: Lines the keystore wrote to the terminal.
        prompts: Prompt texts shown, hidden and visible, in order.
        answers: Lines the stand-in person types, consumed in order.
        terminal: Whether a person can answer a prompt.
        output_terminal: Whether standard output is a terminal.
        real: The untouched facilities, for the opt-in real-Keychain test.
        real_home: The ``HOME`` the test started with, which the real
            ``security`` tool needs to find the login keychain.
    """

    def __init__(self, root: Path, real: KeystoreRuntime) -> None:
        self.root = root
        self.home = root / "home"
        self.dir = root / "accounts"
        self.security = FakeSecurity()
        self.notes: List[str] = []
        self.prompts: List[str] = []
        self.answers: List[str] = []
        self.terminal = False
        self.output_terminal = False
        self.real = real
        self.real_home = os.environ.get("HOME")

    def answer(self, message: str) -> str:
        self.prompts.append(message)
        if not self.answers:
            raise AssertionError("unexpected prompt")
        return self.answers.pop(0)

    def put(self, name: str, content: Any) -> Path:
        """Write a crafted account file into the test accounts directory."""
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{name}.json"
        text = content if isinstance(content, str) else json.dumps(content, indent=2) + "\n"
        path.write_bytes(text.encode("utf-8"))
        path.chmod(0o600)
        return path

    def copy_vectors(self) -> None:
        """Copy the shared sample files into the test accounts directory."""
        self.dir.mkdir(parents=True, exist_ok=True)
        for source in VECTORS_DIR.iterdir():
            target = self.dir / source.name
            target.write_bytes(source.read_bytes())
            target.chmod(0o600)


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Sandbox:
    for name in VARIABLES:
        monkeypatch.delenv(name, raising=False)
    real = KeystoreRuntime()
    for attribute in vars(real):
        setattr(real, attribute, getattr(runtime, attribute))
    box = Sandbox(tmp_path, real)
    box.home.mkdir()
    # The home directory is replaced twice over: in the keystore's own lookup
    # and in the environment any other code would consult.
    monkeypatch.setenv("HOME", str(box.home))
    monkeypatch.setenv("USERPROFILE", str(box.home))
    monkeypatch.setattr(runtime, "platform", "darwin")
    monkeypatch.setattr(runtime, "home", lambda: str(box.home))
    monkeypatch.setattr(runtime, "security", box.security.run)
    monkeypatch.setattr(runtime, "terminal_attached", lambda: box.terminal)
    monkeypatch.setattr(runtime, "output_is_terminal", lambda: box.output_terminal)
    monkeypatch.setattr(runtime, "prompt_hidden", box.answer)
    monkeypatch.setattr(runtime, "prompt_visible", box.answer)
    monkeypatch.setattr(runtime, "notify", box.notes.append)
    return box


def new_account() -> SimpleNamespace:
    """Generate a random account as an address and private-key pair of strings."""
    account = Account.random()
    return SimpleNamespace(address=str(account.address), private_key=str(account.private_key))


def failure(call: Callable[[], Any]) -> KeystoreError:
    """Return the keystore error a call fails with."""
    try:
        call()
    except KeystoreError as error:
        return error
    raise AssertionError("expected the call to fail")


def code_of(call: Callable[[], Any]) -> str:
    """Return the error code a call fails with."""
    return failure(call).code


def check_secret(actual: object, expected: object) -> None:
    """Compare two secrets without putting either into an assertion message."""
    if not isinstance(actual, str) or actual != expected:
        raise AssertionError("the secret values differ")


def absent(text: str, secret: str) -> None:
    """Assert a secret does not occur in some text, without printing either."""
    if secret in text:
        raise AssertionError("the text contains a secret")


def mode(path: Any) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)
