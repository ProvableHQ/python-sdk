"""Tests for the account commands (``python -m aleo.keystore``).

Each command runs in-process against a temporary accounts directory, with the
environment, the home directory, the Keychain tool and the terminal replaced
(see ``keystore_helpers``). Output is captured, so the keys that ``export``
and ``create --storage none`` print by design never reach the test log.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, List, NamedTuple

import pytest

import aleo
from aleo import keystore
from aleo.keystore._cli import main
from aleo.keystore._runtime import runtime
from aleo.mainnet import PrivateKey

from keystore_helpers import (  # noqa: F401  (sandbox is a fixture)
    OTHER_PASSWORD,
    PASSWORD,
    POSIX,
    SAMPLE,
    Sandbox,
    VARIABLES,
    VECTORS,
    absent,
    check_secret,
    key_entry,
    mode,
    new_account,
    sandbox,
    saved_password_entry,
)

pytestmark = pytest.mark.usefixtures("sandbox")

PHRASE = VECTORS["mnemonic"]["phrase"]


class Outcome(NamedTuple):
    """What one command left behind: exit status, standard output, standard error."""

    status: int
    out: str
    err: str

    @property
    def json(self) -> Any:
        return json.loads(self.out)


@pytest.fixture
def run(sandbox: Sandbox, capsys: pytest.CaptureFixture[str]):
    """Run one command against the test accounts directory."""

    def invoke(*argv: str, with_dir: bool = True) -> Outcome:
        sandbox.notes.clear()
        arguments: List[str] = list(argv)
        if with_dir:
            arguments += ["--dir", str(sandbox.dir)]
        try:
            status = main(arguments)
        except SystemExit as exit_request:
            status = exit_request.code if isinstance(exit_request.code, int) else 1
        captured = capsys.readouterr()
        # Notes and errors go through the replaced terminal; usage text goes
        # straight to standard error.
        return Outcome(status, captured.out, captured.err + "".join(f"{note}\n" for note in sandbox.notes))

    return invoke


def stored_key(sandbox: Sandbox, name: str = "default", password: Any = None) -> str:
    return keystore.export(dir=sandbox.dir, name=name, password=password, prompt=False)


# ---------------------------------------------------------------------------
# Usage errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param([], id="no command"),
        pytest.param(["destroy"], id="an unknown command"),
        pytest.param(["show", "--verbose"], id="an unknown flag"),
        pytest.param(["show", "--nam", "x"], id="an abbreviated flag"),
        pytest.param(["show", "extra"], id="a stray argument"),
        pytest.param(["create", "--storage", "vault"], id="an unknown storage"),
        pytest.param(["create", "--password", "x"], id="a password on the command line"),
        pytest.param(["protect"], id="protect without a storage"),
        pytest.param(["protect", "--storage", "none"], id="protect to none"),
        pytest.param(["protect", "--save-password"], id="protect saving a password without a storage"),
        pytest.param(["protect", "--storage", "encrypted", "--save-password", "--forget-password"], id="save and forget"),
        pytest.param(["migrate"], id="migrate without --from"),
        pytest.param(["migrate", "--from", "x.json", "--storage", "none"], id="migrate to none"),
        pytest.param(["import", "--key-file", "a", "--mnemonic-file", "b"], id="a key file and a phrase file"),
        pytest.param(["import", "--index", "-1"], id="a negative index"),
        pytest.param(["import", "--index", "2147483648"], id="an index beyond 2^31 - 1"),
        pytest.param(["import", "--index", "one"], id="an index that is not a number"),
        pytest.param(["import", "--derivation", "bip44"], id="an unknown derivation"),
        pytest.param(["show", "--forget-password"], id="a flag of another command"),
        pytest.param(["export", "--storage", "plaintext"], id="a storage on export"),
        pytest.param(["create", "--name"], id="a flag without its value"),
    ],
)
def test_usage_errors_exit_64(sandbox: Sandbox, run: Any, argv: List[str]) -> None:
    outcome = run(*argv)
    assert outcome.status == 64
    assert outcome.out == ""
    assert "usage:" in outcome.err
    assert not sandbox.dir.exists()


def test_help_exits_0(run: Any) -> None:
    outcome = run("--help", with_dir=False)
    assert outcome.status == 0
    for command in ("create", "import", "show", "protect", "export", "migrate"):
        assert command in outcome.out
    assert run("import", "--help", with_dir=False).status == 0


def test_common_flags_work_before_and_after_the_command(sandbox: Sandbox, run: Any) -> None:
    account = new_account()
    keystore.save(account.private_key, dir=sandbox.dir, name="named")
    after = run("show", "--name", "named", "--json")
    before = run("--json", "--name", "named", "--dir", str(sandbox.dir), "show", with_dir=False)
    assert after.status == before.status == 0
    assert after.json == before.json
    assert after.json["name"] == "named"


# ---------------------------------------------------------------------------
# Output and exit status
# ---------------------------------------------------------------------------


def test_show_prints_the_account_and_never_the_key(sandbox: Sandbox, run: Any) -> None:
    account = new_account()
    stored = keystore.save(account.private_key, dir=sandbox.dir)

    outcome = run("show", "--json")
    assert outcome.status == 0
    assert outcome.err == ""
    assert outcome.out.count("\n") == 1
    assert outcome.json == {
        "name": "default",
        "address": account.address,
        "storage": "plaintext",
        "path": str(stored.path),
    }
    assert list(outcome.json) == ["name", "address", "storage", "path"]

    plain = run("show")
    assert plain.status == 0
    for value in ("default", account.address, "plaintext", str(stored.path)):
        assert value in plain.out
    absent(plain.out + plain.err + outcome.out, account.private_key)


def test_show_reads_only_the_account_file(sandbox: Sandbox, run: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    account = new_account()
    keystore.save(account.private_key, dir=sandbox.dir, storage="encrypted", password=PASSWORD)
    monkeypatch.setenv("ALEO_PRIVATE_KEY", "")
    outcome = run("show", "--json")
    assert outcome.status == 0
    assert outcome.json["storage"] == "encrypted"
    assert sandbox.security.calls == []
    assert sandbox.prompts == []


def test_errors_go_to_standard_error_as_code_and_message(sandbox: Sandbox, run: Any) -> None:
    outcome = run("show")
    assert outcome.status == 2
    assert outcome.out == ""
    assert outcome.err.startswith("ACCOUNT_NOT_FOUND: ")
    assert str(sandbox.dir / "default.json") in outcome.err


def test_errors_go_to_standard_output_as_json_with_the_flag(sandbox: Sandbox, run: Any) -> None:
    outcome = run("show", "--json")
    assert outcome.status == 2
    assert outcome.err == ""
    assert list(outcome.json) == ["error"]
    assert list(outcome.json["error"]) == ["code", "message"]
    assert outcome.json["error"]["code"] == "ACCOUNT_NOT_FOUND"
    assert str(sandbox.dir / "default.json") in outcome.json["error"]["message"]


def test_needs_a_person_exits_2_and_other_failures_exit_1(sandbox: Sandbox, run: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    account = new_account()
    # ACCOUNT_NOT_FOUND
    assert run("show").status == 2
    # KEY_NOT_STORED
    keystore.save(account.private_key, dir=sandbox.dir, name="address-only", storage="none")
    outcome = run("protect", "--storage", "plaintext", "--name", "address-only", "--json")
    assert (outcome.status, outcome.json["error"]["code"]) == (2, "KEY_NOT_STORED")
    # PASSWORD_REQUIRED
    outcome = run("create", "--storage", "encrypted", "--json")
    assert (outcome.status, outcome.json["error"]["code"]) == (2, "PASSWORD_REQUIRED")
    # TERMINAL_REQUIRED
    outcome = run("export", "--name", "address-only")
    assert outcome.status == 2 and outcome.err.startswith("TERMINAL_REQUIRED: ")
    # Everything else is a plain failure.
    keystore.save(account.private_key, dir=sandbox.dir)
    outcome = run("create", "--json")
    assert (outcome.status, outcome.json["error"]["code"]) == (1, "ACCOUNT_EXISTS")
    outcome = run("show", "--name", "Not_Valid", "--json")
    assert (outcome.status, outcome.json["error"]["code"]) == (1, "INVALID_NAME")
    monkeypatch.setenv("ALEO_ACCOUNT_PASSWORD", PASSWORD)
    keystore.save(account.private_key, dir=sandbox.dir, name="locked", storage="encrypted")
    monkeypatch.setenv("ALEO_ACCOUNT_PASSWORD", OTHER_PASSWORD)
    outcome = run("protect", "--storage", "plaintext", "--name", "locked", "--json")
    assert (outcome.status, outcome.json["error"]["code"]) == (1, "WRONG_PASSWORD")
    absent(outcome.out + outcome.err, OTHER_PASSWORD)


def test_a_file_system_failure_exits_1(sandbox: Sandbox, run: Any) -> None:
    sandbox.dir.mkdir()
    (sandbox.dir / "default.json").mkdir()
    outcome = run("show")
    assert outcome.status == 1
    assert outcome.out == ""
    assert outcome.err.startswith("aleo.keystore: ")


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


def test_create_generates_a_key_and_stores_it_in_plaintext_by_default(sandbox: Sandbox, run: Any) -> None:
    outcome = run("create", "--json")
    assert outcome.status == 0
    assert list(outcome.json) == ["name", "address", "storage", "path"]
    assert outcome.json["name"] == "default"
    assert outcome.json["storage"] == "plaintext"
    assert outcome.json["path"] == str(sandbox.dir / "default.json")
    key = stored_key(sandbox)
    assert str(PrivateKey.from_string(key).address) == outcome.json["address"]
    absent(outcome.out + outcome.err, key)
    if POSIX:
        assert mode(sandbox.dir / "default.json") == 0o600
        assert mode(sandbox.dir) == 0o700

    # A second account under another name gets another key.
    other = run("create", "--name", "second")
    assert other.status == 0
    assert "second" in other.out
    assert stored_key(sandbox, "second") != key
    absent(other.out + other.err, stored_key(sandbox, "second"))


def test_create_refuses_to_replace_an_existing_account(sandbox: Sandbox, run: Any) -> None:
    assert run("create").status == 0
    data = (sandbox.dir / "default.json").read_bytes()
    outcome = run("create")
    assert outcome.status == 1
    assert outcome.err.startswith("ACCOUNT_EXISTS: ")
    assert (sandbox.dir / "default.json").read_bytes() == data


def test_create_takes_the_name_and_directory_from_the_variables(sandbox: Sandbox, run: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALEO_ACCOUNT", "from-env")
    monkeypatch.setenv("ALEO_ACCOUNTS_DIR", str(sandbox.root / "env-dir"))
    outcome = run("create", "--json", with_dir=False)
    assert outcome.status == 0
    assert outcome.json["path"] == str(sandbox.root / "env-dir" / "from-env.json")


def test_create_without_a_stored_key_needs_a_terminal_and_shows_the_key_once(sandbox: Sandbox, run: Any) -> None:
    # No terminal: nothing is generated or written.
    outcome = run("create", "--storage", "none")
    assert outcome.status == 2
    assert outcome.err.startswith("TERMINAL_REQUIRED: ")
    assert "APrivateKey1" not in outcome.out + outcome.err
    assert not sandbox.dir.exists()

    # Standard output redirected: the key would land in a pipe or a file.
    sandbox.terminal = True
    assert run("create", "--storage", "none").status == 2
    assert not sandbox.dir.exists()

    # A key never appears with --json, even at a terminal.
    sandbox.output_terminal = True
    outcome = run("create", "--storage", "none", "--json")
    assert outcome.status == 2
    assert outcome.json["error"]["code"] == "TERMINAL_REQUIRED"
    assert "APrivateKey1" not in outcome.out + outcome.err
    assert not sandbox.dir.exists()

    outcome = run("create", "--storage", "none")
    assert outcome.status == 0
    shown = [word for word in outcome.out.split() if word.startswith("APrivateKey1")]
    assert len(shown) == 1
    assert "APrivateKey1" not in outcome.err
    stored = keystore.info(dir=sandbox.dir)
    assert stored is not None and stored.storage == "none"
    address = str(PrivateKey.from_string(shown[0]).address)
    if address != stored.address:
        raise AssertionError("the key shown is not the key of the stored address")
    absent((sandbox.dir / "default.json").read_text(encoding="utf-8"), shown[0])
    assert "ALEO_PRIVATE_KEY" in outcome.err


def _phrase_shown(out: str) -> str:
    """The recovery phrase a command printed, without its label."""
    lines = [line for line in out.splitlines() if line.startswith("Recovery phrase: ")]
    assert len(lines) == 1
    return lines[0][len("Recovery phrase: "):]


def test_create_from_a_new_phrase_needs_a_terminal_and_shows_the_phrase_once(sandbox: Sandbox, run: Any) -> None:
    # No terminal: nothing is generated or written.
    outcome = run("create", "--mnemonic")
    assert outcome.status == 2
    assert outcome.err.startswith("TERMINAL_REQUIRED: ")
    assert "recovery phrase" in outcome.err
    assert "Recovery phrase:" not in outcome.out + outcome.err
    assert not sandbox.dir.exists()

    # Standard output redirected: the phrase would land in a pipe or a file.
    sandbox.terminal = True
    assert run("create", "--mnemonic").status == 2
    assert not sandbox.dir.exists()

    # A phrase never appears with --json, even at a terminal.
    sandbox.output_terminal = True
    outcome = run("create", "--mnemonic", "--json")
    assert outcome.status == 2
    assert outcome.json["error"]["code"] == "TERMINAL_REQUIRED"
    assert not sandbox.dir.exists()

    outcome = run("create", "--mnemonic")
    assert outcome.status == 0
    phrase = _phrase_shown(outcome.out)
    assert len(phrase.split(" ")) == 12
    assert "Recovery phrase:" not in outcome.err
    assert "stored nowhere" in outcome.err
    # No key is printed: the account is stored, and the phrase is the backup.
    assert "APrivateKey1" not in outcome.out + outcome.err

    # The stored account is the first account of the phrase, as a wallet derives it.
    stored = keystore.info(dir=sandbox.dir)
    assert stored is not None and stored.storage == "plaintext"
    client = aleo.Aleo(aleo.HTTPProvider("http://127.0.0.1:1"))
    derived = client.account.from_mnemonic(phrase)
    if str(derived.address) != stored.address:
        raise AssertionError("the phrase shown does not derive the stored address")
    if str(derived.private_key) != stored_key(sandbox):
        raise AssertionError("the phrase shown does not derive the stored key")
    # The phrase is in no file.
    for path in sandbox.dir.iterdir():
        absent(path.read_text(encoding="utf-8"), phrase)
        for word in phrase.split(" ")[:3]:
            assert f" {word} " not in path.read_text(encoding="utf-8")


def test_create_from_a_new_phrase_makes_a_different_account_each_time(sandbox: Sandbox, run: Any) -> None:
    sandbox.terminal = sandbox.output_terminal = True
    first = run("create", "--mnemonic", "--name", "one")
    second = run("create", "--mnemonic", "--name", "two")
    assert first.status == 0 and second.status == 0
    assert _phrase_shown(first.out) != _phrase_shown(second.out)
    one, two = keystore.info(dir=sandbox.dir, name="one"), keystore.info(dir=sandbox.dir, name="two")
    assert one is not None and two is not None and one.address != two.address


def test_create_from_a_new_phrase_shows_nothing_when_the_account_cannot_be_stored(sandbox: Sandbox, run: Any) -> None:
    sandbox.terminal = sandbox.output_terminal = True
    assert run("create").status == 0
    before = (sandbox.dir / "default.json").read_bytes()
    outcome = run("create", "--mnemonic")
    assert outcome.status == 1
    assert outcome.err.startswith("ACCOUNT_EXISTS: ")
    assert "Recovery phrase:" not in outcome.out + outcome.err
    assert (sandbox.dir / "default.json").read_bytes() == before


def test_create_from_a_new_phrase_in_the_keychain_names_the_phrase_as_the_backup(sandbox: Sandbox, run: Any) -> None:
    sandbox.terminal = sandbox.output_terminal = True
    outcome = run("create", "--mnemonic", "--storage", "keychain", "--name", "wallet")
    assert outcome.status == 0
    phrase = _phrase_shown(outcome.out)
    stored = keystore.info(dir=sandbox.dir, name="wallet")
    assert stored is not None and stored.storage == "keychain"
    key = sandbox.security.entries[key_entry(stored.path)]
    client = aleo.Aleo(aleo.HTTPProvider("http://127.0.0.1:1"))
    if str(client.account.from_mnemonic(phrase).private_key) != key:
        raise AssertionError("the phrase shown does not derive the key in the Keychain")
    absent(outcome.out + outcome.err, key)
    # The export hint is for accounts with no phrase.
    assert "export" not in outcome.err


def test_create_from_a_new_phrase_with_no_stored_key_shows_the_phrase_and_the_key(sandbox: Sandbox, run: Any) -> None:
    sandbox.terminal = sandbox.output_terminal = True
    outcome = run("create", "--mnemonic", "--storage", "none")
    assert outcome.status == 0
    phrase = _phrase_shown(outcome.out)
    shown = [word for word in outcome.out.split() if word.startswith("APrivateKey1")]
    assert len(shown) == 1
    client = aleo.Aleo(aleo.HTTPProvider("http://127.0.0.1:1"))
    if str(client.account.from_mnemonic(phrase).private_key) != shown[0]:
        raise AssertionError("the key shown is not the key of the phrase shown")
    stored = keystore.info(dir=sandbox.dir)
    assert stored is not None and stored.storage == "none"


def test_create_in_the_keychain_says_the_key_has_no_other_copy(sandbox: Sandbox, run: Any) -> None:
    outcome = run("create", "--storage", "keychain", "--json")
    assert outcome.status == 0
    assert outcome.json["storage"] == "keychain"
    key = sandbox.security.entries[key_entry(outcome.json["path"])]
    absent(outcome.out + outcome.err, key)
    assert "APrivateKey1" not in (sandbox.dir / "default.json").read_text(encoding="utf-8")
    assert "export" in outcome.err
    assert "Keychain" in outcome.err


def test_create_in_the_keychain_fails_outside_macos_before_anything_is_written(
    sandbox: Sandbox, run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime, "platform", "linux")
    outcome = run("create", "--storage", "keychain")
    assert outcome.status == 1
    assert outcome.err.startswith("KEYCHAIN_UNAVAILABLE: ")
    assert not sandbox.dir.exists()


def test_create_encrypted_takes_the_password_from_the_variable_or_the_prompt(
    sandbox: Sandbox, run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No password source and no terminal: needs a person.
    assert run("create", "--storage", "encrypted").status == 2
    assert not sandbox.dir.exists()

    monkeypatch.setenv("ALEO_ACCOUNT_PASSWORD", PASSWORD)
    outcome = run("create", "--storage", "encrypted", "--json")
    assert outcome.status == 0
    assert outcome.json["storage"] == "encrypted"
    assert sandbox.security.calls == []
    monkeypatch.delenv("ALEO_ACCOUNT_PASSWORD")
    key = stored_key(sandbox, password=PASSWORD)
    absent(outcome.out + outcome.err, key)
    absent(outcome.out + outcome.err, PASSWORD)

    # At a terminal the password is asked for twice.
    sandbox.terminal = True
    sandbox.answers.extend([OTHER_PASSWORD, OTHER_PASSWORD])
    outcome = run("create", "--storage", "encrypted", "--name", "typed")
    assert outcome.status == 0
    assert len(sandbox.prompts) == 2
    assert str(PrivateKey.from_string(stored_key(sandbox, "typed", OTHER_PASSWORD)).address) in outcome.out


def test_create_can_save_the_password_in_the_keychain(sandbox: Sandbox, run: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALEO_ACCOUNT_PASSWORD", PASSWORD)
    outcome = run("create", "--storage", "encrypted", "--save-password", "--json")
    assert outcome.status == 0
    assert sandbox.security.entries == {saved_password_entry(outcome.json["path"]): PASSWORD.encode("utf-8").hex()}
    assert "--save-password" not in outcome.err

    # Where it cannot apply, the flag is reported as having had no effect.
    plain = run("create", "--name", "plain", "--save-password")
    assert plain.status == 0
    assert "--save-password" in plain.err
    monkeypatch.setattr(runtime, "platform", "linux")
    elsewhere = run("create", "--name", "elsewhere", "--storage", "encrypted", "--save-password")
    assert elsewhere.status == 0
    assert "--save-password" in elsewhere.err
    assert len(sandbox.security.entries) == 1


def test_storing_plaintext_on_windows_says_file_modes_are_not_enforced(
    sandbox: Sandbox, run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime, "platform", "win32")
    outcome = run("create")
    assert outcome.status == 0
    assert "Windows" in outcome.err
    monkeypatch.setenv("ALEO_ACCOUNT_PASSWORD", PASSWORD)
    encrypted = run("create", "--name", "locked", "--storage", "encrypted")
    assert encrypted.status == 0
    assert "Windows" not in encrypted.err


# ---------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------


def test_import_stores_the_key_of_a_key_file(sandbox: Sandbox, run: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    account = new_account()
    key_file = sandbox.root / "key.txt"
    key_file.write_text(f"\n  {account.private_key}\n", encoding="utf-8")
    # The flag comes first: the variables are not consulted.
    monkeypatch.setenv("ALEO_PRIVATE_KEY", "")
    outcome = run("import", "--key-file", str(key_file), "--json")
    assert outcome.status == 0
    assert outcome.json == {
        "name": "default",
        "address": account.address,
        "storage": "plaintext",
        "path": str(sandbox.dir / "default.json"),
    }
    check_secret(stored_key(sandbox), account.private_key)
    absent(outcome.out + outcome.err, account.private_key)

    # Importing the same key again changes nothing and succeeds.
    assert run("import", "--key-file", str(key_file)).status == 0
    # Another key under the same name is refused.
    key_file.write_text(new_account().private_key, encoding="utf-8")
    again = run("import", "--key-file", str(key_file), "--json")
    assert (again.status, again.json["error"]["code"]) == (1, "ACCOUNT_EXISTS")


@pytest.mark.parametrize(
    ("content", "code"),
    [(None, "EMPTY_KEY_SOURCE"), (" \n", "EMPTY_KEY_SOURCE"), ("APrivateKey1nope", "INVALID_KEY")],
)
def test_import_reports_an_unusable_key_file(sandbox: Sandbox, run: Any, content: Any, code: str) -> None:
    key_file = sandbox.root / "key.txt"
    if content is not None:
        key_file.write_text(content, encoding="utf-8")
    for flag in ("--key-file", "--mnemonic-file"):
        outcome = run("import", flag, str(key_file), "--json")
        expected = "INVALID_MNEMONIC" if (flag, code) == ("--mnemonic-file", "INVALID_KEY") else code
        assert (outcome.status, outcome.json["error"]["code"]) == (1, expected)
        assert "nope" not in outcome.out
    assert not sandbox.dir.exists()


def test_import_stores_a_key_supplied_through_the_variables(sandbox: Sandbox, run: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    account = new_account()
    monkeypatch.setenv("SHIELD_SWAP_PRIVATE_KEY", account.private_key)
    outcome = run("import", "--storage", "keychain", "--json")
    assert outcome.status == 0
    assert outcome.json["address"] == account.address
    check_secret(sandbox.security.entries[key_entry(outcome.json["path"])], account.private_key)
    absent(outcome.out + outcome.err, account.private_key)

    key_file = sandbox.root / "key.txt"
    key_file.write_text(account.private_key, encoding="utf-8")
    monkeypatch.delenv("SHIELD_SWAP_PRIVATE_KEY")
    monkeypatch.setenv("ALEO_PRIVATE_KEY_FILE", str(key_file))
    from_file = run("import", "--name", "from-file", "--json")
    assert from_file.status == 0
    check_secret(stored_key(sandbox, "from-file"), account.private_key)


def test_import_applies_the_rules_of_the_key_variables(sandbox: Sandbox, run: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALEO_PRIVATE_KEY", "")
    outcome = run("import", "--json")
    assert (outcome.status, outcome.json["error"]["code"]) == (1, "EMPTY_KEY_SOURCE")
    monkeypatch.setenv("ALEO_PRIVATE_KEY", new_account().private_key)
    monkeypatch.setenv("SHIELD_SWAP_PRIVATE_KEY", new_account().private_key)
    outcome = run("import", "--json")
    assert (outcome.status, outcome.json["error"]["code"]) == (1, "KEY_CONFLICT")
    assert not sandbox.dir.exists()


@pytest.mark.parametrize("expected", VECTORS["mnemonic"]["accounts"], ids=lambda entry: f"{entry['derivation']}-{entry['index']}")
def test_import_stores_the_derived_key_of_a_recovery_phrase_file_and_not_the_phrase(
    sandbox: Sandbox, run: Any, expected: Any
) -> None:
    phrase_file = sandbox.root / "phrase.txt"
    # Pasted phrases arrive with stray white space and capitals.
    phrase_file.write_text("  " + PHRASE.upper().replace(" ", " \n\t ") + "\n\n", encoding="utf-8")
    outcome = run(
        "import", "--mnemonic-file", str(phrase_file),
        "--index", str(expected["index"]), "--derivation", expected["derivation"], "--json",
    )
    assert outcome.status == 0
    assert outcome.json["address"] == expected["address"]
    check_secret(stored_key(sandbox), expected["privateKey"])
    text = (sandbox.dir / "default.json").read_text(encoding="utf-8")
    assert "abandon" not in text and "about" not in text
    assert "abandon" not in outcome.out + outcome.err


def test_import_defaults_to_the_first_account_on_the_standard_path(sandbox: Sandbox, run: Any) -> None:
    phrase_file = sandbox.root / "phrase.txt"
    phrase_file.write_text(PHRASE, encoding="utf-8")
    outcome = run("import", "--mnemonic-file", str(phrase_file), "--json")
    first = next(a for a in VECTORS["mnemonic"]["accounts"] if a["derivation"] == "standard" and a["index"] == 0)
    assert outcome.json["address"] == first["address"]


def test_import_rejects_an_invalid_recovery_phrase(sandbox: Sandbox, run: Any) -> None:
    phrase_file = sandbox.root / "phrase.txt"
    words = PHRASE.split()
    for phrase in (
        " ".join(words[:-1] + ["abandon"]),  # checksum
        " ".join(words[:-1]),  # word count
        " ".join(words[:-1] + ["zzzzzz"]),  # word list
    ):
        phrase_file.write_text(phrase, encoding="utf-8")
        outcome = run("import", "--mnemonic-file", str(phrase_file), "--json")
        assert (outcome.status, outcome.json["error"]["code"]) == (1, "INVALID_MNEMONIC")
        assert "abandon" not in outcome.out and "zzzzzz" not in outcome.out
    assert not sandbox.dir.exists()


def test_import_asks_at_a_hidden_prompt_when_nothing_else_supplies_a_key(sandbox: Sandbox, run: Any) -> None:
    # No source and no terminal: needs a person.
    outcome = run("import", "--json")
    assert (outcome.status, outcome.json["error"]["code"]) == (2, "ACCOUNT_NOT_FOUND")
    assert sandbox.prompts == []

    sandbox.terminal = True
    # Text that starts with APrivateKey1 is a key.
    account = new_account()
    sandbox.answers.append(f" {account.private_key}\n")
    outcome = run("import", "--json")
    assert outcome.status == 0
    assert outcome.json["address"] == account.address
    assert len(sandbox.prompts) == 1
    absent(outcome.out + outcome.err, account.private_key)

    # Anything else is a recovery phrase, tidied before it is checked.
    second = next(a for a in VECTORS["mnemonic"]["accounts"] if a["derivation"] == "legacy" and a["index"] == 1)
    sandbox.answers.append("  " + PHRASE.title().replace(" ", "   ") + " ")
    outcome = run("import", "--name", "phrase", "--index", "1", "--derivation", "legacy", "--json")
    assert outcome.status == 0
    assert outcome.json["address"] == second["address"]
    check_secret(stored_key(sandbox, "phrase"), second["privateKey"])

    sandbox.answers.append("APrivateKey1nope")
    outcome = run("import", "--name", "bad-key", "--json")
    assert (outcome.status, outcome.json["error"]["code"]) == (1, "INVALID_KEY")
    sandbox.answers.append("not a recovery phrase")
    outcome = run("import", "--name", "bad-phrase", "--json")
    assert (outcome.status, outcome.json["error"]["code"]) == (1, "INVALID_MNEMONIC")
    sandbox.answers.append("   ")
    outcome = run("import", "--name", "nothing", "--json")
    assert (outcome.status, outcome.json["error"]["code"]) == (1, "EMPTY_KEY_SOURCE")


def test_import_can_record_the_address_only(sandbox: Sandbox, run: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    account = new_account()
    monkeypatch.setenv("ALEO_PRIVATE_KEY", account.private_key)
    outcome = run("import", "--storage", "none", "--json")
    assert outcome.status == 0
    assert outcome.json["storage"] == "none"
    absent((sandbox.dir / "default.json").read_text(encoding="utf-8"), account.private_key)
    absent(outcome.out + outcome.err, account.private_key)


# ---------------------------------------------------------------------------
# protect
# ---------------------------------------------------------------------------


def test_protect_changes_the_storage(sandbox: Sandbox, run: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    account = new_account()
    keystore.save(account.private_key, dir=sandbox.dir)

    outcome = run("protect", "--storage", "keychain", "--json")
    assert outcome.status == 0
    assert outcome.json == {
        "name": "default",
        "address": account.address,
        "storage": "keychain",
        "path": str(sandbox.dir / "default.json"),
    }

    monkeypatch.setenv("ALEO_ACCOUNT_PASSWORD", PASSWORD)
    outcome = run("protect", "--storage", "encrypted", "--save-password", "--json")
    assert outcome.status == 0
    assert outcome.json["storage"] == "encrypted"
    assert sandbox.security.entries == {saved_password_entry(outcome.json["path"]): PASSWORD.encode("utf-8").hex()}

    outcome = run("protect", "--storage", "plaintext")
    assert outcome.status == 0
    assert sandbox.security.entries == {}
    monkeypatch.delenv("ALEO_ACCOUNT_PASSWORD")
    check_secret(stored_key(sandbox), account.private_key)
    absent(outcome.out + outcome.err, account.private_key)


def test_protect_sets_a_new_password_on_an_encrypted_account_from_the_prompt_only(
    sandbox: Sandbox, run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    account = new_account()
    keystore.save(account.private_key, dir=sandbox.dir, storage="encrypted", password=PASSWORD)
    before = (sandbox.dir / "default.json").read_bytes()
    monkeypatch.setenv("ALEO_ACCOUNT_PASSWORD", PASSWORD)

    outcome = run("protect", "--storage", "encrypted", "--json")
    assert (outcome.status, outcome.json["error"]["code"]) == (2, "PASSWORD_REQUIRED")
    assert (sandbox.dir / "default.json").read_bytes() == before

    sandbox.terminal = True
    sandbox.answers.extend([OTHER_PASSWORD, OTHER_PASSWORD])
    assert run("protect", "--storage", "encrypted").status == 0
    assert (sandbox.dir / "default.json").read_bytes() != before
    monkeypatch.delenv("ALEO_ACCOUNT_PASSWORD")
    check_secret(stored_key(sandbox, password=OTHER_PASSWORD), account.private_key)


def test_protect_can_forget_a_saved_password_and_leave_the_account_as_it_is(sandbox: Sandbox, run: Any) -> None:
    account = new_account()
    keystore.save(account.private_key, dir=sandbox.dir, storage="encrypted", password=PASSWORD, save_password=True)
    before = (sandbox.dir / "default.json").read_bytes()
    assert sandbox.security.entries != {}

    outcome = run("protect", "--forget-password", "--json")
    assert outcome.status == 0
    assert outcome.json["storage"] == "encrypted"
    assert sandbox.security.entries == {}
    assert (sandbox.dir / "default.json").read_bytes() == before
    assert sandbox.prompts == []
    # The account still opens with its password; only the saved copy is gone.
    check_secret(stored_key(sandbox, password=PASSWORD), account.private_key)
    # Forgetting again is not an error.
    assert run("protect", "--forget-password").status == 0

    assert run("protect", "--forget-password", "--name", "absent").status == 2


def test_protect_forget_password_with_a_storage_change_removes_the_saved_password(
    sandbox: Sandbox, run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    account = new_account()
    keystore.save(account.private_key, dir=sandbox.dir, storage="encrypted", password=PASSWORD, save_password=True)
    monkeypatch.setenv("ALEO_ACCOUNT_PASSWORD", PASSWORD)
    outcome = run("protect", "--storage", "keychain", "--forget-password", "--json")
    assert outcome.status == 0
    assert list(sandbox.security.entries) == [key_entry(outcome.json["path"])]


def test_protect_forget_password_fails_outside_macos(sandbox: Sandbox, run: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    keystore.save(new_account().private_key, dir=sandbox.dir, storage="encrypted", password=PASSWORD)
    monkeypatch.setattr(runtime, "platform", "linux")
    outcome = run("protect", "--forget-password", "--json")
    assert (outcome.status, outcome.json["error"]["code"]) == (1, "KEYCHAIN_UNAVAILABLE")
    assert sandbox.security.calls == []


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def test_export_prints_the_key_only_for_a_person_who_confirms(sandbox: Sandbox, run: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    account = new_account()
    keystore.save(account.private_key, dir=sandbox.dir)
    # A key in the environment is never what export prints.
    monkeypatch.setenv("ALEO_PRIVATE_KEY", new_account().private_key)

    # No terminal: refused before anything is read or asked.
    outcome = run("export")
    assert outcome.status == 2
    assert outcome.err.startswith("TERMINAL_REQUIRED: ")
    assert outcome.out == ""
    sandbox.terminal = True
    assert run("export").status == 2
    sandbox.output_terminal = True
    outcome = run("export", "--json")
    assert (outcome.status, outcome.json["error"]["code"]) == (2, "TERMINAL_REQUIRED")
    assert sandbox.prompts == []

    # Declined, or answered with anything but yes: nothing is shown.
    for answer in ("n", "", "maybe"):
        sandbox.answers.append(answer)
        outcome = run("export")
        assert outcome.status == 1
        assert outcome.out == ""
        absent(outcome.err, account.private_key)

    sandbox.prompts.clear()
    sandbox.answers.append("y")
    outcome = run("export")
    assert outcome.status == 0
    check_secret(outcome.out, f"{account.private_key}\n")
    absent(outcome.err, account.private_key)
    assert len(sandbox.prompts) == 1
    assert account.address in sandbox.prompts[0] and "default" in sandbox.prompts[0]

    sandbox.answers.append("YES")
    assert run("export").status == 0


def test_export_reports_a_missing_account_or_key_before_asking(sandbox: Sandbox, run: Any) -> None:
    sandbox.terminal = sandbox.output_terminal = True
    outcome = run("export")
    assert outcome.status == 2
    assert outcome.err.startswith("ACCOUNT_NOT_FOUND: ")

    account = new_account()
    keystore.save(account.private_key, dir=sandbox.dir, storage="none")
    sandbox.answers.append("y")
    outcome = run("export")
    assert outcome.status == 2
    assert outcome.err.startswith("KEY_NOT_STORED: ")
    assert outcome.out == ""


def test_export_unlocks_an_encrypted_account_at_the_prompt(sandbox: Sandbox, run: Any) -> None:
    account = new_account()
    keystore.save(account.private_key, dir=sandbox.dir, storage="encrypted", password=PASSWORD)
    sandbox.terminal = sandbox.output_terminal = True
    sandbox.answers.extend(["y", OTHER_PASSWORD, PASSWORD])
    outcome = run("export")
    assert outcome.status == 0
    check_secret(outcome.out, f"{account.private_key}\n")
    assert "Wrong password." in outcome.err
    absent(outcome.out + outcome.err, PASSWORD)


# ---------------------------------------------------------------------------
# migrate
# ---------------------------------------------------------------------------


def test_migrate_copies_a_legacy_key_and_reports_the_copy_that_remains(sandbox: Sandbox, run: Any) -> None:
    account = new_account()
    source = sandbox.root / "profile.json"
    text = json.dumps({"address": account.address, "private_key": account.private_key, "network": "testnet"}, indent=2) + "\n"
    source.write_text(text, encoding="utf-8")

    outcome = run("migrate", "--from", str(source), "--json")
    assert outcome.status == 0
    assert outcome.json == {
        "name": "default",
        "address": account.address,
        "storage": "plaintext",
        "path": str(sandbox.dir / "default.json"),
    }
    assert source.read_text(encoding="utf-8") == text
    assert str(source) in outcome.err
    assert "--remove-legacy-key" in outcome.err
    absent(outcome.out + outcome.err, account.private_key)
    check_secret(stored_key(sandbox), account.private_key)


def test_migrate_can_strip_the_legacy_key(sandbox: Sandbox, run: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    account = new_account()
    source = sandbox.root / "state.json"
    source.write_text(
        json.dumps({"privateKey": account.private_key, "address": account.address, "apiUrl": "https://example.test"}, indent=2) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ALEO_ACCOUNT_PASSWORD", PASSWORD)
    outcome = run("migrate", "--from", str(source), "--storage", "encrypted", "--remove-legacy-key", "--json")
    assert outcome.status == 0
    assert outcome.json["storage"] == "encrypted"
    assert json.loads(source.read_text(encoding="utf-8")) == {"address": account.address, "apiUrl": "https://example.test"}
    assert "--remove-legacy-key" not in outcome.err
    if POSIX:
        assert mode(source) == 0o600
    check_secret(stored_key(sandbox, password=PASSWORD), account.private_key)

    again = run("migrate", "--from", str(source), "--json")
    assert (again.status, again.json["error"]["code"]) == (1, "LEGACY_KEY_NOT_FOUND")


# ---------------------------------------------------------------------------
# The module entry point
# ---------------------------------------------------------------------------


def test_runs_as_a_module_with_the_documented_exit_status(sandbox: Sandbox) -> None:
    account = new_account()
    stored = keystore.save(account.private_key, dir=sandbox.dir)
    environment = {name: value for name, value in os.environ.items() if name not in VARIABLES}
    environment["PYTHONPATH"] = str(Path(aleo.__file__).resolve().parents[1])
    environment["HOME"] = str(sandbox.home)

    def module(*argv: str) -> "subprocess.CompletedProcess[str]":
        return subprocess.run(
            [sys.executable, "-m", "aleo.keystore", *argv],
            env=environment, cwd=str(sandbox.root), capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120,
        )

    shown = module("show", "--dir", str(sandbox.dir), "--json")
    assert shown.returncode == 0
    assert shown.stderr == ""
    assert json.loads(shown.stdout) == {
        "name": "default",
        "address": account.address,
        "storage": "plaintext",
        "path": str(stored.path),
    }

    assert module("show", "--dir", str(sandbox.root / "empty")).returncode == 2
    # Not a terminal: the key is not printed.
    exported = module("export", "--dir", str(sandbox.dir))
    assert exported.returncode == 2
    assert exported.stderr.startswith("TERMINAL_REQUIRED: ")
    assert exported.stdout == ""
    usage = module("show", "--unknown")
    assert usage.returncode == 64
    assert "usage:" in usage.stderr
    assert module().returncode == 64
