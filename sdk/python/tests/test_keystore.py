"""Tests for the account keystore (``aleo.keystore``).

Covers the account file bytes, the lookup order, the password sources, the
Keychain commands and the storage changes. Every test runs in a temporary
directory with the environment, the home directory, the Keychain tool and the
terminal replaced (see ``keystore_helpers``). One opt-in test at the end uses
the real macOS Keychain with the public sample account only.
"""
from __future__ import annotations

import base64
import getpass
import json
import os
import re
import sys
import unicodedata
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import pytest

import aleo
from aleo import keystore
from aleo.keystore import KeystoreError, LoadedAccount, StoredAccount
from aleo.keystore import _runtime as runtime_module
from aleo.keystore._encrypted import derive_secret, normalize_password
from aleo.keystore._file import parse_account_file, serialize_account_file
from aleo.keystore._keychain import KEY_SERVICE, PASSWORD_SERVICE, delete_keychain, read_keychain
from aleo.keystore._password import password_to_keychain_value
from aleo.keystore._runtime import runtime
from aleo.mainnet import Account, PrivateKey, PrivateKeyCiphertext

from keystore_helpers import (  # noqa: F401  (sandbox is a fixture)
    OTHER_PASSWORD,
    PASSWORD,
    POSIX,
    SAMPLE,
    VECTORS,
    VECTORS_DIR,
    Sandbox,
    absent,
    check_secret,
    code_of,
    failure,
    mode,
    new_account,
    sandbox,
)

pytestmark = pytest.mark.usefixtures("sandbox")

posix_only = pytest.mark.skipif(not POSIX, reason="file modes are not enforced on Windows")


def read_json(path: Any) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Shared sample files
# ---------------------------------------------------------------------------


def test_sources_under_test_are_the_checkout() -> None:
    """The package under test is the source tree next to these tests."""
    assert Path(aleo.__file__).resolve().parent == Path(__file__).resolve().parents[1] / "aleo"


def test_reads_every_sample_account_file(sandbox: Sandbox) -> None:
    sandbox.copy_vectors()
    expected = {
        "none": "none",
        "plaintext": "plaintext",
        "keychain": "keychain",
        "encrypted-js": "encrypted",
        "encrypted-py": "encrypted",
    }
    files = sorted(path.name for path in VECTORS_DIR.iterdir() if path.name != "vectors.json")
    assert files == sorted(f"{name}.json" for name in expected)
    for name, storage in expected.items():
        assert keystore.info(dir=sandbox.dir, name=name) == StoredAccount(
            name=name, path=sandbox.dir / f"{name}.json", address=SAMPLE.address, storage=storage
        )


def test_renders_every_sample_account_file_byte_for_byte() -> None:
    """Parsing a sample and rendering it again gives the same bytes, encrypted ones included."""
    for path in VECTORS_DIR.iterdir():
        if path.name == "vectors.json":
            continue
        data = path.read_bytes()
        assert not data.startswith(b"\xef\xbb\xbf") and b"\r" not in data and data.endswith(b"}\n")
        parsed = parse_account_file(data.decode("utf-8"), str(path))
        assert serialize_account_file(parsed).encode("utf-8") == data, path.name


@pytest.mark.parametrize("storage", ["none", "plaintext", "keychain"])
def test_writes_the_sample_byte_for_byte(sandbox: Sandbox, storage: str) -> None:
    stored = keystore.save(SAMPLE.private_key, dir=sandbox.dir, name=storage, storage=storage)
    assert stored == StoredAccount(
        name=storage, path=sandbox.dir / f"{storage}.json", address=SAMPLE.address, storage=storage
    )
    assert stored.path.read_bytes() == (VECTORS_DIR / f"{storage}.json").read_bytes()


def test_returns_the_sample_key_from_the_plaintext_and_keychain_samples(sandbox: Sandbox) -> None:
    sandbox.copy_vectors()
    sandbox.security.entries[f"{KEY_SERVICE} {SAMPLE.address}"] = SAMPLE.private_key
    for name in ("plaintext", "keychain"):
        loaded = keystore.load(dir=sandbox.dir, name=name)
        assert loaded.source == "account-file"
        assert loaded.address == SAMPLE.address
        check_secret(loaded.private_key, SAMPLE.private_key)
    assert code_of(lambda: keystore.load(dir=sandbox.dir, name="none")) == "KEY_NOT_STORED"


@pytest.mark.parametrize("file", VECTORS["encrypted"]["files"])
def test_decrypts_the_sample_and_rejects_the_wrong_password(sandbox: Sandbox, file: str) -> None:
    sandbox.copy_vectors()
    name = re.sub(r"\.json$", "", file)
    password = VECTORS["encrypted"]["password"]
    wrong = VECTORS["encrypted"]["wrongPassword"]
    check_secret(keystore.export(dir=sandbox.dir, name=name, password=password, prompt=False), SAMPLE.private_key)

    loaded = keystore.load(dir=sandbox.dir, name=name, password=password, prompt=False)
    assert loaded.address == SAMPLE.address
    check_secret(loaded.private_key, SAMPLE.private_key)
    assert loaded.stored is not None and loaded.stored.storage == "encrypted"

    assert code_of(lambda: keystore.export(dir=sandbox.dir, name=name, password=wrong, prompt=False)) == "WRONG_PASSWORD"
    assert code_of(lambda: keystore.load(dir=sandbox.dir, name=name, password=wrong, prompt=False)) == "WRONG_PASSWORD"


def test_sample_files_name_both_encrypted_samples() -> None:
    assert VECTORS["encrypted"]["files"] == ["encrypted-js.json", "encrypted-py.json"]


def test_reproduces_both_scrypt_vectors_including_the_nfkc_one() -> None:
    assert len(VECTORS["scrypt"]) == 2
    for vector in VECTORS["scrypt"]:
        assert [vector["N"], vector["r"], vector["p"]] == [131072, 8, 1]
        secret = derive_secret(normalize_password(vector["password"]), base64.b64decode(vector["salt"]), vector["N"])
        assert secret == vector["secret"]
    nfkc = VECTORS["scrypt"][1]
    assert nfkc["password"] != nfkc["normalized"]
    assert unicodedata.normalize("NFKC", nfkc["password"]) == nfkc["normalized"]
    assert normalize_password(nfkc["password"]) == nfkc["normalized"].encode("utf-8")


def test_uses_the_sample_keychain_service_names_and_password_encoding() -> None:
    assert KEY_SERVICE == VECTORS["keychain"]["keyService"]
    assert PASSWORD_SERVICE == VECTORS["keychain"]["passwordService"]
    encoded = password_to_keychain_value(normalize_password(VECTORS["encrypted"]["password"]))
    assert encoded == VECTORS["keychain"]["passwordValue"]


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_exports_exactly_the_keystore_names() -> None:
    assert sorted(keystore.__all__) == sorted(
        ["KeystoreError", "LoadedAccount", "Storage", "StoredAccount", "export", "info", "load", "migrate", "protect", "save"]
    )
    assert aleo.keystore is keystore
    # A module of the package, not something bound to a network client.
    assert not hasattr(aleo.Aleo, "keystore")


def test_reports_failures_as_keystore_error_with_a_stable_code(sandbox: Sandbox) -> None:
    error = failure(lambda: keystore.load(dir=sandbox.dir))
    assert isinstance(error, KeystoreError)
    assert isinstance(error, aleo.AleoError)
    assert type(error).__name__ == "KeystoreError"
    assert error.code == "ACCOUNT_NOT_FOUND"
    assert str(sandbox.dir / "default.json") in str(error)


def test_result_types_are_frozen_and_hide_the_key(sandbox: Sandbox) -> None:
    account = new_account()
    stored = keystore.save(account.private_key, dir=sandbox.dir)
    loaded = keystore.load(dir=sandbox.dir)
    assert isinstance(loaded, LoadedAccount)
    assert isinstance(stored.path, Path)
    absent(repr(loaded), account.private_key)
    absent(str(loaded), account.private_key)
    with pytest.raises(FrozenInstanceError):
        loaded.address = "aleo1other"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        stored.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Names and directory
# ---------------------------------------------------------------------------


def test_defaults_to_the_account_default_under_the_home_directory(sandbox: Sandbox) -> None:
    stored = keystore.save(Account.random())
    assert stored.name == "default"
    assert stored.path == sandbox.home / ".aleo" / "accounts" / "default.json"
    assert stored.path.exists()


def test_takes_the_name_from_the_argument_then_the_variable(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALEO_ACCOUNT", "from-env")
    assert keystore.save(Account.random(), dir=sandbox.dir).name == "from-env"
    assert keystore.save(Account.random(), dir=sandbox.dir, name="from-option").name == "from-option"
    monkeypatch.setenv("ALEO_ACCOUNT", "")
    assert keystore.save(Account.random(), dir=sandbox.dir).name == "default"


def test_takes_the_directory_from_the_argument_then_the_variable(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    from_env = sandbox.root / "env-dir"
    monkeypatch.setenv("ALEO_ACCOUNTS_DIR", str(from_env))
    assert keystore.save(Account.random()).path == from_env / "default.json"
    assert keystore.save(Account.random(), dir=sandbox.dir).path == sandbox.dir / "default.json"
    # A path object and its text are the same directory; an empty one counts as absent.
    assert keystore.info(dir=str(sandbox.dir)) == keystore.info(dir=sandbox.dir)
    assert keystore.save(Account.random(), dir="", name="other").path == from_env / "other.json"


def test_expands_a_leading_tilde_in_the_argument_and_in_the_variable(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    assert keystore.save(Account.random(), dir="~/keys").path == sandbox.home / "keys" / "default.json"
    monkeypatch.setenv("ALEO_ACCOUNTS_DIR", "~/from-env")
    assert keystore.save(Account.random()).path == sandbox.home / "from-env" / "default.json"
    monkeypatch.setenv("ALEO_ACCOUNTS_DIR", "~")
    assert keystore.save(Account.random(), name="top").path == sandbox.home / "top.json"


def test_leaves_a_tilde_that_names_another_user_alone(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(sandbox.root)
    stored = keystore.save(Account.random(), dir="~someone")
    assert stored.path == sandbox.root / "~someone" / "default.json"


def test_resolves_a_relative_directory_against_the_working_directory(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(sandbox.root)
    assert keystore.save(Account.random(), dir="relative/../keys").path == sandbox.root / "keys" / "default.json"


@pytest.mark.parametrize(
    "name", ["", "Default", "a/b", "../x", ".hidden", "-lead", "_lead", "a.b", "a b", "abc\n", "x" * 65]
)
def test_rejects_the_name(sandbox: Sandbox, name: str) -> None:
    assert code_of(lambda: keystore.save(Account.random(), dir=sandbox.dir, name=name)) == "INVALID_NAME"
    assert code_of(lambda: keystore.info(dir=sandbox.dir, name=name)) == "INVALID_NAME"


def test_rejects_a_name_that_is_not_text(sandbox: Sandbox) -> None:
    assert code_of(lambda: keystore.info(dir=sandbox.dir, name=7)) == "INVALID_NAME"  # type: ignore[arg-type]


def test_rejects_an_invalid_name_from_the_variable(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALEO_ACCOUNT", "Not_Valid")
    assert code_of(lambda: keystore.load(dir=sandbox.dir)) == "INVALID_NAME"


@pytest.mark.parametrize("name", ["a", "0", "main-net_2", "x" * 64])
def test_accepts_the_name(sandbox: Sandbox, name: str) -> None:
    assert keystore.save(Account.random(), dir=sandbox.dir, name=name).name == name


@posix_only
def test_creates_the_directory_at_0700_and_the_file_at_0600(sandbox: Sandbox) -> None:
    previous = os.umask(0o022)
    try:
        stored = keystore.save(Account.random(), dir=sandbox.root / "deep" / "accounts")
    finally:
        os.umask(previous)
    assert mode(stored.path.parent) == 0o700
    assert mode(stored.path) == 0o600
    # Only the account file is left behind: no temporary file.
    assert os.listdir(stored.path.parent) == ["default.json"]


@posix_only
def test_leaves_the_mode_of_an_existing_directory_alone(sandbox: Sandbox) -> None:
    sandbox.dir.mkdir()
    sandbox.dir.chmod(0o755)
    keystore.save(Account.random(), dir=sandbox.dir)
    assert mode(sandbox.dir) == 0o755


@posix_only
def test_tightens_a_file_that_is_wider_than_0600_when_it_is_read(sandbox: Sandbox) -> None:
    stored = keystore.save(Account.random(), dir=sandbox.dir)
    stored.path.chmod(0o644)
    keystore.info(dir=sandbox.dir)
    assert mode(stored.path) == 0o600
    stored.path.chmod(0o640)
    keystore.load(dir=sandbox.dir)
    assert mode(stored.path) == 0o600
    stored.path.chmod(0o400)
    keystore.load(dir=sandbox.dir)
    assert mode(stored.path) == 0o400


@posix_only
def test_does_not_tighten_file_modes_on_windows(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    stored = keystore.save(Account.random(), dir=sandbox.dir)
    stored.path.chmod(0o644)
    monkeypatch.setattr(runtime, "platform", "win32")
    keystore.info(dir=sandbox.dir)
    assert mode(stored.path) == 0o644


# ---------------------------------------------------------------------------
# Account file format
# ---------------------------------------------------------------------------

ADDRESS = SAMPLE.address


def encrypted_sample() -> Dict[str, Any]:
    return read_json(VECTORS_DIR / "encrypted-js.json")


def test_returns_nothing_for_a_missing_file(sandbox: Sandbox) -> None:
    assert keystore.info(dir=sandbox.dir) is None


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("not json", id="text that is not JSON"),
        pytest.param(
            "\N{ZERO WIDTH NO-BREAK SPACE}" + json.dumps({"version": 1, "address": ADDRESS, "key": {"type": "none"}}),
            id="a byte-order mark",
        ),
        pytest.param("[]", id="an array"),
        pytest.param("null", id="null"),
        pytest.param({"address": ADDRESS, "key": {"type": "none"}}, id="a missing version"),
        pytest.param({"version": "1", "address": ADDRESS, "key": {"type": "none"}}, id="a string version"),
        pytest.param({"version": 1.5, "address": ADDRESS, "key": {"type": "none"}}, id="a fractional version"),
        pytest.param({"version": True, "address": ADDRESS, "key": {"type": "none"}}, id="a boolean version"),
        pytest.param({"version": 1, "key": {"type": "none"}}, id="a missing address"),
        pytest.param({"version": 1, "address": 5, "key": {"type": "none"}}, id="a numeric address"),
        pytest.param({"version": 1, "address": ADDRESS}, id="a missing key"),
        pytest.param({"version": 1, "address": ADDRESS, "key": "none"}, id="a key that is not an object"),
        pytest.param({"version": 1, "address": ADDRESS, "key": {}}, id="a missing key type"),
        pytest.param({"version": 1, "address": ADDRESS, "key": {"type": 1}}, id="a numeric key type"),
        pytest.param(
            {"version": 1, "address": ADDRESS, "key": {"type": "none"}, "network": "testnet"},
            id="an extra top-level field",
        ),
        pytest.param(
            {"version": 1, "address": ADDRESS, "key": {"type": "none", "privateKey": "x"}},
            id="an extra field in a none key",
        ),
        pytest.param({"version": 1, "address": ADDRESS, "key": {"type": "plaintext"}}, id="a plaintext key without privateKey"),
        pytest.param(
            {"version": 1, "address": ADDRESS, "key": {"type": "plaintext", "privateKey": 1}},
            id="a plaintext key with a numeric privateKey",
        ),
        pytest.param(
            {"version": 1, "address": ADDRESS, "key": {"type": "plaintext", "privateKey": "x", "note": ""}},
            id="a plaintext key with an extra field",
        ),
        pytest.param(
            {"version": 1, "address": ADDRESS, "key": {"type": "keychain", "service": KEY_SERVICE}},
            id="a keychain key without account",
        ),
        pytest.param(
            {"version": 1, "address": ADDRESS, "key": {"type": "keychain", "service": "login", "account": ADDRESS}},
            id="a keychain key with another service",
        ),
        pytest.param(
            {"version": 1, "address": ADDRESS, "key": {"type": "keychain", "service": KEY_SERVICE, "account": "aleo1other"}},
            id="a keychain key with another account",
        ),
        pytest.param('{"version": NaN, "address": "a", "key": {"type": "none"}}', id="a NaN version"),
        pytest.param('{"version": 1, "address": "a", "key": {"type": "none"}} trailing', id="trailing text"),
    ],
)
def test_rejects_the_file(sandbox: Sandbox, content: Any) -> None:
    sandbox.put("default", content)
    assert code_of(lambda: keystore.info(dir=sandbox.dir)) == "INVALID_ACCOUNT_FILE"
    assert code_of(lambda: keystore.load(dir=sandbox.dir)) == "INVALID_ACCOUNT_FILE"


def test_rejects_a_file_that_is_not_utf8(sandbox: Sandbox) -> None:
    sandbox.dir.mkdir(parents=True)
    (sandbox.dir / "default.json").write_bytes(bytes([0x7B, 0xFF, 0xFE, 0x7D]))
    assert code_of(lambda: keystore.info(dir=sandbox.dir)) == "INVALID_ACCOUNT_FILE"


def _set(path: Sequence[str], value: Any) -> Callable[[Dict[str, Any]], None]:
    def change(file: Dict[str, Any]) -> None:
        target = file
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value

    return change


def _drop(path: Sequence[str]) -> Callable[[Dict[str, Any]], None]:
    def change(file: Dict[str, Any]) -> None:
        target = file
        for part in path[:-1]:
            target = target[part]
        del target[path[-1]]

    return change


def _strip_salt_padding(file: Dict[str, Any]) -> None:
    file["key"]["kdfParams"]["salt"] = file["key"]["kdfParams"]["salt"].rstrip("=")


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(_set(["key", "kdf"], "argon2id"), id="another kdf"),
        pytest.param(_set(["key", "kdfParams", "N"], 2**13), id="N below 2^14"),
        pytest.param(_set(["key", "kdfParams", "N"], 2**21), id="N above 2^20"),
        pytest.param(_set(["key", "kdfParams", "N"], 100000), id="N that is not a power of two"),
        pytest.param(_set(["key", "kdfParams", "N"], 16384.5), id="a fractional N"),
        pytest.param(_set(["key", "kdfParams", "N"], "131072"), id="a string N"),
        pytest.param(_set(["key", "kdfParams", "N"], True), id="a boolean N"),
        pytest.param(_set(["key", "kdfParams", "r"], 16), id="r other than 8"),
        pytest.param(_set(["key", "kdfParams", "p"], 2), id="p other than 1"),
        pytest.param(_set(["key", "kdfParams", "p"], True), id="a boolean p"),
        pytest.param(
            _set(["key", "kdfParams", "salt"], base64.b64encode(bytes(15)).decode("ascii")),
            id="a salt shorter than 16 bytes",
        ),
        pytest.param(_set(["key", "kdfParams", "salt"], "!!!!!!!!!!!!!!!!!!!!!!!!"), id="a salt that is not base64"),
        pytest.param(_strip_salt_padding, id="a salt without padding"),
        pytest.param(_set(["key", "kdfParams", "salt"], "AAECAwQFBgcICQoLDA0ODw==\n"), id="a salt with a line ending"),
        pytest.param(_drop(["key", "ciphertext"]), id="a missing ciphertext"),
        pytest.param(_set(["key", "kdfParams", "dkLen"], 32), id="an extra kdf parameter"),
        pytest.param(_set(["key", "hint"], "x"), id="an extra key field"),
    ],
)
def test_rejects_an_encrypted_file(sandbox: Sandbox, change: Callable[[Dict[str, Any]], None]) -> None:
    file = encrypted_sample()
    change(file)
    sandbox.put("default", file)
    assert code_of(lambda: keystore.info(dir=sandbox.dir)) == "INVALID_ACCOUNT_FILE"


@pytest.mark.parametrize("cost", [2**14, 2**15, 2**20])
def test_accepts_an_encrypted_file_with_cost(sandbox: Sandbox, cost: int) -> None:
    file = encrypted_sample()
    file["key"]["kdfParams"]["N"] = cost
    sandbox.put("default", file)
    stored = keystore.info(dir=sandbox.dir)
    assert stored is not None and stored.storage == "encrypted"


def test_accepts_whole_numbers_written_with_a_fraction_or_an_exponent(sandbox: Sandbox) -> None:
    # JSON has one number type, so 1.0 and 1e0 are the integer 1.
    text = (VECTORS_DIR / "encrypted-js.json").read_text(encoding="utf-8")
    replacements = [
        ('"version": 1,', '"version": 1.0,'),
        ('"N": 131072,', '"N": 1.31072e5,'),
        ('"r": 8,', '"r": 8.0,'),
        ('"p": 1,', '"p": 1e0,'),
    ]
    for old, new in replacements:
        assert old in text
        text = text.replace(old, new)
    sandbox.put("default", text)
    stored = keystore.info(dir=sandbox.dir)
    assert stored is not None and stored.storage == "encrypted"
    check_secret(
        keystore.export(dir=sandbox.dir, password=VECTORS["encrypted"]["password"], prompt=False), SAMPLE.private_key
    )

    sandbox.put("default", '{"version":2.0,"address":"' + ADDRESS + '","key":{"type":"none"}}\n')
    assert code_of(lambda: keystore.info(dir=sandbox.dir)) == "UNSUPPORTED_VERSION"


def test_decrypts_a_file_whose_cost_is_above_the_default(sandbox: Sandbox) -> None:
    # Written by hand at N = 2^18, which needs more memory than the default cost.
    account = new_account()
    salt = bytes([7]) * 16
    secret = derive_secret(normalize_password(PASSWORD), salt, 2**18)
    ciphertext = str(PrivateKeyCiphertext.encrypt_private_key(PrivateKey.from_string(account.private_key), secret))
    sandbox.put(
        "default",
        {
            "version": 1,
            "address": account.address,
            "key": {
                "type": "encrypted",
                "kdf": "scrypt",
                "kdfParams": {"N": 2**18, "r": 8, "p": 1, "salt": base64.b64encode(salt).decode("ascii")},
                "ciphertext": ciphertext,
            },
        },
    )
    check_secret(keystore.export(dir=sandbox.dir, password=PASSWORD), account.private_key)


@pytest.mark.parametrize("version", [0, 2, -1])
def test_reports_the_version_as_unsupported_even_with_unknown_fields(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch, version: int
) -> None:
    sandbox.put("default", {"version": version, "address": ADDRESS, "key": {"type": "none"}, "added": True})
    assert code_of(lambda: keystore.info(dir=sandbox.dir)) == "UNSUPPORTED_VERSION"
    monkeypatch.setenv("ALEO_PRIVATE_KEY", SAMPLE.private_key)
    assert code_of(lambda: keystore.load(dir=sandbox.dir)) == "UNSUPPORTED_VERSION"


def test_reports_an_unknown_storage_type_as_unsupported_and_does_not_fall_back(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox.put("default", {"version": 1, "address": ADDRESS, "key": {"type": "secret-service", "collection": "login"}})
    assert code_of(lambda: keystore.info(dir=sandbox.dir)) == "UNSUPPORTED_STORAGE"
    monkeypatch.setenv("ALEO_PRIVATE_KEY", SAMPLE.private_key)
    assert code_of(lambda: keystore.load(dir=sandbox.dir)) == "UNSUPPORTED_STORAGE"
    assert code_of(lambda: keystore.load(dir=sandbox.dir, legacy_key=SAMPLE.private_key)) == "UNSUPPORTED_STORAGE"


def test_does_not_quote_a_malformed_file_in_the_error(sandbox: Sandbox) -> None:
    account = new_account()
    sandbox.put(
        "default",
        f'{{"version": 1, "address": "{account.address}", "key": {{"type": "plaintext", "privateKey": "{account.private_key}" oops',
    )
    error = failure(lambda: keystore.load(dir=sandbox.dir))
    assert error.code == "INVALID_ACCOUNT_FILE"
    absent(str(error), account.private_key)
    assert error.__cause__ is None
    assert error.__context__ is None


def test_reports_an_unparseable_or_mismatched_plaintext_key(sandbox: Sandbox) -> None:
    sandbox.put("default", {"version": 1, "address": ADDRESS, "key": {"type": "plaintext", "privateKey": "APrivateKey1nope"}})
    stored = keystore.info(dir=sandbox.dir)
    assert stored is not None and stored.storage == "plaintext"
    assert code_of(lambda: keystore.load(dir=sandbox.dir)) == "INVALID_KEY"

    other = new_account()
    sandbox.put("default", {"version": 1, "address": ADDRESS, "key": {"type": "plaintext", "privateKey": other.private_key}})
    error = failure(lambda: keystore.load(dir=sandbox.dir))
    assert error.code == "ADDRESS_MISMATCH"
    assert ADDRESS in str(error)
    assert other.address in str(error)
    absent(str(error), other.private_key)


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------


def test_stores_plaintext_by_default_and_returns_the_account_without_its_key(sandbox: Sandbox) -> None:
    account = new_account()
    stored = keystore.save(account.private_key, dir=sandbox.dir)
    assert stored == StoredAccount(
        name="default", path=sandbox.dir / "default.json", address=account.address, storage="plaintext"
    )
    file = read_json(stored.path)
    assert list(file) == ["version", "address", "key"]
    assert list(file["key"]) == ["type", "privateKey"]
    check_secret(file["key"]["privateKey"], account.private_key)
    assert stored.path.read_text(encoding="utf-8") == json.dumps(file, indent=2, ensure_ascii=False) + "\n"


def test_accepts_accounts_and_keys_of_either_network_and_key_strings(sandbox: Sandbox) -> None:
    import aleo.testnet as testnet

    mainnet_account = Account.random()
    testnet_account = testnet.Account.random()
    testnet_key = testnet.PrivateKey.random()
    mainnet_key = PrivateKey.random()
    cases = [
        ("mainnet-account", mainnet_account, str(mainnet_account.address), str(mainnet_account.private_key)),
        ("testnet-account", testnet_account, str(testnet_account.address), str(testnet_account.private_key)),
        ("mainnet-key", mainnet_key, str(mainnet_key.address), str(mainnet_key)),
        ("testnet-key", testnet_key, str(testnet_key.address), str(testnet_key)),
        ("key-string", str(mainnet_key), str(mainnet_key.address), str(mainnet_key)),
    ]
    for name, account, address, private_key in cases:
        stored = keystore.save(account, dir=sandbox.dir, name=name)
        assert stored.address == address
        check_secret(keystore.export(dir=sandbox.dir, name=name), private_key)


def test_accepts_a_loaded_account(sandbox: Sandbox) -> None:
    account = new_account()
    loaded = keystore.load(dir=sandbox.dir, private_key=account.private_key)
    stored = keystore.save(loaded, dir=sandbox.dir)
    assert stored.address == account.address
    check_secret(keystore.export(dir=sandbox.dir), account.private_key)


def test_returns_without_rewriting_for_the_same_account_and_storage(sandbox: Sandbox) -> None:
    account = new_account()
    first = keystore.save(account.private_key, dir=sandbox.dir)
    before = os.stat(first.path)
    again = keystore.save(account.private_key, dir=sandbox.dir)
    assert again == first
    after = os.stat(first.path)
    assert after.st_ino == before.st_ino
    assert after.st_mtime_ns == before.st_mtime_ns


def test_returns_for_the_same_encrypted_account_without_a_password_or_a_keychain_call(sandbox: Sandbox) -> None:
    account = new_account()
    first = keystore.save(account.private_key, dir=sandbox.dir, storage="encrypted", password=PASSWORD)
    data = first.path.read_bytes()
    assert keystore.save(account.private_key, dir=sandbox.dir, storage="encrypted") == first
    again = keystore.save(
        account.private_key, dir=sandbox.dir, storage="encrypted", password=OTHER_PASSWORD, save_password=True
    )
    assert again == first
    assert first.path.read_bytes() == data
    assert sandbox.security.calls == []
    assert sandbox.prompts == []


def test_refuses_the_same_account_with_another_storage_and_points_at_protect(sandbox: Sandbox) -> None:
    account = new_account()
    keystore.save(account.private_key, dir=sandbox.dir)
    error = failure(lambda: keystore.save(account.private_key, dir=sandbox.dir, storage="none"))
    assert error.code == "ACCOUNT_EXISTS"
    assert "protect" in str(error)
    stored = keystore.info(dir=sandbox.dir)
    assert stored is not None and stored.storage == "plaintext"


def test_refuses_to_replace_a_file_that_holds_another_address(sandbox: Sandbox) -> None:
    first = new_account()
    second = new_account()
    stored = keystore.save(first.private_key, dir=sandbox.dir)
    data = stored.path.read_bytes()
    error = failure(lambda: keystore.save(second.private_key, dir=sandbox.dir))
    assert error.code == "ACCOUNT_EXISTS"
    assert first.address in str(error)
    assert second.address in str(error)
    assert stored.path.read_bytes() == data


def test_does_not_overwrite_a_file_it_cannot_read(sandbox: Sandbox) -> None:
    path = sandbox.put("default", {"version": 2, "address": "aleo1future", "key": {}})
    data = path.read_bytes()
    assert code_of(lambda: keystore.save(Account.random(), dir=sandbox.dir)) == "UNSUPPORTED_VERSION"
    assert path.read_bytes() == data


def test_rejects_a_key_that_cannot_be_parsed_or_does_not_derive_the_address(sandbox: Sandbox) -> None:
    account = new_account()
    bad = failure(lambda: keystore.save("APrivateKey1nope", dir=sandbox.dir))
    assert bad.code == "INVALID_KEY"
    assert "APrivateKey1nope" not in str(bad)
    assert bad.__cause__ is None and bad.__context__ is None
    for not_a_key in (None, 7, b"APrivateKey1", object()):
        assert code_of(lambda: keystore.save(not_a_key, dir=sandbox.dir)) == "INVALID_KEY"
    # An object that states an address must hold the key of that address.
    stated = LoadedAccount(address=SAMPLE.address, private_key=account.private_key, source="argument")
    assert code_of(lambda: keystore.save(stated, dir=sandbox.dir)) == "ADDRESS_MISMATCH"
    assert not (sandbox.dir / "default.json").exists()


def test_rejects_an_unknown_storage_type(sandbox: Sandbox) -> None:
    for storage in ("vault", "", None, 1, ["plaintext"]):
        assert code_of(lambda: keystore.save(Account.random(), dir=sandbox.dir, storage=storage)) == "UNSUPPORTED_STORAGE"  # type: ignore[arg-type]
    assert not sandbox.dir.exists()


def test_writes_the_address_only_form_for_storage_none(sandbox: Sandbox) -> None:
    account = new_account()
    stored = keystore.save(account.private_key, dir=sandbox.dir, storage="none")
    assert stored.storage == "none"
    absent(stored.path.read_text(encoding="utf-8"), account.private_key)
    assert read_json(stored.path) == {"version": 1, "address": account.address, "key": {"type": "none"}}


# ---------------------------------------------------------------------------
# load: the lookup order
# ---------------------------------------------------------------------------


def test_fails_with_account_not_found_when_nothing_supplies_a_key_and_generates_nothing(sandbox: Sandbox) -> None:
    assert code_of(lambda: keystore.load(dir=sandbox.dir)) == "ACCOUNT_NOT_FOUND"
    assert not sandbox.dir.exists()


def test_reads_the_account_file_when_the_environment_supplies_nothing(sandbox: Sandbox) -> None:
    account = new_account()
    stored = keystore.save(account.private_key, dir=sandbox.dir)
    loaded = keystore.load(dir=sandbox.dir)
    assert loaded.source == "account-file"
    assert loaded.stored == stored
    assert loaded.address == account.address
    check_secret(loaded.private_key, account.private_key)
    assert isinstance(loaded.address, str) and isinstance(loaded.private_key, str)


def test_a_loaded_key_becomes_an_account_on_either_network(sandbox: Sandbox) -> None:
    account = new_account()
    keystore.save(account.private_key, dir=sandbox.dir)
    loaded = keystore.load(dir=sandbox.dir)
    for network in ("mainnet", "testnet"):
        client = aleo.Aleo(aleo.HTTPProvider("https://api.provable.com/v2", network=network))
        assert str(client.account.from_private_key(loaded.private_key).address) == account.address


def test_prefers_an_explicit_key_and_then_ignores_the_key_variables(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    account = new_account()
    monkeypatch.setenv("ALEO_PRIVATE_KEY", "")
    monkeypatch.setenv("SHIELD_SWAP_PRIVATE_KEY", new_account().private_key)
    loaded = keystore.load(dir=sandbox.dir, private_key=account.private_key)
    assert loaded.source == "argument"
    assert loaded.stored is None
    assert loaded.address == account.address


@pytest.mark.parametrize("variable", ["ALEO_PRIVATE_KEY", "SHIELD_SWAP_PRIVATE_KEY"])
def test_reads_the_key_from_the_variable(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch, variable: str) -> None:
    account = new_account()
    monkeypatch.setenv(variable, f"  {account.private_key}\n")
    loaded = keystore.load(dir=sandbox.dir)
    assert loaded.source == "env"
    assert loaded.address == account.address
    check_secret(loaded.private_key, account.private_key)
    # A key from the environment is used for the process and never stored.
    assert not sandbox.dir.exists()


@pytest.mark.parametrize("variable", ["ALEO_PRIVATE_KEY_FILE", "SHIELD_SWAP_PRIVATE_KEY_FILE"])
def test_reads_the_key_from_the_file_named_by_the_variable(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    account = new_account()
    key_file = sandbox.root / "key.txt"
    key_file.write_text(f"\n {account.private_key} \n", encoding="utf-8")
    monkeypatch.setenv(variable, str(key_file))
    loaded = keystore.load(dir=sandbox.dir)
    assert loaded.source == "env-file"
    assert loaded.address == account.address
    assert not sandbox.dir.exists()


def test_reads_a_key_file_that_starts_with_a_byte_order_mark(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    account = new_account()
    key_file = sandbox.root / "key.txt"
    key_file.write_bytes(b"\xef\xbb\xbf" + account.private_key.encode("ascii") + b"\r\n")
    monkeypatch.setenv("ALEO_PRIVATE_KEY_FILE", str(key_file))
    assert keystore.load(dir=sandbox.dir).address == account.address


def test_reports_the_variable_ahead_of_the_key_file_when_both_hold_the_same_key(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    account = new_account()
    key_file = sandbox.root / "key.txt"
    key_file.write_text(account.private_key, encoding="utf-8")
    monkeypatch.setenv("ALEO_PRIVATE_KEY_FILE", str(key_file))
    monkeypatch.setenv("SHIELD_SWAP_PRIVATE_KEY_FILE", str(key_file))
    monkeypatch.setenv("SHIELD_SWAP_PRIVATE_KEY", account.private_key)
    monkeypatch.setenv("ALEO_PRIVATE_KEY", account.private_key)
    loaded = keystore.load(dir=sandbox.dir)
    assert loaded.source == "env"
    assert loaded.address == account.address


@pytest.mark.parametrize(
    ("one", "two"),
    [
        ("ALEO_PRIVATE_KEY", "SHIELD_SWAP_PRIVATE_KEY"),
        ("ALEO_PRIVATE_KEY", "ALEO_PRIVATE_KEY_FILE"),
        ("SHIELD_SWAP_PRIVATE_KEY", "SHIELD_SWAP_PRIVATE_KEY_FILE"),
        ("ALEO_PRIVATE_KEY_FILE", "SHIELD_SWAP_PRIVATE_KEY_FILE"),
    ],
)
def test_fails_with_key_conflict_when_two_variables_hold_different_keys(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch, one: str, two: str
) -> None:
    first = new_account()
    second = new_account()

    def set_variable(variable: str, key: str) -> None:
        if variable.endswith("_FILE"):
            key_file = sandbox.root / f"{variable}.txt"
            key_file.write_text(key, encoding="utf-8")
            monkeypatch.setenv(variable, str(key_file))
        else:
            monkeypatch.setenv(variable, key)

    set_variable(one, first.private_key)
    set_variable(two, second.private_key)
    error = failure(lambda: keystore.load(dir=sandbox.dir))
    assert error.code == "KEY_CONFLICT"
    assert one in str(error)
    assert two in str(error)
    absent(str(error), first.private_key)
    absent(str(error), second.private_key)


@pytest.mark.parametrize(
    "variable", ["ALEO_PRIVATE_KEY", "SHIELD_SWAP_PRIVATE_KEY", "ALEO_PRIVATE_KEY_FILE", "SHIELD_SWAP_PRIVATE_KEY_FILE"]
)
def test_fails_with_empty_key_source_when_the_variable_is_set_but_empty_even_with_an_account_file(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    keystore.save(Account.random(), dir=sandbox.dir)
    monkeypatch.setenv(variable, "")
    assert code_of(lambda: keystore.load(dir=sandbox.dir)) == "EMPTY_KEY_SOURCE"
    monkeypatch.setenv(variable, "   ")
    assert code_of(lambda: keystore.load(dir=sandbox.dir)) == "EMPTY_KEY_SOURCE"


def test_fails_with_empty_key_source_for_an_empty_or_missing_key_file(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    key_file = sandbox.root / "key.txt"
    key_file.write_text(" \n", encoding="utf-8")
    monkeypatch.setenv("ALEO_PRIVATE_KEY_FILE", str(key_file))
    assert code_of(lambda: keystore.load(dir=sandbox.dir)) == "EMPTY_KEY_SOURCE"
    monkeypatch.setenv("ALEO_PRIVATE_KEY_FILE", str(sandbox.root / "absent.txt"))
    error = failure(lambda: keystore.load(dir=sandbox.dir, legacy_key=new_account().private_key))
    assert error.code == "EMPTY_KEY_SOURCE"
    assert "absent.txt" in str(error)
    assert "ENOENT" in str(error)


def test_fails_with_invalid_key_without_echoing_the_value(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALEO_PRIVATE_KEY", "APrivateKey1definitelynotakey")
    error = failure(lambda: keystore.load(dir=sandbox.dir))
    assert error.code == "INVALID_KEY"
    assert "ALEO_PRIVATE_KEY" in str(error)
    assert "definitelynotakey" not in str(error)
    assert error.__cause__ is None and error.__context__ is None
    monkeypatch.delenv("ALEO_PRIVATE_KEY")
    assert code_of(lambda: keystore.load(dir=sandbox.dir, private_key="")) == "INVALID_KEY"
    assert code_of(lambda: keystore.load(dir=sandbox.dir, private_key=7)) == "INVALID_KEY"  # type: ignore[arg-type]


def test_requires_a_key_from_the_caller_or_the_environment_to_match_the_account_file(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    account = new_account()
    other = new_account()
    stored = keystore.save(account.private_key, dir=sandbox.dir)

    monkeypatch.setenv("ALEO_PRIVATE_KEY", account.private_key)
    matching = keystore.load(dir=sandbox.dir)
    assert matching.source == "env"
    assert matching.stored == stored

    monkeypatch.setenv("ALEO_PRIVATE_KEY", other.private_key)
    error = failure(lambda: keystore.load(dir=sandbox.dir))
    assert error.code == "ADDRESS_MISMATCH"
    assert account.address in str(error)
    assert other.address in str(error)

    monkeypatch.delenv("ALEO_PRIVATE_KEY")
    assert code_of(lambda: keystore.load(dir=sandbox.dir, private_key=other.private_key)) == "ADDRESS_MISMATCH"
    assert keystore.load(dir=sandbox.dir, private_key=account.private_key).source == "argument"


def test_uses_an_address_only_file_to_check_a_key_from_the_environment(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    account = new_account()
    keystore.save(account.private_key, dir=sandbox.dir, storage="none")
    assert code_of(lambda: keystore.load(dir=sandbox.dir)) == "KEY_NOT_STORED"
    assert code_of(lambda: keystore.load(dir=sandbox.dir, legacy_key=account.private_key)) == "KEY_NOT_STORED"

    monkeypatch.setenv("ALEO_PRIVATE_KEY", account.private_key)
    loaded = keystore.load(dir=sandbox.dir)
    assert loaded.source == "env"
    assert loaded.stored is not None and loaded.stored.storage == "none"

    monkeypatch.setenv("ALEO_PRIVATE_KEY", new_account().private_key)
    assert code_of(lambda: keystore.load(dir=sandbox.dir)) == "ADDRESS_MISMATCH"


def test_checks_the_resolved_key_against_the_expected_address(sandbox: Sandbox) -> None:
    account = new_account()
    keystore.save(account.private_key, dir=sandbox.dir)
    assert keystore.load(dir=sandbox.dir, expected_address=account.address).address == account.address
    error = failure(lambda: keystore.load(dir=sandbox.dir, expected_address=SAMPLE.address))
    assert error.code == "ADDRESS_MISMATCH"
    assert account.address in str(error)
    assert SAMPLE.address in str(error)
    empty = sandbox.root / "empty"
    assert (
        code_of(lambda: keystore.load(dir=empty, private_key=account.private_key, expected_address=SAMPLE.address))
        == "ADDRESS_MISMATCH"
    )


def test_uses_the_legacy_key_only_when_nothing_else_supplies_one(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    legacy = new_account()
    from_legacy = keystore.load(dir=sandbox.dir, legacy_key=legacy.private_key)
    assert from_legacy.source == "legacy"
    assert from_legacy.stored is None
    assert from_legacy.address == legacy.address
    assert not sandbox.dir.exists()

    env = new_account()
    monkeypatch.setenv("ALEO_PRIVATE_KEY", env.private_key)
    from_env = keystore.load(dir=sandbox.dir, legacy_key=legacy.private_key)
    assert from_env.source == "env"
    assert from_env.address == env.address
    monkeypatch.delenv("ALEO_PRIVATE_KEY")

    assert code_of(lambda: keystore.load(dir=sandbox.dir, legacy_key="APrivateKey1nope")) == "INVALID_KEY"


def test_prefers_the_account_file_over_a_legacy_key_and_requires_them_to_match(sandbox: Sandbox) -> None:
    account = new_account()
    keystore.save(account.private_key, dir=sandbox.dir)
    loaded = keystore.load(dir=sandbox.dir, legacy_key=account.private_key)
    assert loaded.source == "account-file"
    error = failure(lambda: keystore.load(dir=sandbox.dir, legacy_key=new_account().private_key))
    assert error.code == "ADDRESS_MISMATCH"
    assert account.address in str(error)


def test_reports_a_bad_variable_ahead_of_a_bad_account_file(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox.put("default", "not json")
    monkeypatch.setenv("SHIELD_SWAP_PRIVATE_KEY_FILE", "")
    assert code_of(lambda: keystore.load(dir=sandbox.dir)) == "EMPTY_KEY_SOURCE"
    # With an explicit key the variables are not read, so the file is what fails.
    assert code_of(lambda: keystore.load(dir=sandbox.dir, private_key=new_account().private_key)) == "INVALID_ACCOUNT_FILE"


def test_validates_every_variable_that_is_set(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    account = new_account()
    monkeypatch.setenv("ALEO_PRIVATE_KEY", account.private_key)
    monkeypatch.setenv("SHIELD_SWAP_PRIVATE_KEY", "APrivateKey1nope")
    error = failure(lambda: keystore.load(dir=sandbox.dir))
    assert error.code == "INVALID_KEY"
    assert "SHIELD_SWAP_PRIVATE_KEY" in str(error)


def test_treats_a_directory_named_as_the_key_file_as_an_empty_source(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALEO_PRIVATE_KEY_FILE", str(sandbox.home))
    assert code_of(lambda: keystore.load(dir=sandbox.dir)) == "EMPTY_KEY_SOURCE"
    # The path is used as written: a leading "~" is not expanded.
    (sandbox.home / "key.txt").write_text(new_account().private_key, encoding="utf-8")
    monkeypatch.setenv("ALEO_PRIVATE_KEY_FILE", "~/key.txt")
    assert code_of(lambda: keystore.load(dir=sandbox.dir)) == "EMPTY_KEY_SOURCE"


def test_does_not_trim_a_key_passed_in_code(sandbox: Sandbox) -> None:
    account = new_account()
    padded = f" {account.private_key}\n"
    assert code_of(lambda: keystore.load(dir=sandbox.dir, private_key=padded)) == "INVALID_KEY"
    assert code_of(lambda: keystore.load(dir=sandbox.dir, legacy_key=padded)) == "INVALID_KEY"
    assert code_of(lambda: keystore.save(padded, dir=sandbox.dir)) == "INVALID_KEY"


def test_ignores_an_unused_legacy_key_when_the_environment_supplies_the_key(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    account = new_account()
    monkeypatch.setenv("ALEO_PRIVATE_KEY", account.private_key)
    loaded = keystore.load(dir=sandbox.dir, legacy_key="APrivateKey1nope")
    assert loaded.source == "env"
    assert loaded.address == account.address


def test_reads_the_stored_key_before_checking_the_expected_address(sandbox: Sandbox) -> None:
    account = new_account()
    keystore.save(account.private_key, dir=sandbox.dir, storage="encrypted", password=PASSWORD)
    assert code_of(lambda: keystore.load(dir=sandbox.dir, expected_address=SAMPLE.address)) == "PASSWORD_REQUIRED"
    assert (
        code_of(lambda: keystore.load(dir=sandbox.dir, expected_address=SAMPLE.address, password=PASSWORD))
        == "ADDRESS_MISMATCH"
    )


def test_passes_file_system_failures_on_the_account_file_through(sandbox: Sandbox) -> None:
    sandbox.dir.mkdir()
    blocked = sandbox.dir / "default.json"
    blocked.mkdir()
    before = mode(blocked)
    for call in (
        lambda: keystore.info(dir=sandbox.dir),
        lambda: keystore.load(dir=sandbox.dir),
        lambda: keystore.export(dir=sandbox.dir),
        lambda: keystore.save(Account.random(), dir=sandbox.dir),
        lambda: keystore.protect(dir=sandbox.dir, storage="plaintext"),
    ):
        with pytest.raises(OSError) as raised:
            call()
        assert not isinstance(raised.value, KeystoreError)
    assert mode(blocked) == before


# ---------------------------------------------------------------------------
# Encrypted storage and password sources
# ---------------------------------------------------------------------------


def test_writes_the_encrypted_form_with_a_fresh_salt_and_the_fixed_scrypt_cost(sandbox: Sandbox) -> None:
    account = new_account()
    stored = keystore.save(account.private_key, dir=sandbox.dir, storage="encrypted", password=PASSWORD)
    assert stored.storage == "encrypted"
    text = stored.path.read_text(encoding="utf-8")
    absent(text, account.private_key)
    absent(text, PASSWORD)
    file = json.loads(text)
    assert list(file) == ["version", "address", "key"]
    assert list(file["key"]) == ["type", "kdf", "kdfParams", "ciphertext"]
    assert list(file["key"]["kdfParams"]) == ["N", "r", "p", "salt"]
    assert file["key"]["kdf"] == "scrypt"
    params = file["key"]["kdfParams"]
    assert [params["N"], params["r"], params["p"]] == [131072, 8, 1]
    assert all(type(params[field]) is int for field in ("N", "r", "p"))
    assert type(file["version"]) is int
    assert len(base64.b64decode(params["salt"])) == 16
    assert re.fullmatch(r"[A-Za-z0-9+/]{22}==", params["salt"])
    assert file["key"]["ciphertext"].startswith("ciphertext1")
    assert text == json.dumps(file, indent=2, ensure_ascii=False) + "\n"
    # Nothing reached the Keychain without save_password.
    assert sandbox.security.calls == []

    second = keystore.save(account.private_key, dir=sandbox.dir, name="second", storage="encrypted", password=PASSWORD)
    other = read_json(second.path)
    assert other["key"]["kdfParams"]["salt"] != params["salt"]


def test_takes_the_password_from_the_argument_then_the_variable(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    account = new_account()
    keystore.save(account.private_key, dir=sandbox.dir, storage="encrypted", password=PASSWORD)

    via_argument = keystore.load(dir=sandbox.dir, password=PASSWORD, prompt=False)
    assert via_argument.source == "account-file"
    check_secret(via_argument.private_key, account.private_key)

    monkeypatch.setenv("ALEO_ACCOUNT_PASSWORD", PASSWORD)
    assert keystore.load(dir=sandbox.dir, prompt=False).address == account.address

    # The argument wins over the variable, and a wrong one does not fall through.
    monkeypatch.setenv("ALEO_ACCOUNT_PASSWORD", OTHER_PASSWORD)
    assert keystore.load(dir=sandbox.dir, password=PASSWORD).address == account.address
    monkeypatch.setenv("ALEO_ACCOUNT_PASSWORD", PASSWORD)
    assert code_of(lambda: keystore.load(dir=sandbox.dir, password=OTHER_PASSWORD)) == "WRONG_PASSWORD"


def test_does_not_fall_through_to_the_keychain_or_the_prompt_after_a_wrong_variable(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    account = new_account()
    keystore.save(account.private_key, dir=sandbox.dir, storage="encrypted", password=PASSWORD, save_password=True)
    sandbox.terminal = True
    sandbox.answers.append(PASSWORD)
    monkeypatch.setenv("ALEO_ACCOUNT_PASSWORD", OTHER_PASSWORD)
    error = failure(lambda: keystore.load(dir=sandbox.dir))
    assert error.code == "WRONG_PASSWORD"
    absent(str(error), OTHER_PASSWORD)
    assert sandbox.prompts == []


def test_sets_the_password_from_the_variable_when_none_is_passed(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    account = new_account()
    monkeypatch.setenv("ALEO_ACCOUNT_PASSWORD", PASSWORD)
    keystore.save(account.private_key, dir=sandbox.dir, storage="encrypted")
    monkeypatch.delenv("ALEO_ACCOUNT_PASSWORD")
    check_secret(keystore.export(dir=sandbox.dir, password=PASSWORD), account.private_key)


def test_treats_passwords_that_are_equal_under_nfkc_as_the_same_password(sandbox: Sandbox) -> None:
    account = new_account()
    written = "\N{FULLWIDTH LATIN SMALL LETTER P}ass\N{NO-BREAK SPACE}word e\N{COMBINING ACUTE ACCENT}"
    typed = "pass word \N{LATIN SMALL LETTER E WITH ACUTE}"
    assert written != typed
    keystore.save(account.private_key, dir=sandbox.dir, storage="encrypted", password=written)
    check_secret(keystore.export(dir=sandbox.dir, password=typed), account.private_key)


def test_rejects_an_empty_password(sandbox: Sandbox) -> None:
    account = new_account()
    assert (
        code_of(lambda: keystore.save(account.private_key, dir=sandbox.dir, storage="encrypted", password=""))
        == "PASSWORD_REQUIRED"
    )
    assert not (sandbox.dir / "default.json").exists()
    keystore.save(account.private_key, dir=sandbox.dir, storage="encrypted", password=PASSWORD)
    assert code_of(lambda: keystore.load(dir=sandbox.dir, password="")) == "PASSWORD_REQUIRED"


def test_fails_with_password_required_when_no_source_supplies_a_password(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    account = new_account()
    assert code_of(lambda: keystore.save(account.private_key, dir=sandbox.dir, storage="encrypted")) == "PASSWORD_REQUIRED"
    assert not (sandbox.dir / "default.json").exists()

    keystore.save(account.private_key, dir=sandbox.dir, storage="encrypted", password=PASSWORD)
    assert code_of(lambda: keystore.load(dir=sandbox.dir)) == "PASSWORD_REQUIRED"
    assert code_of(lambda: keystore.export(dir=sandbox.dir)) == "PASSWORD_REQUIRED"
    # An empty variable counts as unset.
    monkeypatch.setenv("ALEO_ACCOUNT_PASSWORD", "")
    assert code_of(lambda: keystore.load(dir=sandbox.dir)) == "PASSWORD_REQUIRED"
    # A terminal is attached, but the caller forbade prompting.
    sandbox.terminal = True
    assert code_of(lambda: keystore.load(dir=sandbox.dir, prompt=False)) == "PASSWORD_REQUIRED"
    assert sandbox.prompts == []


def test_reads_a_saved_password_from_the_keychain(sandbox: Sandbox) -> None:
    account = new_account()
    keystore.save(account.private_key, dir=sandbox.dir, storage="encrypted", password=PASSWORD, save_password=True)
    assert sandbox.security.entries.get(f"{PASSWORD_SERVICE} {account.address}") == PASSWORD.encode("utf-8").hex()
    sandbox.security.calls.clear()

    loaded = keystore.load(dir=sandbox.dir, prompt=False)
    assert loaded.address == account.address
    assert sandbox.security.calls == [
        {"args": ["find-generic-password", "-s", "aleo-account-password", "-a", account.address, "-w"], "stdin": None}
    ]


def test_fails_on_a_wrong_saved_password_without_prompting(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    account = new_account()
    keystore.save(account.private_key, dir=sandbox.dir, storage="encrypted", password=PASSWORD)
    sandbox.security.entries[f"{PASSWORD_SERVICE} {account.address}"] = OTHER_PASSWORD.encode("utf-8").hex()
    sandbox.terminal = True
    sandbox.answers.append(PASSWORD)
    assert code_of(lambda: keystore.load(dir=sandbox.dir)) == "WRONG_PASSWORD"
    assert sandbox.prompts == []
    # The variable is consulted first, so it still unlocks the account.
    monkeypatch.setenv("ALEO_ACCOUNT_PASSWORD", PASSWORD)
    assert keystore.load(dir=sandbox.dir).address == account.address


def test_reports_a_failing_keychain_lookup_instead_of_prompting(sandbox: Sandbox) -> None:
    account = new_account()
    keystore.save(account.private_key, dir=sandbox.dir, storage="encrypted", password=PASSWORD)
    sandbox.security.force(lambda args: (36, "") if args[0] == "find-generic-password" else None)
    sandbox.terminal = True
    assert code_of(lambda: keystore.load(dir=sandbox.dir)) == "KEYCHAIN_FAILED"
    sandbox.security.force(None)
    for not_a_saved_password in ("not-hex", "", "abc", "ABCD"):
        sandbox.security.entries[f"{PASSWORD_SERVICE} {account.address}"] = not_a_saved_password
        assert code_of(lambda: keystore.load(dir=sandbox.dir)) == "KEYCHAIN_FAILED"
    assert sandbox.prompts == []


def test_skips_the_keychain_outside_macos_and_ignores_save_password_there(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime, "platform", "linux")
    account = new_account()
    keystore.save(account.private_key, dir=sandbox.dir, storage="encrypted", password=PASSWORD, save_password=True)
    assert code_of(lambda: keystore.load(dir=sandbox.dir)) == "PASSWORD_REQUIRED"
    monkeypatch.setenv("ALEO_ACCOUNT_PASSWORD", PASSWORD)
    assert keystore.load(dir=sandbox.dir).address == account.address
    assert sandbox.security.calls == []


def test_prompts_on_the_terminal_and_allows_three_attempts(sandbox: Sandbox) -> None:
    account = new_account()
    keystore.save(account.private_key, dir=sandbox.dir, storage="encrypted", password=PASSWORD)
    sandbox.terminal = True

    sandbox.answers.extend([OTHER_PASSWORD, "", PASSWORD])
    loaded = keystore.load(dir=sandbox.dir)
    assert loaded.address == account.address
    assert len(sandbox.prompts) == 3
    assert account.address in sandbox.prompts[0]
    assert "default" in sandbox.prompts[0]
    assert sandbox.notes == ["Wrong password.", "The password must not be empty."]

    sandbox.prompts.clear()
    sandbox.answers.extend([OTHER_PASSWORD, OTHER_PASSWORD, OTHER_PASSWORD, PASSWORD])
    assert code_of(lambda: keystore.load(dir=sandbox.dir)) == "WRONG_PASSWORD"
    assert len(sandbox.prompts) == 3
    assert sandbox.answers == [PASSWORD]


def test_asks_twice_when_setting_a_password_at_the_terminal(sandbox: Sandbox) -> None:
    account = new_account()
    sandbox.terminal = True
    sandbox.answers.extend([PASSWORD, OTHER_PASSWORD, "", "", PASSWORD, PASSWORD])
    keystore.save(account.private_key, dir=sandbox.dir, storage="encrypted")
    assert len(sandbox.prompts) == 6
    assert sandbox.notes == ["The passwords do not match.", "The password must not be empty."]
    check_secret(keystore.export(dir=sandbox.dir, password=PASSWORD, prompt=False), account.private_key)

    sandbox.answers.extend([PASSWORD, OTHER_PASSWORD, PASSWORD, OTHER_PASSWORD, PASSWORD, OTHER_PASSWORD])
    assert (
        code_of(lambda: keystore.save(Account.random(), dir=sandbox.dir, name="second", storage="encrypted"))
        == "PASSWORD_REQUIRED"
    )
    assert not (sandbox.dir / "second.json").exists()


def test_reports_a_damaged_ciphertext_as_a_wrong_password(sandbox: Sandbox) -> None:
    account = new_account()
    stored = keystore.save(account.private_key, dir=sandbox.dir, storage="encrypted", password=PASSWORD)
    file = read_json(stored.path)
    file["key"]["ciphertext"] = "ciphertext1qqqq"
    sandbox.put("default", file)
    error = failure(lambda: keystore.load(dir=sandbox.dir, password=PASSWORD))
    assert error.code == "WRONG_PASSWORD"
    assert error.__cause__ is None and error.__context__ is None


def test_reports_a_ciphertext_for_another_account_as_a_wrong_password(sandbox: Sandbox) -> None:
    account = new_account()
    stored = keystore.save(account.private_key, dir=sandbox.dir, storage="encrypted", password=PASSWORD)
    file = read_json(stored.path)
    file["address"] = SAMPLE.address
    sandbox.put("default", file)
    assert code_of(lambda: keystore.load(dir=sandbox.dir, password=PASSWORD)) == "WRONG_PASSWORD"


# ---------------------------------------------------------------------------
# Keychain storage
# ---------------------------------------------------------------------------


def test_writes_the_key_through_security_i_and_never_as_a_process_argument(sandbox: Sandbox) -> None:
    account = new_account()
    stored = keystore.save(account.private_key, dir=sandbox.dir, storage="keychain")
    assert stored.storage == "keychain"
    calls = sandbox.security.calls
    assert len(calls) == 2
    assert calls[0]["args"] == ["-i"]
    check_secret(calls[0]["stdin"], f"add-generic-password -U -s aleo-account-key -a {account.address} -l aleo-account-key:default -w {account.private_key}\n")
    # The write is confirmed by reading the entry back.
    assert calls[1] == {
        "args": ["find-generic-password", "-s", "aleo-account-key", "-a", account.address, "-w"],
        "stdin": None,
    }
    for call in calls:
        assert not any("APrivateKey1" in argument for argument in call["args"])

    text = stored.path.read_text(encoding="utf-8")
    assert "APrivateKey1" not in text
    assert json.loads(text) == {
        "version": 1,
        "address": account.address,
        "key": {"type": "keychain", "service": "aleo-account-key", "account": account.address},
    }


def test_labels_each_entry_with_the_account_name_and_still_finds_it_by_address(sandbox: Sandbox) -> None:
    first = new_account()
    keystore.save(first.private_key, dir=sandbox.dir, name="trading", storage="keychain")
    assert sandbox.security.labels[f"{KEY_SERVICE} {first.address}"] == "aleo-account-key:trading"

    second = new_account()
    keystore.save(second.private_key, dir=sandbox.dir, name="bot_1-a", storage="encrypted", password=PASSWORD, save_password=True)
    assert sandbox.security.labels[f"{PASSWORD_SERVICE} {second.address}"] == "aleo-account-password:bot_1-a"

    # Changing the storage writes the new entry under the same account name.
    keystore.protect(dir=sandbox.dir, name="bot_1-a", storage="keychain", password=PASSWORD)
    assert sandbox.security.labels[f"{KEY_SERVICE} {second.address}"] == "aleo-account-key:bot_1-a"
    keystore.protect(dir=sandbox.dir, name="trading", storage="encrypted", password=PASSWORD, save_password=True)
    assert sandbox.security.labels[f"{PASSWORD_SERVICE} {first.address}"] == "aleo-account-password:trading"

    # The label plays no part in the lookup: reads name only the service and the address.
    sandbox.security.calls.clear()
    keystore.load(dir=sandbox.dir, name="bot_1-a")
    assert sandbox.security.calls == [
        {"args": ["find-generic-password", "-s", "aleo-account-key", "-a", second.address, "-w"], "stdin": None}
    ]


def test_reads_the_key_back_from_the_keychain(sandbox: Sandbox) -> None:
    account = new_account()
    keystore.save(account.private_key, dir=sandbox.dir, storage="keychain")
    sandbox.security.calls.clear()
    loaded = keystore.load(dir=sandbox.dir)
    assert loaded.source == "account-file"
    check_secret(loaded.private_key, account.private_key)
    assert sandbox.security.calls == [
        {"args": ["find-generic-password", "-s", "aleo-account-key", "-a", account.address, "-w"], "stdin": None}
    ]


def test_maps_exit_status_44_to_item_not_found_and_other_failures_to_keychain_failed(sandbox: Sandbox) -> None:
    account = new_account()
    keystore.save(account.private_key, dir=sandbox.dir, storage="keychain")
    sandbox.security.entries.clear()
    missing = failure(lambda: keystore.load(dir=sandbox.dir))
    assert missing.code == "KEYCHAIN_ITEM_NOT_FOUND"
    assert account.address in str(missing)

    sandbox.security.force(lambda args: (51, ""))
    assert code_of(lambda: keystore.load(dir=sandbox.dir)) == "KEYCHAIN_FAILED"
    # A timeout or a tool that cannot start has no exit status.
    sandbox.security.force(lambda args: (None, ""))
    assert code_of(lambda: keystore.load(dir=sandbox.dir)) == "KEYCHAIN_FAILED"


def test_fails_with_address_mismatch_when_the_keychain_entry_holds_another_key(sandbox: Sandbox) -> None:
    account = new_account()
    other = new_account()
    keystore.save(account.private_key, dir=sandbox.dir, storage="keychain")
    sandbox.security.entries[f"{KEY_SERVICE} {account.address}"] = other.private_key
    error = failure(lambda: keystore.load(dir=sandbox.dir))
    assert error.code == "ADDRESS_MISMATCH"
    assert account.address in str(error)
    assert other.address in str(error)
    absent(str(error), other.private_key)


def test_writes_no_file_when_the_keychain_write_fails(sandbox: Sandbox) -> None:
    sandbox.security.force(lambda args: (1, "") if args[0] == "-i" else None)
    assert code_of(lambda: keystore.save(Account.random(), dir=sandbox.dir, storage="keychain")) == "KEYCHAIN_FAILED"
    assert not (sandbox.dir / "default.json").exists()


def test_fails_when_the_entry_does_not_hold_the_key_after_the_write(sandbox: Sandbox) -> None:
    sandbox.security.force(lambda args: (0, "") if args[0] == "-i" else None)
    assert code_of(lambda: keystore.save(Account.random(), dir=sandbox.dir, storage="keychain")) == "KEYCHAIN_FAILED"
    assert not (sandbox.dir / "default.json").exists()


def test_never_echoes_the_security_tools_output_in_an_error(sandbox: Sandbox) -> None:
    account = new_account()
    keystore.save(account.private_key, dir=sandbox.dir, storage="keychain")
    sandbox.security.force(lambda args: (1, f"{account.private_key}\n"))
    error = failure(lambda: keystore.load(dir=sandbox.dir))
    assert error.code == "KEYCHAIN_FAILED"
    absent(str(error), account.private_key)


def test_refuses_keychain_storage_outside_macos_before_anything_is_written(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime, "platform", "linux")
    account = new_account()
    error = failure(lambda: keystore.save(account.private_key, dir=sandbox.dir, storage="keychain"))
    assert error.code == "KEYCHAIN_UNAVAILABLE"
    assert not sandbox.dir.exists()
    assert sandbox.security.calls == []

    keystore.save(account.private_key, dir=sandbox.dir)
    assert code_of(lambda: keystore.protect(dir=sandbox.dir, storage="keychain")) == "KEYCHAIN_UNAVAILABLE"
    stored = keystore.info(dir=sandbox.dir)
    assert stored is not None and stored.storage == "plaintext"


def test_fails_to_load_a_keychain_file_outside_macos_and_never_falls_back(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    account = new_account()
    keystore.save(account.private_key, dir=sandbox.dir, storage="keychain")
    monkeypatch.setattr(runtime, "platform", "linux")
    sandbox.security.calls.clear()

    stored = keystore.info(dir=sandbox.dir)
    assert stored is not None and stored.storage == "keychain"
    error = failure(lambda: keystore.load(dir=sandbox.dir, legacy_key=account.private_key))
    assert error.code == "KEYCHAIN_UNAVAILABLE"
    assert re.search(r"[Ee]xport", str(error))
    assert re.search(r"import", str(error))
    assert code_of(lambda: keystore.export(dir=sandbox.dir)) == "KEYCHAIN_UNAVAILABLE"
    assert sandbox.security.calls == []

    # A key from the environment needs no Keychain; the file still checks it.
    monkeypatch.setenv("ALEO_PRIVATE_KEY", account.private_key)
    assert keystore.load(dir=sandbox.dir).source == "env"


# ---------------------------------------------------------------------------
# protect
# ---------------------------------------------------------------------------


def test_moves_an_account_through_every_storage_and_removes_each_old_secret(sandbox: Sandbox) -> None:
    account = new_account()
    key_entry = f"{KEY_SERVICE} {account.address}"
    password_entry = f"{PASSWORD_SERVICE} {account.address}"
    keystore.save(account.private_key, dir=sandbox.dir)

    # plaintext -> keychain
    stored = keystore.protect(dir=sandbox.dir, storage="keychain")
    assert stored == StoredAccount(
        name="default", path=sandbox.dir / "default.json", address=account.address, storage="keychain"
    )
    assert "APrivateKey1" not in stored.path.read_text(encoding="utf-8")
    check_secret(sandbox.security.entries[key_entry], account.private_key)
    check_secret(keystore.export(dir=sandbox.dir), account.private_key)

    # keychain -> encrypted, password saved: the key entry goes, the password entry appears.
    stored = keystore.protect(dir=sandbox.dir, storage="encrypted", password=PASSWORD, save_password=True, prompt=False)
    assert stored.storage == "encrypted"
    assert key_entry not in sandbox.security.entries
    assert sandbox.security.entries.get(password_entry) == PASSWORD.encode("utf-8").hex()
    check_secret(keystore.export(dir=sandbox.dir, prompt=False), account.private_key)

    # encrypted -> plaintext: the saved password goes.
    stored = keystore.protect(dir=sandbox.dir, storage="plaintext", prompt=False)
    assert stored.storage == "plaintext"
    assert sandbox.security.entries == {}
    check_secret(keystore.export(dir=sandbox.dir), account.private_key)
    if POSIX:
        assert mode(stored.path) == 0o600
    assert os.listdir(sandbox.dir) == ["default.json"]


def test_sets_a_new_password_and_a_new_salt_on_an_encrypted_account(sandbox: Sandbox) -> None:
    account = new_account()
    stored = keystore.save(account.private_key, dir=sandbox.dir, storage="encrypted", password=PASSWORD, save_password=True)
    before = read_json(stored.path)

    keystore.protect(dir=sandbox.dir, storage="encrypted", password=PASSWORD, new_password=OTHER_PASSWORD, prompt=False)
    after = read_json(stored.path)
    assert after["key"]["kdfParams"]["salt"] != before["key"]["kdfParams"]["salt"]
    assert after["key"]["ciphertext"] != before["key"]["ciphertext"]
    # save_password was not repeated, so the saved (old) password is gone.
    assert sandbox.security.entries == {}
    assert code_of(lambda: keystore.export(dir=sandbox.dir, password=PASSWORD)) == "WRONG_PASSWORD"
    check_secret(keystore.export(dir=sandbox.dir, password=OTHER_PASSWORD), account.private_key)


def test_takes_a_replacement_password_from_new_password_or_the_prompt_only(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    account = new_account()
    keystore.save(account.private_key, dir=sandbox.dir, storage="encrypted", password=PASSWORD)
    monkeypatch.setenv("ALEO_ACCOUNT_PASSWORD", PASSWORD)

    # The variable holds the current password and is not reused as the new one.
    assert code_of(lambda: keystore.protect(dir=sandbox.dir, storage="encrypted")) == "PASSWORD_REQUIRED"
    check_secret(keystore.export(dir=sandbox.dir), account.private_key)

    sandbox.terminal = True
    assert code_of(lambda: keystore.protect(dir=sandbox.dir, storage="encrypted", prompt=False)) == "PASSWORD_REQUIRED"
    sandbox.answers.extend([OTHER_PASSWORD, OTHER_PASSWORD])
    keystore.protect(dir=sandbox.dir, storage="encrypted", save_password=True)
    assert len(sandbox.prompts) == 2
    assert sandbox.security.entries.get(f"{PASSWORD_SERVICE} {account.address}") == OTHER_PASSWORD.encode("utf-8").hex()
    monkeypatch.setenv("ALEO_ACCOUNT_PASSWORD", OTHER_PASSWORD)
    check_secret(keystore.export(dir=sandbox.dir), account.private_key)


def test_sets_the_first_password_from_new_password_password_then_the_variable(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    account = new_account()
    keystore.save(account.private_key, dir=sandbox.dir)
    monkeypatch.setenv("ALEO_ACCOUNT_PASSWORD", PASSWORD)
    keystore.protect(dir=sandbox.dir, storage="encrypted", prompt=False)
    monkeypatch.delenv("ALEO_ACCOUNT_PASSWORD")
    check_secret(keystore.export(dir=sandbox.dir, password=PASSWORD), account.private_key)

    keystore.protect(dir=sandbox.dir, storage="plaintext", password=PASSWORD)
    keystore.protect(dir=sandbox.dir, storage="encrypted", password=PASSWORD, new_password=OTHER_PASSWORD)
    check_secret(keystore.export(dir=sandbox.dir, password=OTHER_PASSWORD), account.private_key)

    keystore.protect(dir=sandbox.dir, storage="plaintext", password=OTHER_PASSWORD)
    assert code_of(lambda: keystore.protect(dir=sandbox.dir, storage="encrypted", prompt=False)) == "PASSWORD_REQUIRED"
    stored = keystore.info(dir=sandbox.dir)
    assert stored is not None and stored.storage == "plaintext"

    # With no new_password, the password argument is the one to set.
    keystore.protect(dir=sandbox.dir, storage="encrypted", password=PASSWORD, prompt=False)
    check_secret(keystore.export(dir=sandbox.dir, password=PASSWORD), account.private_key)


def test_keeps_the_account_usable_when_the_current_password_is_wrong(sandbox: Sandbox) -> None:
    account = new_account()
    stored = keystore.save(account.private_key, dir=sandbox.dir, storage="encrypted", password=PASSWORD)
    data = stored.path.read_bytes()
    assert code_of(lambda: keystore.protect(dir=sandbox.dir, storage="plaintext", password=OTHER_PASSWORD)) == "WRONG_PASSWORD"
    assert stored.path.read_bytes() == data


def test_reports_a_wrong_current_password_before_a_missing_new_one(sandbox: Sandbox) -> None:
    account = new_account()
    keystore.save(account.private_key, dir=sandbox.dir, storage="encrypted", password=PASSWORD)
    assert (
        code_of(lambda: keystore.protect(dir=sandbox.dir, storage="encrypted", password=OTHER_PASSWORD, prompt=False))
        == "WRONG_PASSWORD"
    )
    assert (
        code_of(lambda: keystore.protect(dir=sandbox.dir, storage="encrypted", password=PASSWORD, prompt=False))
        == "PASSWORD_REQUIRED"
    )


def test_leaves_a_saved_password_alone_until_the_new_storage_reads_back(sandbox: Sandbox) -> None:
    account = new_account()
    keystore.save(account.private_key, dir=sandbox.dir, storage="encrypted", password=PASSWORD, save_password=True)
    saved = dict(sandbox.security.entries)
    data = (sandbox.dir / "default.json").read_bytes()
    sandbox.security.force(lambda args: (1, "") if args[0] == "-i" else None)
    assert code_of(lambda: keystore.protect(dir=sandbox.dir, storage="keychain", prompt=False)) == "KEYCHAIN_FAILED"
    assert sandbox.security.entries == saved
    assert (sandbox.dir / "default.json").read_bytes() == data
    sandbox.security.force(None)
    check_secret(keystore.export(dir=sandbox.dir, prompt=False), account.private_key)


def test_returns_without_rewriting_when_the_storage_is_already_in_place(sandbox: Sandbox) -> None:
    account = new_account()
    stored = keystore.save(account.private_key, dir=sandbox.dir)
    before = os.stat(stored.path)
    assert keystore.protect(dir=sandbox.dir, storage="plaintext") == stored
    assert os.stat(stored.path).st_ino == before.st_ino

    keystore.protect(dir=sandbox.dir, storage="keychain")
    sandbox.security.calls.clear()
    assert keystore.protect(dir=sandbox.dir, storage="keychain").storage == "keychain"
    assert sandbox.security.calls == []


def test_does_not_move_an_account_to_none_and_needs_a_stored_key(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    account = new_account()
    keystore.save(account.private_key, dir=sandbox.dir)
    assert code_of(lambda: keystore.protect(dir=sandbox.dir, storage="none")) == "UNSUPPORTED_STORAGE"  # type: ignore[arg-type]
    stored = keystore.info(dir=sandbox.dir)
    assert stored is not None and stored.storage == "plaintext"

    keystore.save(account.private_key, dir=sandbox.dir, name="address-only", storage="none")
    monkeypatch.setenv("ALEO_PRIVATE_KEY", account.private_key)
    assert code_of(lambda: keystore.protect(dir=sandbox.dir, name="address-only", storage="plaintext")) == "KEY_NOT_STORED"
    assert code_of(lambda: keystore.protect(dir=sandbox.dir, name="absent", storage="plaintext")) == "ACCOUNT_NOT_FOUND"


def test_keeps_the_old_secret_when_the_new_storage_cannot_be_written(sandbox: Sandbox) -> None:
    account = new_account()
    stored = keystore.save(account.private_key, dir=sandbox.dir)
    data = stored.path.read_bytes()
    sandbox.security.force(lambda args: (1, "") if args[0] == "-i" else None)
    assert code_of(lambda: keystore.protect(dir=sandbox.dir, storage="keychain")) == "KEYCHAIN_FAILED"
    assert stored.path.read_bytes() == data


def _vanishing_entry() -> Callable[[Sequence[str]], Optional[Any]]:
    """The entry is written and confirmed, then disappears before the check."""
    finds: List[int] = []

    def override(args: Sequence[str]) -> Optional[Any]:
        if args[0] != "find-generic-password":
            return None
        finds.append(1)
        return (44, "") if len(finds) >= 2 else None

    return override


def test_puts_the_original_file_back_when_the_new_storage_does_not_read_back(sandbox: Sandbox) -> None:
    account = new_account()
    stored = keystore.save(account.private_key, dir=sandbox.dir)
    data = stored.path.read_bytes()
    sandbox.security.force(_vanishing_entry())
    assert code_of(lambda: keystore.protect(dir=sandbox.dir, storage="keychain")) == "KEYCHAIN_ITEM_NOT_FOUND"
    assert stored.path.read_bytes() == data
    if POSIX:
        assert mode(stored.path) == 0o600
    assert os.listdir(sandbox.dir) == ["default.json"]
    sandbox.security.force(None)
    check_secret(keystore.export(dir=sandbox.dir), account.private_key)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def test_returns_the_stored_key_and_never_reads_the_key_variables(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    account = new_account()
    keystore.save(account.private_key, dir=sandbox.dir)
    monkeypatch.setenv("ALEO_PRIVATE_KEY", new_account().private_key)
    check_secret(keystore.export(dir=sandbox.dir), account.private_key)


def test_fails_when_there_is_no_file_or_no_stored_key(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    account = new_account()
    monkeypatch.setenv("ALEO_PRIVATE_KEY", account.private_key)
    assert code_of(lambda: keystore.export(dir=sandbox.dir)) == "ACCOUNT_NOT_FOUND"
    keystore.save(account.private_key, dir=sandbox.dir, storage="none")
    assert code_of(lambda: keystore.export(dir=sandbox.dir)) == "KEY_NOT_STORED"


# ---------------------------------------------------------------------------
# migrate
# ---------------------------------------------------------------------------


def legacy_content(field: str, key: str) -> Dict[str, Any]:
    return {
        "address": "placeholder",
        field: key,
        "apiUrl": "https://example.test",
        "accessRedeemed": True,
        "nested": {"b": 1, "a": [1, 2]},
    }


def legacy_text(field: str, key: str) -> str:
    return json.dumps(legacy_content(field, key), indent=2) + "\n"


@pytest.mark.parametrize("field", ["privateKey", "private_key"])
def test_copies_the_field_into_an_account_file_and_keeps_the_legacy_file(sandbox: Sandbox, field: str) -> None:
    account = new_account()
    source = sandbox.root / "state.json"
    source.write_text(legacy_text(field, account.private_key), encoding="utf-8")
    before = source.read_bytes()

    migrated = keystore.migrate(source, dir=sandbox.dir)
    assert migrated == StoredAccount(
        name="default", path=sandbox.dir / "default.json", address=account.address, storage="plaintext"
    )
    assert source.read_bytes() == before
    check_secret(keystore.export(dir=sandbox.dir), account.private_key)


def test_prefers_the_private_key_field_when_a_legacy_file_has_both(sandbox: Sandbox) -> None:
    first = new_account()
    second = new_account()
    source = sandbox.root / "state.json"
    source.write_text(json.dumps({"private_key": second.private_key, "privateKey": first.private_key}), encoding="utf-8")
    assert keystore.migrate(str(source), dir=sandbox.dir, remove_legacy_key=True).address == first.address
    # Only the field that was used is stripped.
    assert read_json(source) == {"private_key": second.private_key}


def test_strips_the_key_field_and_keeps_the_other_fields_their_order_and_the_indentation(sandbox: Sandbox) -> None:
    account = new_account()
    source = sandbox.root / "state.json"
    source.write_text(legacy_text("privateKey", account.private_key), encoding="utf-8")
    source.chmod(0o644)

    migrated = keystore.migrate(source, dir=sandbox.dir, storage="encrypted", password=PASSWORD, remove_legacy_key=True)
    assert migrated.storage == "encrypted"
    # What JSON.stringify(rest, null, 2) + "\n" produces for the remaining fields.
    expected = (
        "{\n"
        '  "address": "placeholder",\n'
        '  "apiUrl": "https://example.test",\n'
        '  "accessRedeemed": true,\n'
        '  "nested": {\n'
        '    "b": 1,\n'
        '    "a": [\n'
        "      1,\n"
        "      2\n"
        "    ]\n"
        "  }\n"
        "}\n"
    )
    assert source.read_text(encoding="utf-8") == expected
    if POSIX:
        assert mode(source) == 0o600
    assert sorted(os.listdir(sandbox.root)) == ["accounts", "home", "state.json"]
    check_secret(keystore.export(dir=sandbox.dir, password=PASSWORD), account.private_key)

    # The key is gone, so a second run finds nothing to migrate.
    assert code_of(lambda: keystore.migrate(source, dir=sandbox.dir)) == "LEGACY_KEY_NOT_FOUND"


def test_keeps_four_space_tab_one_space_and_single_line_layouts(sandbox: Sandbox) -> None:
    account = new_account()
    key = account.private_key
    # Each pair is the file before and what JSON.stringify writes without the key.
    cases = [
        (
            '{\n    "network": "testnet",\n    "private_key": "' + key + '",\n    "endpoint": "https://example.test"\n}',
            '{\n    "network": "testnet",\n    "endpoint": "https://example.test"\n}',
        ),
        (
            '{\n\t"network": "testnet",\n\t"private_key": "' + key + '",\n\t"endpoint": "https://example.test"\n}\n',
            '{\n\t"network": "testnet",\n\t"endpoint": "https://example.test"\n}\n',
        ),
        (
            '{"network":"testnet","private_key":"' + key + '","endpoint":"https://example.test"}',
            '{"network":"testnet","endpoint":"https://example.test"}',
        ),
        (
            '{\n "network": "testnet",\n "private_key": "' + key + '",\n "endpoint": "https://example.test"\n}\n',
            '{\n "network": "testnet",\n "endpoint": "https://example.test"\n}\n',
        ),
        (
            # Written by Python's json.dumps on one line: re-rendered without spaces.
            '{"network": "testnet", "private_key": "' + key + '", "endpoint": "https://example.test"}\n',
            '{"network":"testnet","endpoint":"https://example.test"}\n',
        ),
        (
            # Windows line endings select the layout; the rewrite uses LF.
            '{\r\n  "network": "testnet",\r\n  "private_key": "' + key + '"\r\n}\r\n',
            '{\n  "network": "testnet"\n}\n',
        ),
        (
            # Nothing but the key: an empty object, as JSON.stringify writes it.
            '{\n  "private_key": "' + key + '"\n}\n',
            "{}\n",
        ),
        (
            # Indentation beyond ten characters is cut to ten, the JSON.stringify limit.
            '{\n            "private_key": "' + key + '",\n            "a": [],\n            "b": {}\n}',
            '{\n          "a": [],\n          "b": {}\n}',
        ),
    ]
    for index, (before, after) in enumerate(cases):
        source = sandbox.root / f"profile-{index}.json"
        source.write_bytes(before.encode("utf-8"))
        keystore.migrate(source, dir=sandbox.dir, remove_legacy_key=True)
        assert source.read_bytes() == after.encode("utf-8"), f"layout {index}"


def test_keeps_text_outside_ascii_and_escapes_as_json_stringify_does(sandbox: Sandbox) -> None:
    account = new_account()
    source = sandbox.root / "state.json"
    # A non-ASCII letter stays as written; an escaped control character keeps
    # its short escape; an escaped lone surrogate keeps its escaped form.
    before = (
        '{"privateKey":"' + account.private_key + '","label":"caf\N{LATIN SMALL LETTER E WITH ACUTE} \\u00e9",'
        '"tab":"a\\tb\\u0001","lone":"\\ud83d","slash":"a\\/b"}'
    )
    after = (
        '{"label":"caf\N{LATIN SMALL LETTER E WITH ACUTE} \N{LATIN SMALL LETTER E WITH ACUTE}",'
        '"tab":"a\\tb\\u0001","lone":"\\ud83d","slash":"a/b"}'
    )
    source.write_bytes(before.encode("utf-8"))
    keystore.migrate(source, dir=sandbox.dir, remove_legacy_key=True)
    assert source.read_bytes() == after.encode("utf-8")


def test_follows_the_rules_of_save_for_an_existing_account_file(sandbox: Sandbox) -> None:
    account = new_account()
    source = sandbox.root / "state.json"
    source.write_text(legacy_text("privateKey", account.private_key), encoding="utf-8")

    keystore.save(Account.random(), dir=sandbox.dir)
    assert code_of(lambda: keystore.migrate(source, dir=sandbox.dir, remove_legacy_key=True)) == "ACCOUNT_EXISTS"
    assert account.private_key in source.read_text(encoding="utf-8")

    keystore.save(account.private_key, dir=sandbox.dir, name="mine")
    assert code_of(lambda: keystore.migrate(source, dir=sandbox.dir, name="mine", storage="keychain")) == "ACCOUNT_EXISTS"
    migrated = keystore.migrate(source, dir=sandbox.dir, name="mine", remove_legacy_key=True)
    assert migrated.storage == "plaintext"
    absent(source.read_text(encoding="utf-8"), account.private_key)


def test_keeps_the_legacy_key_when_the_new_account_does_not_read_back(sandbox: Sandbox) -> None:
    account = new_account()
    source = sandbox.root / "state.json"
    source.write_text(legacy_text("privateKey", account.private_key), encoding="utf-8")
    sandbox.security.force(_vanishing_entry())
    assert (
        code_of(lambda: keystore.migrate(source, dir=sandbox.dir, storage="keychain", remove_legacy_key=True))
        == "KEYCHAIN_ITEM_NOT_FOUND"
    )
    assert account.private_key in source.read_text(encoding="utf-8")


def test_does_not_strip_the_legacy_key_for_an_address_only_account(sandbox: Sandbox) -> None:
    account = new_account()
    source = sandbox.root / "state.json"
    source.write_text(legacy_text("privateKey", account.private_key), encoding="utf-8")
    assert (
        code_of(lambda: keystore.migrate(source, dir=sandbox.dir, storage="none", remove_legacy_key=True))
        == "KEY_NOT_STORED"
    )
    assert not sandbox.dir.exists()
    assert keystore.migrate(source, dir=sandbox.dir, storage="none").storage == "none"
    assert account.private_key in source.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(None, id="a missing file"),
        pytest.param('{"privateKey": "APrivateKey1', id="text that is not JSON"),
        pytest.param("[]", id="an array"),
        pytest.param('{"address": "aleo1x"}', id="a file without a key field"),
        pytest.param('{"privateKey": ""}', id="an empty key field"),
        pytest.param('{"account": {"privateKey": "APrivateKey1x"}}', id="a nested key field"),
        pytest.param('{"private_key": 7}', id="a key field that is not a string"),
        pytest.param(b'{"privateKey": "APrivateKey1\xff"}', id="bytes that are not UTF-8"),
    ],
)
def test_fails_with_legacy_key_not_found(sandbox: Sandbox, content: Any) -> None:
    source = sandbox.root / "state.json"
    if content is not None:
        source.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    error = failure(lambda: keystore.migrate(source, dir=sandbox.dir))
    assert error.code == "LEGACY_KEY_NOT_FOUND"
    assert str(source) in str(error)
    assert "APrivateKey1" not in str(error)
    assert error.__context__ is None or isinstance(error.__context__, OSError)
    assert not sandbox.dir.exists()


def test_fails_with_invalid_key_for_a_legacy_key_that_cannot_be_parsed(sandbox: Sandbox) -> None:
    source = sandbox.root / "state.json"
    source.write_text('{"privateKey": "APrivateKey1nope"}', encoding="utf-8")
    assert code_of(lambda: keystore.migrate(source, dir=sandbox.dir)) == "INVALID_KEY"


# ---------------------------------------------------------------------------
# The terminal and the security tool
# ---------------------------------------------------------------------------


def test_hidden_prompt_reads_a_line_through_the_terminal_and_not_standard_output(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: List[Any] = []

    def fake_getpass(prompt: str = "", stream: Any = None) -> str:
        seen.append((prompt, stream))
        return "typed"

    monkeypatch.setattr(getpass, "getpass", fake_getpass)
    assert sandbox.real.prompt_hidden("Password: ") == "typed"
    assert seen == [("Password: ", sys.stderr)]
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("interruption", [EOFError, KeyboardInterrupt])
def test_hidden_prompt_fails_with_password_required_when_cancelled_or_when_the_input_ends(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch, interruption: type
) -> None:
    def fake_getpass(prompt: str = "", stream: Any = None) -> str:
        raise interruption()

    monkeypatch.setattr(getpass, "getpass", fake_getpass)
    error = failure(lambda: sandbox.real.prompt_hidden("Password: "))
    assert error.code == "PASSWORD_REQUIRED"


def test_notes_go_to_standard_error(sandbox: Sandbox, capsys: pytest.CaptureFixture[str]) -> None:
    sandbox.real.notify("Wrong password.")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Wrong password.\n"


def test_no_terminal_is_attached_when_the_streams_are_not_terminals(sandbox: Sandbox, capsys: pytest.CaptureFixture[str]) -> None:
    # Under capture neither stream is a terminal.
    assert sandbox.real.terminal_attached() is False
    assert sandbox.real.output_is_terminal() is False


@posix_only
def test_runs_the_security_tool_by_its_full_path_with_the_command_on_standard_input(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert runtime_module.SECURITY_PATH == "/usr/bin/security"
    assert runtime_module.SECURITY_TIMEOUT_SECONDS == 15
    tool = sandbox.root / "tool"
    tool.write_text('#!/bin/sh\nprintf "%s|" "$@"\ncat\necho "noise" >&2\nexit 7\n', encoding="utf-8")
    tool.chmod(0o700)
    monkeypatch.setattr(runtime_module, "SECURITY_PATH", str(tool))
    assert sandbox.real.security(["-i"], "add-generic-password -U\n") == (7, "-i|add-generic-password -U\n")
    assert sandbox.real.security(["find-generic-password", "-w"], None) == (7, "find-generic-password|-w|")


@posix_only
def test_reports_a_tool_that_cannot_start_hangs_or_is_killed_as_having_no_status(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_module, "SECURITY_PATH", str(sandbox.root / "absent"))
    assert sandbox.real.security(["-i"], None) == (None, "")

    slow = sandbox.root / "slow"
    slow.write_text("#!/bin/sh\nsleep 5\n", encoding="utf-8")
    slow.chmod(0o700)
    monkeypatch.setattr(runtime_module, "SECURITY_PATH", str(slow))
    monkeypatch.setattr(runtime_module, "SECURITY_TIMEOUT_SECONDS", 0.2)
    assert sandbox.real.security(["-i"], None) == (None, "")

    killed = sandbox.root / "killed"
    killed.write_text("#!/bin/sh\nkill -9 $$\n", encoding="utf-8")
    killed.chmod(0o700)
    monkeypatch.setattr(runtime_module, "SECURITY_PATH", str(killed))
    assert sandbox.real.security(["-i"], None) == (None, "")


def test_refuses_a_keychain_value_that_cannot_be_passed_unquoted(sandbox: Sandbox) -> None:
    from aleo.keystore._keychain import write_keychain

    for address, value in (("aleo1x", "two words"), ("aleo1 x", "value"), ("aleo1x", "quote\"d"), ("aleo1x", "")):
        assert code_of(lambda: write_keychain(KEY_SERVICE, address, value, "default")) == "KEYCHAIN_FAILED"
    # The account name goes into the label unquoted, so it is held to the same rule.
    for name in ("two words", "quote\"d", "", "Upper", "a\n"):
        assert code_of(lambda: write_keychain(KEY_SERVICE, "aleo1x", "value", name)) == "KEYCHAIN_FAILED"
    assert sandbox.security.calls == []


# ---------------------------------------------------------------------------
# The real macOS Keychain
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("ALEO_KEYSTORE_KEYCHAIN_TEST") != "1" or sys.platform != "darwin",
    reason="set ALEO_KEYSTORE_KEYCHAIN_TEST=1 on macOS to exercise the real Keychain",
)
def test_stores_reads_and_removes_the_sample_account_through_the_real_keychain(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Uses the public sample account only and deletes both of its entries afterwards."""
    # Undo the stand-ins for this test only.
    monkeypatch.setattr(runtime, "platform", sandbox.real.platform)
    monkeypatch.setattr(runtime, "security", sandbox.real.security)
    # The security tool finds the login keychain through HOME. Account files
    # still go to the temporary directory passed as ``dir``.
    if sandbox.real_home is not None:
        monkeypatch.setenv("HOME", sandbox.real_home)

    def leftovers() -> List[bool]:
        return [
            read_keychain(KEY_SERVICE, SAMPLE.address) is None,
            read_keychain(PASSWORD_SERVICE, SAMPLE.address) is None,
        ]

    # Checked before the cleanup below is armed: entries that are already
    # there belong to someone else's run and are left alone.
    assert leftovers() == [True, True], "the sample account already has Keychain entries; retry later"
    password = VECTORS["encrypted"]["password"]
    try:
        stored = keystore.save(SAMPLE.private_key, dir=sandbox.dir, name="keychain", storage="keychain")
        assert stored.path.read_bytes() == (VECTORS_DIR / "keychain.json").read_bytes()
        loaded = keystore.load(dir=sandbox.dir, name="keychain", prompt=False)
        assert loaded.source == "account-file"
        assert loaded.address == SAMPLE.address
        check_secret(loaded.private_key, SAMPLE.private_key)

        # Move to an encrypted file with the password saved: the key entry goes.
        keystore.protect(
            dir=sandbox.dir, name="keychain", storage="encrypted", password=password, save_password=True, prompt=False
        )
        assert read_keychain(KEY_SERVICE, SAMPLE.address) is None
        assert read_keychain(PASSWORD_SERVICE, SAMPLE.address) == VECTORS["keychain"]["passwordValue"]
        check_secret(keystore.export(dir=sandbox.dir, name="keychain", prompt=False), SAMPLE.private_key)

        # Back to plaintext: the saved password goes too.
        keystore.protect(dir=sandbox.dir, name="keychain", storage="plaintext", prompt=False)
        assert leftovers() == [True, True]
    finally:
        delete_keychain(KEY_SERVICE, SAMPLE.address)
        delete_keychain(PASSWORD_SERVICE, SAMPLE.address)
    assert leftovers() == [True, True]
