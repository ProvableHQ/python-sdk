"""Check that profiles retain the intended account without exposing its key.

The account_sandbox fixture supplies temporary files and a fake Keychain.
Accounts are generated for tests and never funded. Use same_secret for key
comparisons so pytest does not include private keys in failure output.
"""
from __future__ import annotations

import json
import logging
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import Mock

import pytest
from aleo import keystore, testnet
from aleo.keystore import KeystoreError

from aleo_shield_swap import ShieldSwap
from aleo_shield_swap import profile as profile_module
from aleo_shield_swap.profile import DEFAULT_ENDPOINT, Profile

# Fixed passwords for unfunded test accounts.
PASSWORD = "correct horse battery staple"
OTHER_PASSWORD = "another test password"


def new_account() -> SimpleNamespace:
    """Generate an unfunded account for a test."""
    key = testnet.PrivateKey.random()
    return SimpleNamespace(address=str(key.address), private_key=str(key))


def same_secret(actual: object, expected: object) -> None:
    """Fail on unequal secrets without printing either value."""
    if not isinstance(actual, str) or actual != expected:
        raise AssertionError("the secret values differ")


def failure(call: Callable[[], Any]) -> KeystoreError:
    """Capture a failed call so the test can inspect its error code."""
    try:
        call()
    except KeystoreError as error:
        return error
    raise AssertionError("expected the call to fail")


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def files_under(*roots: Path) -> list[Path]:
    return [path for root in roots if root.exists() for path in root.rglob("*") if path.is_file()]


def assert_key_on_no_disk(private_key: str, *roots: Path) -> None:
    """Check for leaked private keys in files below the supplied directories."""
    for path in files_under(*roots):
        if private_key.encode() in path.read_bytes():
            raise AssertionError(f"{path} contains a private key")


def saved_profile(home: Path) -> dict[str, Any]:
    return json.loads((home / "profile.json").read_text())


def write_legacy_profile(home: Path, account: SimpleNamespace, **extra: Any) -> Path:
    """Create a profile in the older format for compatibility and migration tests."""
    home.mkdir(parents=True, exist_ok=True)
    path = home / "profile.json"
    data = {"address": account.address, "private_key": account.private_key,
            "network": "testnet", "endpoint": DEFAULT_ENDPOINT, **extra}
    path.write_text(json.dumps(data, indent=1))
    path.chmod(0o600)
    return path


@pytest.fixture(autouse=True)
def fresh_legacy_notices(monkeypatch):
    """Let each test observe its own migration warnings."""
    monkeypatch.setattr(profile_module, "_legacy_notices", set())


@pytest.fixture
def lookups(monkeypatch):
    """Record lookup arguments to check account selection and key caching."""
    calls: list[dict[str, Any]] = []
    real = keystore.load

    def counted(**kwargs: Any):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(keystore, "load", counted)
    return calls


# Test isolation

def test_tests_run_against_a_stand_in_home(account_sandbox, tmp_path):
    """Verify that the fixture redirects default paths and clears account variables."""
    import os

    assert Path.home() == account_sandbox.home
    assert Profile.default_home() == account_sandbox.home / ".shield-swap"
    for name in ("ALEO_PRIVATE_KEY", "ALEO_PRIVATE_KEY_FILE", "ALEO_ACCOUNT", "ALEO_ACCOUNTS_DIR",
                 "ALEO_ACCOUNT_PASSWORD", "SHIELD_SWAP_PRIVATE_KEY", "SHIELD_SWAP_PRIVATE_KEY_FILE",
                 "SHIELD_SWAP_HOME"):
        assert name not in os.environ
    assert keystore.info() is None
    stored = keystore.save(new_account().private_key)
    assert stored.path == account_sandbox.accounts / "default.json"


# Creating a profile

def test_create_then_reload_is_stable(tmp_path):
    p1 = Profile.load_or_create(tmp_path / "home")
    assert p1.address.startswith("aleo1")
    assert p1.private_key.startswith("APrivateKey1")
    assert p1.network == "testnet"
    p2 = Profile.load_or_create(tmp_path / "home")     # reload, no regen
    assert p2.address == p1.address
    same_secret(p2.private_key, p1.private_key)


def test_new_profile_stores_the_key_in_the_account_file_not_the_profile(tmp_path, account_sandbox):
    profile = Profile.load_or_create(tmp_path / "home", network="mainnet", endpoint="https://node.example")
    saved = saved_profile(profile.home)
    assert saved == {"address": profile.address, "account": "default",
                     "network": "mainnet", "endpoint": "https://node.example"}
    assert_key_on_no_disk(profile.private_key, tmp_path)

    account_file = account_sandbox.accounts / "default.json"
    stored = json.loads(account_file.read_text())
    assert stored["address"] == profile.address
    assert stored["key"]["type"] == "plaintext"
    same_secret(stored["key"]["privateKey"], profile.private_key)
    assert mode(account_file) == 0o600
    assert keystore.info() == keystore.StoredAccount(
        name="default", path=account_file, address=profile.address, storage="plaintext")


def test_profile_file_is_owner_only(tmp_path):
    p = Profile.load_or_create(tmp_path / "home")
    assert mode(p.home / "profile.json") == 0o600


def test_loose_profile_mode_is_tightened_on_load(tmp_path):
    p = Profile.load_or_create(tmp_path / "home")
    (p.home / "profile.json").chmod(0o644)
    Profile.load_or_create(tmp_path / "home")
    assert mode(p.home / "profile.json") == 0o600


def test_new_profile_adopts_the_stored_account(tmp_path, account_sandbox):
    account = new_account()
    keystore.save(account.private_key)
    before = (account_sandbox.accounts / "default.json").read_bytes()
    profile = Profile.load_or_create(tmp_path / "home")
    assert profile.address == account.address
    same_secret(profile.private_key, account.private_key)
    assert (account_sandbox.accounts / "default.json").read_bytes() == before
    assert saved_profile(profile.home)["account"] == "default"


def test_two_homes_share_the_stored_account(tmp_path):
    first = Profile.load_or_create(tmp_path / "one")
    second = Profile.load_or_create(tmp_path / "two")
    assert second.address == first.address


@pytest.mark.parametrize("variable", ["ALEO_PRIVATE_KEY", "SHIELD_SWAP_PRIVATE_KEY"])
def test_key_from_the_environment_is_used_and_never_written(tmp_path, monkeypatch, account_sandbox, variable):
    account = new_account()
    monkeypatch.setenv(variable, account.private_key)
    profile = Profile.load_or_create(tmp_path / "home")
    assert profile.address == account.address
    same_secret(profile.private_key, account.private_key)
    assert saved_profile(profile.home) == {"address": account.address, "account": "default",
                                           "network": "testnet", "endpoint": DEFAULT_ENDPOINT}
    assert not account_sandbox.accounts.exists()
    assert_key_on_no_disk(account.private_key, tmp_path, account_sandbox.home)


@pytest.mark.parametrize("variable", ["ALEO_PRIVATE_KEY_FILE", "SHIELD_SWAP_PRIVATE_KEY_FILE"])
def test_key_file_from_the_environment_is_used_and_never_copied(tmp_path, monkeypatch, account_sandbox, variable):
    account = new_account()
    key_file = tmp_path / "key.txt"
    key_file.write_text(account.private_key + "\n")
    monkeypatch.setenv(variable, str(key_file))
    profile = Profile.load_or_create(tmp_path / "home")
    assert profile.address == account.address
    same_secret(profile.private_key, account.private_key)
    assert not account_sandbox.accounts.exists()
    assert_key_on_no_disk(account.private_key, tmp_path / "home", account_sandbox.home)


def test_profile_made_from_an_environment_key_needs_it_again(tmp_path, monkeypatch):
    account = new_account()
    monkeypatch.setenv("ALEO_PRIVATE_KEY", account.private_key)
    Profile.load_or_create(tmp_path / "home")
    monkeypatch.delenv("ALEO_PRIVATE_KEY")

    later = Profile.load_or_create(tmp_path / "home")          # The saved address is still available without the key.
    assert later.address == account.address
    assert failure(lambda: later.private_key).code == "ACCOUNT_NOT_FOUND"

    monkeypatch.setenv("ALEO_PRIVATE_KEY", account.private_key)
    same_secret(Profile.load_or_create(tmp_path / "home").private_key, account.private_key)


def test_environment_key_must_match_the_selected_account_file(tmp_path, monkeypatch):
    stored, supplied = new_account(), new_account()
    keystore.save(stored.private_key)
    monkeypatch.setenv("ALEO_PRIVATE_KEY", supplied.private_key)
    error = failure(lambda: Profile.load_or_create(tmp_path / "home"))
    assert error.code == "ADDRESS_MISMATCH"
    assert stored.address in str(error) and supplied.address in str(error)
    assert not (tmp_path / "home").exists()


@pytest.mark.parametrize("variable", ["ALEO_PRIVATE_KEY", "SHIELD_SWAP_PRIVATE_KEY"])
def test_empty_key_variable_is_an_error_not_a_new_account(tmp_path, monkeypatch, account_sandbox, variable):
    monkeypatch.setenv(variable, "")
    assert failure(lambda: Profile.load_or_create(tmp_path / "home")).code == "EMPTY_KEY_SOURCE"
    assert not (tmp_path / "home").exists()
    assert not account_sandbox.accounts.exists()


def test_empty_key_file_is_an_error_not_a_new_account(tmp_path, monkeypatch, account_sandbox):
    key_file = tmp_path / "key.txt"
    key_file.write_text("\n")
    monkeypatch.setenv("SHIELD_SWAP_PRIVATE_KEY_FILE", str(key_file))
    assert failure(lambda: Profile.load_or_create(tmp_path / "home")).code == "EMPTY_KEY_SOURCE"
    assert not (tmp_path / "home").exists()
    assert not account_sandbox.accounts.exists()


def test_variables_holding_different_keys_are_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ALEO_PRIVATE_KEY", new_account().private_key)
    monkeypatch.setenv("SHIELD_SWAP_PRIVATE_KEY", new_account().private_key)
    assert failure(lambda: Profile.load_or_create(tmp_path / "home")).code == "KEY_CONFLICT"
    assert not (tmp_path / "home").exists()


def test_create_false_raises_when_no_account_resolves(tmp_path, account_sandbox):
    error = failure(lambda: Profile.load_or_create(tmp_path / "home", create=False))
    assert error.code == "ACCOUNT_NOT_FOUND"
    assert not (tmp_path / "home").exists()
    assert not account_sandbox.accounts.exists()


def test_create_false_uses_an_account_that_resolves(tmp_path, monkeypatch):
    stored = new_account()
    keystore.save(stored.private_key)
    assert Profile.load_or_create(tmp_path / "stored", create=False).address == stored.address

    supplied = new_account()
    monkeypatch.setenv("ALEO_ACCOUNT", "per-run")
    monkeypatch.setenv("ALEO_PRIVATE_KEY", supplied.private_key)
    assert Profile.load_or_create(tmp_path / "supplied", create=False).address == supplied.address


def test_creating_without_a_stored_key_is_rejected(tmp_path, account_sandbox):
    error = failure(lambda: Profile.load_or_create(tmp_path / "home", storage="none"))
    assert error.code == "UNSUPPORTED_STORAGE"
    assert not (tmp_path / "home").exists()
    assert not account_sandbox.accounts.exists()


def test_storage_none_is_only_rejected_when_an_account_would_be_created(tmp_path, monkeypatch):
    account = new_account()
    monkeypatch.setenv("ALEO_PRIVATE_KEY", account.private_key)
    assert Profile.load_or_create(tmp_path / "home", storage="none").address == account.address


def test_unknown_storage_is_rejected_before_anything_is_written(tmp_path, account_sandbox):
    error = failure(lambda: Profile.load_or_create(tmp_path / "home", storage="vault"))
    assert error.code == "UNSUPPORTED_STORAGE"
    assert not (tmp_path / "home").exists()
    assert not account_sandbox.accounts.exists()


def test_unknown_network_is_rejected_before_anything_is_written(tmp_path, account_sandbox):
    with pytest.raises(AttributeError):
        Profile.load_or_create(tmp_path / "home", network="nonet")
    assert not (tmp_path / "home").exists()
    assert not account_sandbox.accounts.exists()


def test_account_argument_names_the_account_file(tmp_path, account_sandbox):
    profile = Profile.load_or_create(tmp_path / "home", account="trading")
    assert saved_profile(profile.home)["account"] == "trading"
    assert (account_sandbox.accounts / "trading.json").exists()
    assert not (account_sandbox.accounts / "default.json").exists()


def test_account_variable_names_the_account_file(tmp_path, monkeypatch, account_sandbox):
    monkeypatch.setenv("ALEO_ACCOUNT", "from-env")
    profile = Profile.load_or_create(tmp_path / "home")
    assert saved_profile(profile.home)["account"] == "from-env"
    assert (account_sandbox.accounts / "from-env.json").exists()


@pytest.mark.parametrize("argument, variable, recorded", [
    (None, None, "default"), (None, "from-env", "from-env"), ("from-argument", "from-env", "from-argument"),
])
def test_selected_name_is_recorded_when_the_key_comes_from_the_environment(
        tmp_path, monkeypatch, account_sandbox, argument, variable, recorded):
    """Keep the selected name so the same key can later be imported under it."""
    monkeypatch.setenv("ALEO_PRIVATE_KEY", new_account().private_key)
    if variable is not None:
        monkeypatch.setenv("ALEO_ACCOUNT", variable)
    profile = Profile.load_or_create(tmp_path / "home", account=argument)
    assert saved_profile(profile.home)["account"] == recorded
    assert not account_sandbox.accounts.exists()


def test_account_argument_beats_the_variable_when_creating(tmp_path, monkeypatch, account_sandbox):
    monkeypatch.setenv("ALEO_ACCOUNT", "from-env")
    profile = Profile.load_or_create(tmp_path / "home", account="from-argument")
    assert saved_profile(profile.home)["account"] == "from-argument"
    assert [path.name for path in account_sandbox.accounts.iterdir()] == ["from-argument.json"]


def test_invalid_account_name_is_rejected(tmp_path, account_sandbox):
    assert failure(lambda: Profile.load_or_create(tmp_path / "home", account="Not Valid")).code == "INVALID_NAME"
    assert not (tmp_path / "home").exists()


def test_accounts_dir_is_used_and_not_recorded(tmp_path, account_sandbox):
    accounts = tmp_path / "accounts"
    profile = Profile.load_or_create(tmp_path / "home", accounts_dir=accounts)
    assert (accounts / "default.json").exists()
    assert not account_sandbox.accounts.exists()
    assert set(saved_profile(profile.home)) == {"address", "account", "network", "endpoint"}

    # Reopening from a custom location requires the same directory.
    same_secret(Profile.load_or_create(tmp_path / "home", accounts_dir=accounts).private_key, profile.private_key)
    without = Profile.load_or_create(tmp_path / "home")
    assert failure(lambda: without.private_key).code == "ACCOUNT_NOT_FOUND"


def test_accounts_dir_variable_is_honoured(tmp_path, monkeypatch, account_sandbox):
    monkeypatch.setenv("ALEO_ACCOUNTS_DIR", str(tmp_path / "from-env"))
    profile = Profile.load_or_create(tmp_path / "home")
    assert (tmp_path / "from-env" / "default.json").exists()
    assert not account_sandbox.accounts.exists()
    same_secret(Profile.load_or_create(tmp_path / "home").private_key, profile.private_key)


@pytest.mark.parametrize("name", ["profile", "credentials"])
def test_account_file_that_is_a_profile_file_is_rejected_before_anything_is_written(tmp_path, name):
    home = tmp_path / "home"
    with pytest.raises(ValueError, match=f"{name}.json"):
        Profile.load_or_create(home, accounts_dir=home, account=name)
    assert not home.exists()


def test_account_file_that_is_a_profile_file_is_rejected_when_the_variables_select_it(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("ALEO_ACCOUNTS_DIR", str(home))
    monkeypatch.setenv("ALEO_ACCOUNT", "profile")
    with pytest.raises(ValueError, match="profile.json"):
        Profile.load_or_create(home)
    assert not home.exists()


def test_account_file_that_is_a_profile_file_is_rejected_through_another_spelling_of_the_home(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (tmp_path / "link").symlink_to(home, target_is_directory=True)
    with pytest.raises(ValueError, match="profile.json"):
        Profile.load_or_create(home, accounts_dir=tmp_path / "link", account="profile")
    assert files_under(home) == []


def test_stored_account_named_like_the_credentials_file_is_left_untouched(tmp_path):
    # Saving credentials would replace this account file, so the profile refuses it.
    home = tmp_path / "home"
    account = new_account()
    stored = keystore.save(account.private_key, name="credentials", dir=home)
    before = Path(stored.path).read_bytes()
    with pytest.raises(ValueError, match="credentials.json"):
        Profile.load_or_create(home, accounts_dir=home, account="credentials")
    assert files_under(home) == [Path(stored.path)]
    assert Path(stored.path).read_bytes() == before


def test_other_account_names_may_live_in_the_profile_home(tmp_path):
    home = tmp_path / "home"
    profile = Profile.load_or_create(home, accounts_dir=home, account="trading")
    profile.save_credentials(jwt="token")
    reopened = Profile.load_or_create(home, accounts_dir=home)
    same_secret(reopened.private_key, profile.private_key)
    assert {path.name for path in files_under(home)} == {"profile.json", "credentials.json", "trading.json"}


# Protecting keys with encryption or Keychain

def test_encrypted_account_needs_its_password_on_later_runs(tmp_path, monkeypatch, account_sandbox):
    created = Profile.load_or_create(tmp_path / "home", storage="encrypted", password=PASSWORD)
    stored = json.loads((account_sandbox.accounts / "default.json").read_text())
    assert stored["key"]["type"] == "encrypted"
    assert_key_on_no_disk(created.private_key, tmp_path, account_sandbox.home)

    # Inspecting an existing profile must not require unlocking its key.
    locked = Profile.load_or_create(tmp_path / "home")
    assert locked.address == created.address
    assert failure(lambda: locked.private_key).code == "PASSWORD_REQUIRED"

    same_secret(Profile.load_or_create(tmp_path / "home", password=PASSWORD).private_key, created.private_key)
    wrong = Profile.load_or_create(tmp_path / "home", password=OTHER_PASSWORD)
    assert failure(lambda: wrong.private_key).code == "WRONG_PASSWORD"

    monkeypatch.setenv("ALEO_ACCOUNT_PASSWORD", PASSWORD)
    same_secret(Profile.load_or_create(tmp_path / "home").private_key, created.private_key)


def test_encrypted_account_takes_its_password_from_the_variable_when_created(tmp_path, monkeypatch, account_sandbox):
    monkeypatch.setenv("ALEO_ACCOUNT_PASSWORD", PASSWORD)
    created = Profile.load_or_create(tmp_path / "home", storage="encrypted")
    same_secret(keystore.export(password=PASSWORD), created.private_key)


def test_encrypted_account_without_a_password_source_creates_nothing(tmp_path, account_sandbox):
    error = failure(lambda: Profile.load_or_create(tmp_path / "home", storage="encrypted"))
    assert error.code == "PASSWORD_REQUIRED"
    assert not (tmp_path / "home").exists()
    assert files_under(account_sandbox.home) == []


def test_new_profile_adopts_an_encrypted_account(tmp_path):
    account = new_account()
    keystore.save(account.private_key, storage="encrypted", password=PASSWORD)
    assert failure(lambda: Profile.load_or_create(tmp_path / "locked")).code == "PASSWORD_REQUIRED"
    assert not (tmp_path / "locked").exists()
    profile = Profile.load_or_create(tmp_path / "home", password=PASSWORD)
    assert profile.address == account.address
    same_secret(profile.private_key, account.private_key)


def test_saved_password_unlocks_later_runs(tmp_path, monkeypatch, account_sandbox):
    from aleo.keystore._runtime import runtime
    monkeypatch.setattr(runtime, "platform", "darwin")
    created = Profile.load_or_create(tmp_path / "home", storage="encrypted",
                                     password=PASSWORD, save_password=True)
    assert [entry.split()[0] for entry in account_sandbox.keychain.entries] == ["aleo-account-password"]
    same_secret(Profile.load_or_create(tmp_path / "home").private_key, created.private_key)


def test_keychain_storage_keeps_the_key_out_of_every_file(tmp_path, monkeypatch, account_sandbox):
    from aleo.keystore._runtime import runtime
    monkeypatch.setattr(runtime, "platform", "darwin")
    created = Profile.load_or_create(tmp_path / "home", storage="keychain")
    stored = json.loads((account_sandbox.accounts / "default.json").read_text())
    assert stored["key"]["type"] == "keychain"
    assert_key_on_no_disk(created.private_key, tmp_path, account_sandbox.home)
    (entry, value), = account_sandbox.keychain.entries.items()
    assert entry == f"aleo-account-key {stored['key']['account']}"
    same_secret(value, created.private_key)
    same_secret(Profile.load_or_create(tmp_path / "home").private_key, created.private_key)


def test_keychain_storage_outside_macos_creates_nothing(tmp_path, monkeypatch, account_sandbox):
    from aleo.keystore._runtime import runtime
    monkeypatch.setattr(runtime, "platform", "linux")
    error = failure(lambda: Profile.load_or_create(tmp_path / "home", storage="keychain"))
    assert error.code == "KEYCHAIN_UNAVAILABLE"
    assert not (tmp_path / "home").exists()
    assert files_under(account_sandbox.home) == []
    assert account_sandbox.keychain.entries == {}


# Reopening profiles and supporting older files

def test_legacy_profile_keeps_working_and_is_left_untouched(tmp_path, account_sandbox):
    account = new_account()
    path = write_legacy_profile(tmp_path / "home", account)
    before = path.read_bytes()
    profile = Profile.load_or_create(tmp_path / "home")
    assert profile.address == account.address
    same_secret(profile.private_key, account.private_key)
    assert profile.network == "testnet" and profile.endpoint == DEFAULT_ENDPOINT
    assert path.read_bytes() == before
    assert not account_sandbox.accounts.exists()


def test_legacy_profile_is_reported_once(tmp_path, caplog):
    account = new_account()
    path = write_legacy_profile(tmp_path / "home", account)
    with caplog.at_level(logging.WARNING, logger="aleo_shield_swap.profile"):
        first = Profile.load_or_create(tmp_path / "home")
        assert caplog.records == []                       # Warn only when the legacy key is used.
        first.private_key
        first.private_key
        Profile.load_or_create(tmp_path / "home").private_key
    (record,) = caplog.records
    assert record.levelno == logging.WARNING and record.name == "aleo_shield_swap.profile"
    message = record.getMessage()
    assert f"python -m aleo.keystore migrate --from {path}" in message
    assert account.address in message and str(path) in message
    if account.private_key in message or account.private_key in caplog.text:
        raise AssertionError("the notice contains a private key")


def test_each_legacy_profile_is_reported(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="aleo_shield_swap.profile"):
        for name in ("one", "two"):
            write_legacy_profile(tmp_path / name, new_account())
            Profile.load_or_create(tmp_path / name).private_key
    assert len(caplog.records) == 2


def test_migrated_legacy_profile_reads_the_account_file_without_a_notice(tmp_path, caplog, account_sandbox):
    account = new_account()
    path = write_legacy_profile(tmp_path / "home", account)
    keystore.migrate(path)                                 # Migration keeps the original key unless removal is requested.
    with caplog.at_level(logging.WARNING, logger="aleo_shield_swap.profile"):
        same_secret(Profile.load_or_create(tmp_path / "home").private_key, account.private_key)
    assert caplog.records == []

    keystore.migrate(path, remove_legacy_key=True)
    assert "private_key" not in saved_profile(tmp_path / "home")
    assert_key_on_no_disk(account.private_key, tmp_path)
    with caplog.at_level(logging.WARNING, logger="aleo_shield_swap.profile"):
        same_secret(Profile.load_or_create(tmp_path / "home").private_key, account.private_key)
    assert caplog.records == []


def test_legacy_profile_with_the_same_key_in_the_environment_gives_no_notice(tmp_path, monkeypatch, caplog):
    account = new_account()
    write_legacy_profile(tmp_path / "home", account)
    monkeypatch.setenv("SHIELD_SWAP_PRIVATE_KEY", account.private_key)
    with caplog.at_level(logging.WARNING, logger="aleo_shield_swap.profile"):
        same_secret(Profile.load_or_create(tmp_path / "home").private_key, account.private_key)
    assert caplog.records == []


@pytest.mark.parametrize("legacy", [True, False])
@pytest.mark.parametrize("variable", ["ALEO_PRIVATE_KEY", "SHIELD_SWAP_PRIVATE_KEY", "ALEO_PRIVATE_KEY_FILE"])
def test_environment_key_for_another_account_is_an_error(tmp_path, monkeypatch, legacy, variable):
    other = new_account()
    if legacy:
        bound = new_account()
        write_legacy_profile(tmp_path / "home", bound)
    else:
        bound = SimpleNamespace(address=Profile.load_or_create(tmp_path / "home").address)
    value = other.private_key
    if variable.endswith("_FILE"):
        (tmp_path / "other.key").write_text(other.private_key)
        value = str(tmp_path / "other.key")
    monkeypatch.setenv(variable, value)

    profile = Profile.load_or_create(tmp_path / "home")
    assert profile.address == bound.address
    error = failure(lambda: profile.private_key)
    assert error.code == "ADDRESS_MISMATCH"
    assert bound.address in str(error) and other.address in str(error)
    if other.private_key in str(error):
        raise AssertionError("the error contains a private key")


def test_environment_key_for_the_same_account_is_used(tmp_path, monkeypatch):
    created = Profile.load_or_create(tmp_path / "home")
    key = created.private_key
    monkeypatch.setenv("ALEO_PRIVATE_KEY", key)
    same_secret(Profile.load_or_create(tmp_path / "home").private_key, key)
    assert keystore.load(expected_address=created.address).source == "env"


def test_empty_key_variable_is_an_error_for_an_existing_profile(tmp_path, monkeypatch):
    Profile.load_or_create(tmp_path / "home")
    monkeypatch.setenv("ALEO_PRIVATE_KEY", "")
    profile = Profile.load_or_create(tmp_path / "home")
    assert failure(lambda: profile.private_key).code == "EMPTY_KEY_SOURCE"


def test_account_file_for_another_address_is_an_error(tmp_path, account_sandbox):
    profile = Profile.load_or_create(tmp_path / "home")
    (account_sandbox.accounts / "default.json").unlink()
    other = new_account()
    keystore.save(other.private_key)
    reloaded = Profile.load_or_create(tmp_path / "home")
    error = failure(lambda: reloaded.private_key)
    assert error.code == "ADDRESS_MISMATCH"
    assert profile.address in str(error) and other.address in str(error)


def test_legacy_key_must_match_an_account_file_of_the_same_name(tmp_path):
    legacy, other = new_account(), new_account()
    write_legacy_profile(tmp_path / "home", legacy)
    keystore.save(other.private_key)
    profile = Profile.load_or_create(tmp_path / "home")
    error = failure(lambda: profile.private_key)
    assert error.code == "ADDRESS_MISMATCH"
    assert legacy.address in str(error) and other.address in str(error)


def test_address_only_account_file_needs_the_key_from_the_environment(tmp_path, monkeypatch):
    account = new_account()
    monkeypatch.setenv("ALEO_PRIVATE_KEY", account.private_key)
    keystore.save(account.private_key, storage="none")
    Profile.load_or_create(tmp_path / "home")
    same_secret(Profile.load_or_create(tmp_path / "home").private_key, account.private_key)
    monkeypatch.delenv("ALEO_PRIVATE_KEY")
    profile = Profile.load_or_create(tmp_path / "home")
    assert failure(lambda: profile.private_key).code == "KEY_NOT_STORED"


# Choosing the account for a saved profile

def test_recorded_account_name_is_used(tmp_path, lookups):
    created = Profile.load_or_create(tmp_path / "home", account="trading")
    del lookups[:]
    same_secret(Profile.load_or_create(tmp_path / "home").private_key, created.private_key)
    assert [call["name"] for call in lookups] == ["trading"]


def test_account_variable_beats_the_recorded_name(tmp_path, monkeypatch, account_sandbox, lookups):
    created = Profile.load_or_create(tmp_path / "home", account="trading")
    monkeypatch.setenv("ALEO_ACCOUNT", "elsewhere")
    del lookups[:]
    profile = Profile.load_or_create(tmp_path / "home")
    error = failure(lambda: profile.private_key)
    assert error.code == "ACCOUNT_NOT_FOUND" and "elsewhere.json" in str(error)
    # Leave name unset so keystore.load() can use ALEO_ACCOUNT.
    assert [call["name"] for call in lookups] == [None]

    # Renaming the account is allowed when its address stays the same.
    keystore.save(created.private_key, name="elsewhere")
    same_secret(Profile.load_or_create(tmp_path / "home").private_key, created.private_key)

    # An override must not switch the profile to another address.
    monkeypatch.setenv("ALEO_ACCOUNT", "stranger")
    keystore.save(new_account().private_key, name="stranger")
    stranger = Profile.load_or_create(tmp_path / "home")
    assert failure(lambda: stranger.private_key).code == "ADDRESS_MISMATCH"


def test_account_argument_beats_the_variable_and_the_recorded_name(tmp_path, monkeypatch, lookups):
    created = Profile.load_or_create(tmp_path / "home", account="trading")
    keystore.save(created.private_key, name="copy")
    monkeypatch.setenv("ALEO_ACCOUNT", "missing")
    del lookups[:]
    same_secret(Profile.load_or_create(tmp_path / "home", account="copy").private_key, created.private_key)
    assert [call["name"] for call in lookups] == ["copy"]


def test_empty_account_variable_counts_as_unset(tmp_path, monkeypatch, lookups):
    created = Profile.load_or_create(tmp_path / "home", account="trading")
    monkeypatch.setenv("ALEO_ACCOUNT", "")
    del lookups[:]
    same_secret(Profile.load_or_create(tmp_path / "home").private_key, created.private_key)
    assert [call["name"] for call in lookups] == ["trading"]


def test_profile_without_a_recorded_name_selects_default(tmp_path, lookups):
    account = new_account()
    path = write_legacy_profile(tmp_path / "home", account)
    keystore.migrate(path, remove_legacy_key=True)
    del lookups[:]
    same_secret(Profile.load_or_create(tmp_path / "home").private_key, account.private_key)
    assert [call["name"] for call in lookups] == [None]


def test_lookup_passes_the_recorded_address_and_the_legacy_key(tmp_path, lookups):
    account = new_account()
    write_legacy_profile(tmp_path / "home", account)
    Profile.load_or_create(tmp_path / "home", accounts_dir=tmp_path / "accounts", password=PASSWORD).private_key
    (call,) = lookups
    assert call["expected_address"] == account.address
    same_secret(call["legacy_key"], account.private_key)
    assert call["dir"] == tmp_path / "accounts"
    same_secret(call["password"], PASSWORD)


# Unlocking once and retrying failed lookups

def test_key_is_resolved_once_per_object_and_only_when_read(tmp_path, lookups):
    Profile.load_or_create(tmp_path / "home")
    del lookups[:]
    profile = Profile.load_or_create(tmp_path / "home")
    assert profile.address and profile.network and profile.credentials == {}
    assert lookups == []                                   # Reading metadata does not unlock the account.
    for _ in range(3):
        profile.private_key
    assert len(lookups) == 1
    Profile.load_or_create(tmp_path / "home").private_key  # Each new object must load its own key.
    assert len(lookups) == 2


def test_new_profile_keeps_the_key_it_was_created_with(tmp_path, lookups):
    profile = Profile.load_or_create(tmp_path / "home", storage="encrypted", password=PASSWORD)
    assert len(lookups) == 1                               # Creation first checked for an existing account.
    profile.private_key
    profile.private_key
    assert len(lookups) == 1


def test_failed_lookup_is_retried(tmp_path, monkeypatch):
    created = Profile.load_or_create(tmp_path / "home", storage="encrypted", password=PASSWORD)
    profile = Profile.load_or_create(tmp_path / "home")
    assert failure(lambda: profile.private_key).code == "PASSWORD_REQUIRED"
    monkeypatch.setenv("ALEO_ACCOUNT_PASSWORD", PASSWORD)
    same_secret(profile.private_key, created.private_key)


def test_creation_arguments_are_ignored_for_an_existing_profile(tmp_path, account_sandbox):
    created = Profile.load_or_create(tmp_path / "home")
    again = Profile.load_or_create(tmp_path / "home", network="mainnet", endpoint="https://other.example",
                                   create=False, storage="none", save_password=True)
    assert (again.address, again.network, again.endpoint) == (created.address, "testnet", DEFAULT_ENDPOINT)
    same_secret(again.private_key, created.private_key)
    assert [path.name for path in account_sandbox.accounts.iterdir()] == ["default.json"]


def test_repr_and_saved_profile_hold_no_key(tmp_path):
    profile = Profile.load_or_create(tmp_path / "home")
    key = profile.private_key
    if key in repr(profile) or key in (profile.home / "profile.json").read_text():
        raise AssertionError("the profile exposes a private key")


# Opening a trading client

def test_from_profile_signs_with_the_resolved_key(tmp_path, monkeypatch):
    account = new_account()
    keystore.save(account.private_key, storage="encrypted", password=PASSWORD)
    monkeypatch.setenv("ALEO_ACCOUNT_PASSWORD", PASSWORD)
    aleo = Mock()
    aleo.network_name = "testnet"
    monkeypatch.setattr("aleo.Aleo", Mock(return_value=aleo))
    monkeypatch.setattr("aleo.HTTPProvider", Mock())

    dex = ShieldSwap.from_profile(tmp_path / "home")

    (bound,), _ = aleo.account.from_private_key.call_args
    same_secret(bound, account.private_key)
    assert dex.profile.address == account.address
    assert dex.journal.path == tmp_path / "home" / "journal.jsonl"
    assert saved_profile(tmp_path / "home")["account"] == "default"


def test_from_profile_reports_a_locked_account(tmp_path, monkeypatch):
    keystore.save(new_account().private_key, storage="encrypted", password=PASSWORD)
    monkeypatch.setattr("aleo.Aleo", Mock())
    monkeypatch.setattr("aleo.HTTPProvider", Mock())
    assert failure(lambda: ShieldSwap.from_profile(tmp_path / "home")).code == "PASSWORD_REQUIRED"
    assert not (tmp_path / "home").exists()


# Profile directories, credentials, and journals

def test_env_home_override(tmp_path, monkeypatch):
    monkeypatch.setenv("SHIELD_SWAP_HOME", str(tmp_path / "envhome"))
    p = Profile.load_or_create()
    assert p.home == tmp_path / "envhome"


def test_default_home_is_under_the_home_directory(account_sandbox):
    p = Profile.load_or_create()
    assert p.home == account_sandbox.home / ".shield-swap"
    assert (account_sandbox.home / ".shield-swap" / "profile.json").exists()


def test_credentials_roundtrip(tmp_path):
    p = Profile.load_or_create(tmp_path / "home")
    assert p.credentials == {}
    p.save_credentials(jwt="j", dps_api_key="k")
    p.save_credentials(dps_consumer_id="c")            # merges, not replaces
    p2 = Profile.load_or_create(tmp_path / "home")
    assert p2.credentials == {"jwt": "j", "dps_api_key": "k",
                              "dps_consumer_id": "c"}
    assert mode(p.home / "credentials.json") == 0o600


def test_journal_path(tmp_path):
    p = Profile.load_or_create(tmp_path / "home")
    assert p.journal_path == p.home / "journal.jsonl"


def test_tilde_in_home_is_expanded(tmp_path, monkeypatch):
    """Expand ~ so a restart can find the profile in the expected home directory."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    profile = Profile.load_or_create("~/.shield-swap-tilde")
    assert profile.home == tmp_path / ".shield-swap-tilde"
    assert not (tmp_path / "~").exists(), "created a literal ~ directory"


def test_tilde_in_shield_swap_home_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHIELD_SWAP_HOME", "~/from-env")
    monkeypatch.chdir(tmp_path)
    assert Profile.default_home() == tmp_path / "from-env"
    assert Profile.load_or_create().home == tmp_path / "from-env"
    assert not (tmp_path / "~").exists()
