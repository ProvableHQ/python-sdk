import json
import os
import stat

import pytest

from aleo_bridge.errors import ConfigurationError
from aleo_bridge.profile import DEFAULT_ENDPOINT, Profile


def test_profile_created_once_with_private_mode(tmp_path, monkeypatch):
    monkeypatch.delenv("BRIDGE_PRIVATE_KEY", raising=False)
    monkeypatch.delenv("BRIDGE_PRIVATE_KEY_FILE", raising=False)
    profile = Profile.load_or_create(tmp_path / "home")
    assert profile.address.startswith("aleo1") and len(profile.address) == 63
    assert profile.private_key.startswith("APrivateKey1") and profile.network == "mainnet" and profile.endpoint == DEFAULT_ENDPOINT
    assert stat.S_IMODE(os.stat(tmp_path / "home" / "profile.json").st_mode) == 0o600
    again = Profile.load_or_create(tmp_path / "home", network="testnet", endpoint="https://other.example")
    assert (again.address, again.network, again.endpoint) == (profile.address, "mainnet", DEFAULT_ENDPOINT)  # creation-only args
    data = json.loads((tmp_path / "home" / "profile.json").read_text())
    assert set(data) == {"address", "private_key", "network", "endpoint"}
    assert profile.checkpoint_dir == tmp_path / "home" / "checkpoints" and profile.checkpoint_dir.is_dir()


def test_profile_imports_key_from_env_or_file(tmp_path, monkeypatch):
    from aleo import testnet as net
    key = net.PrivateKey.random()
    monkeypatch.setenv("BRIDGE_PRIVATE_KEY", str(key))
    assert Profile.load_or_create(tmp_path / "a", network="testnet").address == str(key.address)
    monkeypatch.delenv("BRIDGE_PRIVATE_KEY")
    key_file = tmp_path / "key.txt"
    key_file.write_text(f"{key}\n")
    monkeypatch.setenv("BRIDGE_PRIVATE_KEY_FILE", str(key_file))
    assert Profile.load_or_create(tmp_path / "b", network="testnet").address == str(key.address)


def test_default_home_and_tilde_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("ALEO_BRIDGE_HOME", str(tmp_path / "x"))
    assert Profile.default_home() == tmp_path / "x"
    monkeypatch.delenv("ALEO_BRIDGE_HOME")
    assert Profile.default_home().name == ".aleo-bridge"
    monkeypatch.setenv("HOME", str(tmp_path))
    assert Profile.load_or_create("~/p", network="testnet").home == tmp_path / "p"


def test_profile_rejects_unknown_network(tmp_path):
    with pytest.raises(ConfigurationError, match="network"):
        Profile.load_or_create(tmp_path / "bad", network="devnet")


def test_profile_creation_is_exclusive_loser_adopts_winner_key(tmp_path, monkeypatch):
    """Two processes racing Profile.load_or_create() on a fresh home must not clobber each other:
    the exclusive create (O_EXCL) means the loser adopts the winner's already-written key."""
    from aleo import mainnet as net

    monkeypatch.delenv("BRIDGE_PRIVATE_KEY", raising=False)
    monkeypatch.delenv("BRIDGE_PRIVATE_KEY_FILE", raising=False)
    home = tmp_path / "home"
    winner_key = net.PrivateKey.random()
    winner_address = str(winner_key.address)

    def racing_initial_key(network):
        # Simulate another process winning the race: it creates the home dir and profile.json
        # before this process gets to its own exclusive-open attempt.
        home.mkdir(parents=True, exist_ok=True, mode=0o700)
        (home / "profile.json").write_text(json.dumps({
            "address": winner_address, "private_key": str(winner_key), "network": network, "endpoint": DEFAULT_ENDPOINT,
        }))
        loser_key = net.PrivateKey.random()   # must NOT end up written or returned
        return str(loser_key), str(loser_key.address)

    monkeypatch.setattr("aleo_bridge.profile._initial_key", racing_initial_key)
    profile = Profile.load_or_create(home)
    assert profile.address == winner_address
    assert json.loads((home / "profile.json").read_text())["address"] == winner_address
    assert stat.S_IMODE(os.stat(home).st_mode) == 0o700
