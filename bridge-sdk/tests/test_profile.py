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
