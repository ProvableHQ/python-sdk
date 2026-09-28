"""The runnable example saves generated keys before scanner or faucet I/O."""
import json
import runpy
from pathlib import Path
from unittest.mock import Mock

import pytest
from aleo import testnet
from aleo_shield_swap import Profile

EXAMPLE = Path(__file__).parents[1] / "examples/first-swap/swap.py"


def test_example_reuses_persisted_key_before_network(tmp_path, monkeypatch):
    monkeypatch.delenv("SHIELD_SWAP_PRIVATE_KEY", raising=False)
    monkeypatch.delenv("SHIELD_SWAP_PRIVATE_KEY_FILE", raising=False)
    monkeypatch.setattr(Profile, "default_home", staticmethod(lambda: tmp_path))
    # Stop at client construction, before registration/authentication/funding.
    monkeypatch.setattr("aleo.Aleo", Mock(side_effect=RuntimeError("stop before I/O")))
    for _ in range(2):
        with pytest.raises(RuntimeError, match="stop before I/O"):
            runpy.run_path(str(EXAMPLE), run_name="__main__")
        saved = json.loads((tmp_path / "profile.json").read_text())
        if _ == 0:
            first_key = saved["private_key"]
        assert saved["private_key"] == first_key
        assert saved["network"] == "testnet"
        assert (tmp_path / "profile.json").stat().st_mode & 0o777 == 0o600


def test_example_uses_explicit_key_without_loading_profile(monkeypatch):
    monkeypatch.setenv("SHIELD_SWAP_PRIVATE_KEY", str(testnet.PrivateKey.random()))
    load = Mock(side_effect=AssertionError("must not load profile"))
    monkeypatch.setattr(Profile, "load_or_create", load)
    monkeypatch.setattr("aleo.Aleo", Mock(side_effect=RuntimeError("stop before I/O")))
    with pytest.raises(RuntimeError, match="stop before I/O"):
        runpy.run_path(str(EXAMPLE), run_name="__main__")
    load.assert_not_called()
