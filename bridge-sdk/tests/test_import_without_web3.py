"""``import aleo_bridge`` must succeed with web3/eth_account absent; only the first Ethereum(...)
call that actually needs them raises MissingExtraError. Uses the same monkeypatch-and-revert
pattern as test_package.py's test_import_without_optional_extras so the blocked modules and the
reimported aleo_bridge never leak into tests that run after this one.
"""
from __future__ import annotations

import importlib
import sys

import pytest


def test_package_imports_without_web3(monkeypatch):
    monkeypatch.setitem(sys.modules, "web3", None)
    monkeypatch.setitem(sys.modules, "eth_account", None)
    for name in list(sys.modules):
        if name.startswith("aleo_bridge"):
            monkeypatch.delitem(sys.modules, name)

    pkg = importlib.import_module("aleo_bridge")
    assert pkg.Ethereum is not None and pkg.EthModule is not None and pkg.EvmCall is not None

    from aleo_bridge.errors import MissingExtraError

    with pytest.raises(MissingExtraError) as exc_info:
        pkg.Ethereum("http://127.0.0.1:1")
    assert "aleo-bridge-sdk[evm]" in str(exc_info.value)
