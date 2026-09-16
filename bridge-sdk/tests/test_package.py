import importlib
import sys

import pytest


def test_import_without_optional_extras(monkeypatch):
    for mod in ("web3", "eth_account", "solders", "solana", "mcp"):
        monkeypatch.setitem(sys.modules, mod, None)  # any import of these now raises ImportError
    for name in list(sys.modules):
        if name.startswith("aleo_bridge"):
            monkeypatch.delitem(sys.modules, name)  # reverted at teardown — a fresh reimport below must not
                                                     # leak new module/class objects into tests that run after this one
    pkg = importlib.import_module("aleo_bridge")
    assert pkg.__version__ == "0.1.0"
    assert issubclass(pkg.RouteNotFoundError, pkg.BridgeError)


def test_error_hierarchy_and_messages():
    from aleo_bridge import errors as e

    for cls in (e.ConfigurationError, e.MissingExtraError, e.RouteNotFoundError, e.AmbiguousRouteError,
                e.RouteUnavailableError, e.RegistryVersionMismatchError, e.UnsupportedRouteError,
                e.InvalidAmountError, e.InvalidRecipientError, e.InsufficientBalanceError,
                e.ChainMismatchError, e.NotResumableError, e.CheckpointInvalidError, e.AttestationError,
                e.DeliveryUnknownError, e.PollingTimeoutError):
        assert issubclass(cls, e.BridgeError)
    err = e.MissingExtraError("evm", "Ethereum connections")
    assert err.extra == "evm"
    assert "pip install 'aleo-bridge-sdk[evm]'" in str(err)


def test_polling_timeout_carries_status():
    from aleo_bridge.errors import PollingTimeoutError

    err = PollingTimeoutError("Bridge status polling timed out in state DELIVERY_PENDING",
                              status="DELIVERY_PENDING", progress=None)
    assert err.status == "DELIVERY_PENDING" and err.progress is None
    with pytest.raises(PollingTimeoutError):
        raise err
