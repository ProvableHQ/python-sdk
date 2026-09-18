import importlib
import re
import sys
import tomllib
from pathlib import Path

import pytest

import aleo_bridge

ROOT = Path(__file__).resolve().parents[1]


def test_version_is_pinned_in_lockstep():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert pyproject["project"]["version"] == "0.1.0" == aleo_bridge.__version__
    assert pyproject["project"]["name"] == "aleo-bridge-sdk"
    extras = pyproject["project"]["optional-dependencies"]
    assert {"evm", "solana", "mcp", "dev"} <= set(extras)
    assert any(dep.startswith("mcp>=1") and "<2" in dep for dep in extras["mcp"])


def test_wheel_ships_agents_md():
    assert (ROOT / "python" / "aleo_bridge" / "AGENTS.md").exists()
    assert aleo_bridge.agent_guide().startswith("# aleo-bridge")


def test_readme_covers_the_journey():
    readme = (ROOT / "README.md").read_text()
    for needle in ("Bridge.from_env()", "Bridge.from_profile()", "Ethereum(", "Solana(", "progress.next",
                   "recover", "resume", "complete", "shield", "unshield", "python -m aleo_bridge.mcp",
                   "scripts/rehearse.py", "BRIDGE_PRIVATE_KEY", "EVM_PRIVATE_KEY", "SOLANA_PRIVATE_KEY",
                   "BRIDGE_LIVE_MAINNET_EXECUTE", "secret_nonce", "| `resume` |", "| `complete` |"):
        assert needle in readme, needle
    assert "Co-Authored-By" not in readme
    # every active mainnet route appears in the route table
    for route_id in ("xreserve:ethereum/usdc->aleo/usdcx", "hyperlane:ethereum/eth->aleo/eth",
                     "hyperlane:aleo/sol->solana/sol"):
        assert route_id in readme


def test_ci_has_bridge_jobs():
    workflow = (ROOT.parent / ".github" / "workflows" / "sdk-wheels.yml").read_text()
    assert workflow.count("'bridge-sdk/**'") == 2
    for job in ("build-bridge:", "release-bridge:"):
        assert job in workflow
    assert "environment: pypi-bridge" in workflow
    assert 'pip install "$(ls bridge-sdk/dist/*.whl)[evm,solana,mcp]"' in workflow
    assert re.search(r'pip install "\$\(ls bridge-sdk/dist/\*\.whl\)"\s*\n\s*python -c "import aleo_bridge', workflow)


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
