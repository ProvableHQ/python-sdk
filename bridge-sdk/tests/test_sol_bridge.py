"""Task 8: wiring the Solana connection into Bridge (sol property, from_env, status)."""
import pytest

pytest.importorskip("solders")
from solders.keypair import Keypair

import aleo_bridge
from aleo_bridge import Bridge, Solana
from aleo_bridge._base58 import b58encode
from aleo_bridge.errors import ConfigurationError
from aleo_bridge.sol import SolModule
from aleo_bridge.types import ChainStatus
from tests.conftest import FakeAleo
from tests.fakes.fake_solana import FakeSolanaClient


def test_sol_property_raises_when_solana_is_not_configured():
    bridge = Bridge(FakeAleo())
    assert bridge.solana is None
    with pytest.raises(ConfigurationError, match="Solana is not configured"):
        bridge.sol


def test_bare_client_is_wrapped_read_only():
    fake = FakeSolanaClient(balance=5)
    bridge = Bridge(FakeAleo(), solana=fake)
    assert isinstance(bridge.solana, Solana) and bridge.solana.client is fake
    assert bridge.solana.can_sign is False and bridge.solana.address is None
    assert isinstance(bridge.sol, SolModule) and bridge.sol is bridge.sol
    with pytest.raises(ConfigurationError, match="read-only"):
        bridge.sol.balance()
    solana_status = [c for c in bridge.status().chains if c.chain_id == "solana"]
    assert solana_status == [ChainStatus(chain_id="solana", address=None, can_sign=False, balances={})]


def test_configured_connection_reports_sol_balance_in_status():
    keypair = Keypair()
    fake = FakeSolanaClient(balance=1_234)
    bridge = Bridge(FakeAleo(), solana=Solana(client=fake, signer=keypair))
    assert bridge.solana.address == str(keypair.pubkey())
    solana_status = [c for c in bridge.status().chains if c.chain_id == "solana"][0]
    assert solana_status.address == str(keypair.pubkey()) and solana_status.can_sign is True
    assert solana_status.balances == {"solana/sol": 1_234}


def test_an_rpc_url_string_becomes_a_read_only_connection():
    bridge = Bridge(FakeAleo(), solana="https://rpc.example")
    assert isinstance(bridge.solana, Solana) and bridge.solana.rpc_url == "https://rpc.example"
    assert bridge.solana.can_sign is False


def test_an_object_that_is_not_a_solana_client_is_refused():
    class NotAClient:
        def get_latest_blockhash(self, commitment=None):    # pragma: no cover - never called
            return None

    with pytest.raises(ConfigurationError, match="solana="):
        Bridge(FakeAleo(), solana=NotAClient())             # no get_account_info
    with pytest.raises(ConfigurationError, match="solana="):
        Bridge(FakeAleo(), solana=object())


def test_status_omits_the_solana_row_when_the_environment_has_no_solana_chain():
    """Only mainnet has a Solana chain; a testnet client must not invent a "solana" row."""
    bridge = Bridge(FakeAleo(network_name="testnet"), solana=FakeSolanaClient())
    assert [c.chain_id for c in bridge.status().chains if c.chain_id == "solana"] == []


def test_status_derives_the_solana_chain_and_native_asset_from_the_registry():
    keypair = Keypair()
    bridge = Bridge(FakeAleo(), solana=Solana(client=FakeSolanaClient(balance=7), signer=keypair))
    chain = [c for c in bridge.registry.chains(environment="mainnet") if c.family == "solana"][0]
    native = [a for a in bridge.registry.assets(chain=chain.id) if a.kind == "native"][0]
    row = [c for c in bridge.status().chains if c.chain_id == chain.id][0]
    assert row.balances == {native.id: 7}


def test_from_env_builds_the_solana_connection(monkeypatch):
    key = b58encode(bytes(Keypair()))
    seen = {}

    def capture_init(self, aleo, *, ethereum=None, solana=None, environment=None, registry=None, checkpoints=None):
        seen["solana"] = solana

    monkeypatch.setattr(Bridge, "__init__", capture_init)
    monkeypatch.setattr("aleo_bridge.client.build_aleo", lambda *args, **kwargs: FakeAleo())
    for var in ("EVM_PRIVATE_KEY", "ETHEREUM_RPC_URL", "BRIDGE_EVM_PRIVATE_KEY", "BRIDGE_LIVE_ETHEREUM_RPC_URL",
                "BRIDGE_SOLANA_PRIVATE_KEY", "BRIDGE_LIVE_SOLANA_RPC_URL", "BRIDGE_CHECKPOINT_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("BRIDGE_PRIVATE_KEY", "APrivateKey1zkp8CZNn3yeCseEtxuVPbDCwSyhGW6yZKUYKfgXmcpoGPWH")
    monkeypatch.setenv("SOLANA_PRIVATE_KEY", key)
    monkeypatch.setenv("SOLANA_RPC_URL", "https://rpc.example")
    Bridge.from_env()
    assert isinstance(seen["solana"], Solana) and seen["solana"].can_sign and seen["solana"].rpc_url == "https://rpc.example"
    override = Solana(client=FakeSolanaClient())
    Bridge.from_env(solana=override)
    assert seen["solana"] is override


def test_package_exports():
    assert aleo_bridge.Solana is Solana
    assert aleo_bridge.SolModule is SolModule
    assert aleo_bridge.DEFAULT_SOLANA_RPC_URL == "https://api.mainnet-beta.solana.com"
    from aleo_bridge._calls import AleoCall, EvmCall, SolCall
    from aleo_bridge.eth import EthModule
    from aleo_bridge.freezelist import FreezeList
    from aleo_bridge.hyperlane import HyperlaneModule
    from aleo_bridge.profile import Profile
    from aleo_bridge.xreserve import XReserveModule

    assert aleo_bridge.SolCall is SolCall
    # pre-existing exports (plans 1/2/4) must still be exported after this task's __init__.py edit
    assert aleo_bridge.EthModule is EthModule
    assert aleo_bridge.HyperlaneModule is HyperlaneModule
    assert aleo_bridge.XReserveModule is XReserveModule
    assert aleo_bridge.AleoCall is AleoCall
    assert aleo_bridge.EvmCall is EvmCall
    assert aleo_bridge.FreezeList is FreezeList
    assert aleo_bridge.Profile is Profile
