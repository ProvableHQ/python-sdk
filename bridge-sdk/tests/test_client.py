import json

import pytest
from eth_account import Account as EthAccount

from aleo.facade.errors import ProgramNotFound

from aleo_bridge import Bridge, __main__ as cli
from aleo_bridge._calls import AleoCall
from aleo_bridge.errors import ConfigurationError
from aleo_bridge.eth import Ethereum, EthModule
from aleo_bridge.freezelist import FreezeList
from aleo_bridge.hyperlane import HyperlaneModule
from aleo_bridge.privacy import PrivacyModule
from aleo_bridge.registry import DEFAULT_REGISTRY, Registry
from aleo_bridge.types import BridgeStatus
from aleo_bridge.xreserve import XReserveModule
from tests.conftest import SIGNER, FakeAleo, default_mappings
from tests.fakes.fake_web3 import fake_web3


def test_construction_defaults_and_namespaces(fake_aleo):
    bridge = Bridge(fake_aleo)
    assert (bridge.environment, bridge.network, bridge.registry) == ("mainnet", "mainnet", DEFAULT_REGISTRY)
    assert bridge.checkpoints is None and bridge.ethereum is None and bridge.solana is None and bridge.profile is None
    assert isinstance(bridge.hyperlane, HyperlaneModule) and isinstance(bridge.xreserve, XReserveModule)
    assert isinstance(bridge.freezelist, FreezeList) and isinstance(bridge.privacy, PrivacyModule)
    assert bridge.aleo_chain().id == "aleo" and bridge.aleo_address() == SIGNER
    assert Bridge(FakeAleo(network_name="testnet")).aleo_chain().id == "aleo-testnet"


def test_construction_errors(fake_aleo):
    with pytest.raises(ConfigurationError, match="does not match"):
        Bridge(fake_aleo, environment="testnet")
    with pytest.raises(ConfigurationError, match="mainnet or testnet"):
        Bridge(FakeAleo(network_name="devnet"))
    empty = Registry(DEFAULT_REGISTRY.version, DEFAULT_REGISTRY.chains(environment="testnet"),
                     DEFAULT_REGISTRY.assets(environment="testnet"), DEFAULT_REGISTRY.routes(environment="testnet"))
    with pytest.raises(ConfigurationError, match="no chains for mainnet"):
        Bridge(fake_aleo, registry=empty)
    with pytest.raises(ConfigurationError, match="default_account"):
        Bridge(FakeAleo(default_account=False)).aleo_address()


def test_eth_property_wraps_connection_and_sol_property_now_wraps_a_bare_client(fake_aleo):
    """Plan 3 (task 5) landed ``SolModule``, so ``bridge.sol`` on a configured connection now
    succeeds instead of degrading to ``MissingExtraError`` (that fallback covered the window
    before ``SolModule`` existed; ``sol.py``'s ``try/except ImportError`` around the import is
    still exercised by ``test_sol_connection.py``'s no-solders scenarios)."""
    bridge = Bridge(fake_aleo)
    with pytest.raises(ConfigurationError, match="ethereum="):
        bridge.eth
    with pytest.raises(ConfigurationError, match="solana="):
        bridge.sol
    eth_module = Bridge(fake_aleo, ethereum=fake_web3()).eth      # plan 2: real Ethereum wraps a bare Web3
    assert isinstance(eth_module, EthModule)
    pytest.importorskip("solders")
    from aleo_bridge.sol import SolModule
    from tests.fakes.fake_solana import FakeSolanaClient

    sol_module = Bridge(fake_aleo, solana=FakeSolanaClient()).sol  # plan 3: bare client is wrapped in Solana
    assert isinstance(sol_module, SolModule)


def test_program_cache_mapping_value_and_call_registration(fake_aleo):
    fake_aleo.imports = {"hyp_warp_token_wbtc_v2.aleo": ["token_registry.aleo", "hyp_mailbox.aleo"], "hyp_mailbox.aleo": ["token_registry.aleo"]}
    bridge = Bridge(fake_aleo)
    assert bridge.program("credits.aleo") is bridge.program("credits.aleo") and fake_aleo.fetched.count("credits.aleo") == 1
    assert bridge.mapping_value("credits.aleo", "account", SIGNER) == "2392443u64"
    assert bridge.mapping_value("credits.aleo", "account", "aleo1nobody") is None
    fake_aleo.mappings["credits.aleo"]["account"]["aleo1null"] = "null"
    fake_aleo.mappings["credits.aleo"]["account"]["aleo1quoted"] = '"7u64"'
    assert bridge.mapping_value("credits.aleo", "account", "aleo1null") is None
    assert bridge.mapping_value("credits.aleo", "account", "aleo1quoted") == "7u64"
    call = bridge._call("hyp_warp_token_wbtc_v2.aleo", "transfer_remote", ["1u128"], lambda tx, outs: tx)
    assert isinstance(call, AleoCall) and fake_aleo.registered == []
    call.simulate()
    assert fake_aleo.registered == ["token_registry.aleo", "hyp_mailbox.aleo", "hyp_warp_token_wbtc_v2.aleo"]  # dependencies first, root last


def test_mapping_value_returns_none_for_missing_program(fake_aleo):
    fake_aleo.missing_programs = {"missing.aleo"}
    bridge = Bridge(fake_aleo)
    assert bridge.mapping_value("missing.aleo", "balances", SIGNER) is None
    with pytest.raises(ProgramNotFound):
        bridge.program("missing.aleo")


def test_amount_helpers_and_privacy_delegation(fake_aleo):
    from aleo_bridge.freezelist import EMPTY_TREE_ROOT

    # unshield()'s default (unsupplied) merkle_proof= resolves through freezelist.exclusion_proof(),
    # which now requires a readable on-chain root (item 4) — seed the empty-list root.
    fake_aleo.mappings.setdefault("usdcx_freezelist.aleo", {})["freeze_list_root"] = {"1u8": f"{EMPTY_TREE_ROOT}field"}
    bridge = Bridge(fake_aleo)
    assert bridge.to_atomic("0.001", "aleo/wbtc") == 100_000 and bridge.from_atomic(100_000, ("aleo", "wbtc")) == "0.001"
    assert bridge.to_atomic("1", DEFAULT_REGISTRY.asset("ethereum/usdc")) == 1_000_000
    assert bridge.shield("aleo/eth", amount="1").function_name == "shield"
    assert bridge.unshield("aleo/usdcx", amount="2.5").function_name == "transfer_private_to_public"


def test_status_reads_every_aleo_asset_balance(fake_aleo):
    status = Bridge(fake_aleo).status()
    assert isinstance(status, BridgeStatus) and status.environment == "mainnet" and status.registry_version == DEFAULT_REGISTRY.version
    assert status.pending == [] and len(status.chains) == 1
    chain = status.chains[0]
    assert (chain.chain_id, chain.address, chain.can_sign) == ("aleo", SIGNER, True)
    assert chain.balances == {"aleo/aleo": 2392443, "aleo/usdcx": 1000000, "aleo/eth": 0, "aleo/wbtc": 10000,
                              "aleo/usdt": 0, "aleo/sol": 0, "aleo/usad": 0}
    assert "arc20_usdt.aleo" in fake_aleo.fetched and "usad_stablecoin.aleo" in fake_aleo.fetched
    unsigned = Bridge(FakeAleo(mappings=default_mappings(), default_account=False)).status().chains[0]
    assert (unsigned.address, unsigned.can_sign) == (None, False) and set(unsigned.balances.values()) == {0}


def test_from_env_builds_aleo_only(monkeypatch, fake_aleo):
    captured = {}

    def fake_build(endpoint, network, private_key, *, api_key=None, consumer_id=None):
        captured.update(endpoint=endpoint, network=network, private_key=private_key, api_key=api_key, consumer_id=consumer_id)
        return FakeAleo(mappings=default_mappings(), network_name=network)

    monkeypatch.setattr("aleo_bridge.client.build_aleo", fake_build)
    for var in ("BRIDGE_PRIVATE_KEY", "ALEO_ENDPOINT", "ALEO_NETWORK", "ALEO_API_KEY", "ALEO_CONSUMER_ID", "EVM_PRIVATE_KEY",
                "ETHEREUM_RPC_URL", "BRIDGE_EVM_PRIVATE_KEY", "BRIDGE_LIVE_ETHEREUM_RPC_URL",
                "SOLANA_PRIVATE_KEY", "SOLANA_RPC_URL", "BRIDGE_CHECKPOINT_DIR"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ConfigurationError, match="BRIDGE_PRIVATE_KEY"):
        Bridge.from_env()
    monkeypatch.setenv("BRIDGE_PRIVATE_KEY", "APrivateKey1zkpTest")
    bridge = Bridge.from_env()
    assert captured == {"endpoint": "https://edge.provable.com/api", "network": "mainnet", "private_key": "APrivateKey1zkpTest",
                        "api_key": None, "consumer_id": None}
    assert bridge.environment == "mainnet" and bridge.ethereum is None and bridge.solana is None and bridge.checkpoints is None
    monkeypatch.setenv("ALEO_NETWORK", "testnet")
    monkeypatch.setenv("ALEO_ENDPOINT", "https://api.provable.com/v2")
    monkeypatch.setenv("ALEO_API_KEY", "k")
    monkeypatch.setenv("ALEO_CONSUMER_ID", "c")
    assert Bridge.from_env().environment == "testnet"
    assert (captured["endpoint"], captured["api_key"], captured["consumer_id"]) == ("https://api.provable.com/v2", "k", "c")
    marker = object()
    assert Bridge.from_env(ethereum=None, solana=None).ethereum is None
    assert Bridge.from_env(checkpoints=marker).checkpoints is marker
    with pytest.raises(TypeError, match="unexpected"):
        Bridge.from_env(w3=marker)


def test_from_env_side_chain_variables(monkeypatch, tmp_path):
    monkeypatch.setattr("aleo_bridge.client.build_aleo", lambda *a, **k: FakeAleo(mappings=default_mappings()))
    monkeypatch.setenv("BRIDGE_PRIVATE_KEY", "APrivateKey1zkpTest")
    for var in ("EVM_PRIVATE_KEY", "ETHEREUM_RPC_URL", "BRIDGE_EVM_PRIVATE_KEY", "BRIDGE_LIVE_ETHEREUM_RPC_URL",
                "SOLANA_PRIVATE_KEY", "SOLANA_RPC_URL", "BRIDGE_CHECKPOINT_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("EVM_PRIVATE_KEY", "0x" + "11" * 32)
    with pytest.raises(ConfigurationError, match="both EVM_PRIVATE_KEY and ETHEREUM_RPC_URL"):
        Bridge.from_env()
    monkeypatch.setenv("ETHEREUM_RPC_URL", "https://eth.example")
    bridge = Bridge.from_env()                                   # plan 2: real Ethereum connection now constructed
    assert isinstance(bridge.ethereum, Ethereum)
    assert bridge.ethereum.address == EthAccount.from_key("0x" + "11" * 32).address
    monkeypatch.delenv("EVM_PRIVATE_KEY")
    monkeypatch.delenv("ETHEREUM_RPC_URL")
    pytest.importorskip("solders")
    from solders.keypair import Keypair

    from aleo_bridge._base58 import b58encode
    from aleo_bridge.sol import Solana

    solana_key = Keypair()
    monkeypatch.setenv("SOLANA_PRIVATE_KEY", b58encode(bytes(solana_key)))
    bridge = Bridge.from_env()                                   # plan 3/4 (Task 4): real Solana connection now constructed
    assert isinstance(bridge.solana, Solana)
    assert bridge.solana.can_sign and bridge.solana.address == str(solana_key.pubkey())
    monkeypatch.delenv("SOLANA_PRIVATE_KEY")
    monkeypatch.setenv("BRIDGE_CHECKPOINT_DIR", str(tmp_path / "cp"))
    bridge = Bridge.from_env()                                   # plan 4: FileCheckpointStore now wired
    from aleo_bridge.checkpoint import FileCheckpointStore
    assert isinstance(bridge.checkpoints, FileCheckpointStore)
    assert bridge.checkpoints.directory == tmp_path / "cp" and bridge.checkpoints.directory.is_dir()


def test_build_aleo_is_local_only():
    from aleo import testnet as net
    from aleo_bridge.client import build_aleo
    key = net.PrivateKey.random()
    aleo = build_aleo("https://edge.provable.com/api", "testnet", str(key))
    assert aleo.network_name == "testnet" and str(aleo.default_account.address) == str(key.address)


def test_from_profile_uses_profile_and_wires_no_side_chains(tmp_path, monkeypatch):
    captured = {}

    def fake_build(endpoint, network, private_key, *, api_key=None, consumer_id=None):
        captured.update(endpoint=endpoint, network=network, private_key=private_key)
        return FakeAleo(mappings=default_mappings(), network_name=network)

    monkeypatch.setattr("aleo_bridge.client.build_aleo", fake_build)
    for var in ("BRIDGE_PRIVATE_KEY", "BRIDGE_PRIVATE_KEY_FILE", "EVM_PRIVATE_KEY", "ETHEREUM_RPC_URL",
                "BRIDGE_EVM_PRIVATE_KEY", "BRIDGE_LIVE_ETHEREUM_RPC_URL", "SOLANA_PRIVATE_KEY", "ALEO_API_KEY", "ALEO_CONSUMER_ID"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ALEO_BRIDGE_HOME", str(tmp_path / "home"))
    bridge = Bridge.from_profile(network="testnet", endpoint="https://api.provable.com/v2")
    assert bridge.profile is not None and bridge.profile.home == tmp_path / "home"
    assert captured == {"endpoint": "https://api.provable.com/v2", "network": "testnet", "private_key": bridge.profile.private_key}
    assert bridge.environment == "testnet" and bridge.ethereum is None and bridge.solana is None
    from aleo_bridge.checkpoint import FileCheckpointStore
    assert isinstance(bridge.checkpoints, FileCheckpointStore)   # plan 4 binds FileCheckpointStore(profile.checkpoint_dir)
    assert bridge.checkpoints.directory == bridge.profile.checkpoint_dir
    assert bridge.profile.checkpoint_dir.is_dir()
    marker = Ethereum(w3=fake_web3())
    assert Bridge.from_profile(ethereum=marker).ethereum is marker


def test_cli_lists_routes_and_assets(capsys):
    assert cli.main(["routes"]) == 0
    routes = json.loads(capsys.readouterr().out)
    assert len(routes) == 22 and routes[0] == "xreserve:ethereum/usdc->aleo/usdcx"
    assert cli.main(["assets"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 19
    assert cli.main(["bogus"]) == 2
