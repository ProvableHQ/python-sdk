import pytest
from eth_account import Account
from web3 import HTTPProvider, Web3

from aleo_bridge import Bridge, Ethereum, EthModule, EvmCall
from aleo_bridge.errors import ChainMismatchError, ConfigurationError
from aleo_bridge.types import BridgeStatus, ChainStatus
from tests.fakes.fake_web3 import fake_web3, make_bridge

KEY = "0x" + "11" * 32
ACCT = Account.from_key(KEY)
WBTC, USDC, USDT = ("0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599", "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                    "0xdAC17F958D2ee523a2206206994597C13D831ec7")


def test_package_exports():
    assert EthModule is not None and EvmCall is not None and Ethereum is not None


def test_eth_property_requires_a_connection():
    bridge = make_bridge()
    assert bridge.ethereum is None
    with pytest.raises(ConfigurationError,
                       match=r"Pass ethereum=Ethereum\(\.\.\.\) to Bridge\(\.\.\.\) or set EVM_PRIVATE_KEY \+ ETHEREUM_RPC_URL"):
        bridge.eth


def test_eth_property_is_cached_module_bound_to_connection():
    conn = Ethereum(w3=fake_web3(), private_key=KEY)
    bridge = make_bridge(ethereum=conn)
    assert bridge.ethereum is conn and isinstance(bridge.eth, EthModule)
    assert bridge.eth is bridge.eth and bridge.eth.conn is conn and bridge.eth.chain.id == "ethereum"
    assert make_bridge(environment="testnet", ethereum=Ethereum(w3=fake_web3(chain_id=11155111))).eth.chain.id == "sepolia"


def test_bare_web3_is_wrapped_read_only_unless_default_account():
    w3 = fake_web3()
    bridge = make_bridge(ethereum=w3)
    assert isinstance(bridge.ethereum, Ethereum) and bridge.ethereum.w3 is w3 and not bridge.ethereum.can_sign
    w3.eth.default_account = ACCT.address
    assert make_bridge(ethereum=w3).ethereum.address == ACCT.address
    with pytest.raises(ConfigurationError, match="Ethereum connection or a web3.Web3"):
        make_bridge(ethereum="https://not-a-client")


def test_from_env_requires_both_evm_variables(monkeypatch):
    from aleo import PrivateKey

    monkeypatch.setenv("BRIDGE_PRIVATE_KEY", str(PrivateKey.random()))
    monkeypatch.delenv("SOLANA_PRIVATE_KEY", raising=False)
    monkeypatch.delenv("BRIDGE_CHECKPOINT_DIR", raising=False)
    monkeypatch.delenv("BRIDGE_EVM_PRIVATE_KEY", raising=False)
    monkeypatch.delenv("BRIDGE_LIVE_ETHEREUM_RPC_URL", raising=False)
    monkeypatch.setenv("EVM_PRIVATE_KEY", KEY)
    monkeypatch.delenv("ETHEREUM_RPC_URL", raising=False)
    with pytest.raises(ConfigurationError, match="both EVM_PRIVATE_KEY and ETHEREUM_RPC_URL"):
        Bridge.from_env()
    monkeypatch.delenv("EVM_PRIVATE_KEY")
    monkeypatch.setenv("ETHEREUM_RPC_URL", "http://127.0.0.1:1")
    with pytest.raises(ConfigurationError, match="both EVM_PRIVATE_KEY and ETHEREUM_RPC_URL"):
        Bridge.from_env()
    monkeypatch.setenv("EVM_PRIVATE_KEY", KEY)
    bridge = Bridge.from_env()
    assert bridge.ethereum.address == ACCT.address and isinstance(bridge.ethereum.w3.provider, HTTPProvider)
    assert bridge.ethereum.w3.provider.endpoint_uri == "http://127.0.0.1:1"
    override = Ethereum(w3=fake_web3())
    assert Bridge.from_env(ethereum=override).ethereum is override
    monkeypatch.delenv("EVM_PRIVATE_KEY")
    monkeypatch.delenv("ETHEREUM_RPC_URL")
    assert Bridge.from_env().ethereum is None


def test_chain_status_reads_native_and_erc20_balances():
    w3 = fake_web3(eth_balances={ACCT.address: 5}, token_balances={(WBTC, ACCT.address): 7, (USDC, ACCT.address): 2_000_000})
    eth = make_bridge(ethereum=Ethereum(w3=w3, private_key=KEY)).eth
    status = eth.chain_status()
    assert isinstance(status, ChainStatus) and status.chain_id == "ethereum" and status.address == ACCT.address and status.can_sign
    assert status.balances == {"ethereum/eth": 5, "ethereum/usdc": 2_000_000, "ethereum/wbtc": 7, "ethereum/usdt": 0}
    read_only = make_bridge(ethereum=Ethereum(w3=fake_web3())).eth.chain_status()
    assert read_only.address is None and not read_only.can_sign and read_only.balances == {}


def test_chain_status_asserts_the_connected_chain(monkeypatch):
    """A Web3 pointed at the wrong network must fail chain_status() (and therefore Bridge.status())
    with ChainMismatchError before any balance is read."""
    w3 = fake_web3(chain_id=999)
    bridge = make_bridge(ethereum=Ethereum(w3=w3))
    with pytest.raises(ChainMismatchError, match="expected 1"):
        bridge.eth.chain_status()
    assert "eth_getBalance" not in w3.provider.methods and "eth_call" not in w3.provider.methods
    aleo_status = ChainStatus(chain_id="aleo", address=None, can_sign=False, balances={})
    monkeypatch.setattr(Bridge, "_aleo_chain_status", lambda self: aleo_status)
    with pytest.raises(ChainMismatchError, match="expected 1"):
        bridge.status()


def test_bridge_status_includes_evm_chain(monkeypatch):
    aleo_status = ChainStatus(chain_id="aleo", address="aleo1" + "q" * 58, can_sign=True, balances={})
    monkeypatch.setattr(Bridge, "_aleo_chain_status", lambda self: aleo_status)
    w3 = fake_web3(eth_balances={ACCT.address: 5})
    bridge = make_bridge(ethereum=Ethereum(w3=w3, private_key=KEY))
    status = bridge.status()
    assert isinstance(status, BridgeStatus) and [c.chain_id for c in status.chains] == ["aleo", "ethereum"]
    assert status.chains[1].balances["ethereum/eth"] == 5
    assert [c.chain_id for c in make_bridge().status().chains] == ["aleo"]
