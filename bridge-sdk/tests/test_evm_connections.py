from unittest.mock import patch

import pytest

from aleo_bridge import Bridge, Ethereum
from aleo_bridge.errors import ConfigurationError, ChainMismatchError
from tests.conftest import FakeAleo, default_mappings
from tests.fakes.fake_web3 import fake_web3

KEY = '0x' + '11' * 32


def make(**kwargs):
    return Bridge(FakeAleo(mappings=default_mappings()), **kwargs)


def test_route_selected_connections_preserve_ethereum_alias():
    eth, arc = Ethereum(w3=fake_web3(chain_id=1)), Ethereum(w3=fake_web3(chain_id=5042))
    b = make(ethereum=eth, evm={'ARC': arc})
    assert b.eth.conn is eth and b.evm('ethereum') is b.eth
    assert b.evm('arc').conn is arc and b.evm('arc').chain.id == 'arc'


def test_duplicate_connection_cannot_silently_select_signer():
    with pytest.raises(ConfigurationError, match='(?i)conflict'):
        make(ethereum=Ethereum(w3=fake_web3()), evm={'ethereum': Ethereum(w3=fake_web3())})


@pytest.mark.parametrize('chain', ['solana', 'aleo', 'sepolia'])
def test_map_rejects_family_or_environment_mismatch(chain):
    with pytest.raises(ConfigurationError):
        make(evm={chain: Ethereum(w3=fake_web3())})


def test_arc_status_reads_arc_token_without_double_counting_native_view():
    from eth_account import Account
    address = Account.from_key(KEY).address
    w3 = fake_web3(chain_id=5042, eth_balances={address: 10**18},
                   token_balances={('0x3600000000000000000000000000000000000000', address): 10**6})
    b = make(evm={'arc': Ethereum(w3=w3, private_key=KEY)})
    arc = next(c for c in b.status().chains if c.chain_id == 'arc')
    assert arc.balances == {'arc/usdc': 10**6}
    assert b.evm('arc')._native_fee(10**16).amount == '0.01'


def test_arc_connection_rejects_ethereum_rpc_before_status_reads():
    b = make(evm={'arc': Ethereum(w3=fake_web3(chain_id=1), private_key=KEY)})
    with pytest.raises(ChainMismatchError):
        b.evm('arc').chain_status()


def test_arc_only_from_env_does_not_require_ethereum_rpc(monkeypatch):
    monkeypatch.setenv('BRIDGE_PRIVATE_KEY', 'fake')
    monkeypatch.setenv('EVM_PRIVATE_KEY', KEY)
    monkeypatch.setenv('ARC_RPC_URL', 'https://arc.invalid')
    for key in ('ETHEREUM_RPC_URL', 'BRIDGE_LIVE_ETHEREUM_RPC_URL', 'BASE_RPC_URL', 'ARBITRUM_RPC_URL'):
        monkeypatch.delenv(key, raising=False)
    with patch('aleo_bridge.client.build_aleo', return_value=FakeAleo(mappings=default_mappings())):
        b = Bridge.from_env()
    assert b.ethereum is None
    assert b.evm('arc').conn.address is not None


def test_explicit_arc_override_wins_over_environment(monkeypatch):
    monkeypatch.setenv('BRIDGE_PRIVATE_KEY', 'fake')
    monkeypatch.setenv('ARC_RPC_URL', 'https://unused.invalid')
    arc = Ethereum(w3=fake_web3(chain_id=5042))
    with patch('aleo_bridge.client.build_aleo', return_value=FakeAleo(mappings=default_mappings())):
        b = Bridge.from_env(ethereum=None, solana=None, evm={'arc': arc})
    assert b.evm('arc').conn is arc


def test_explicit_ethereum_map_override_wins_over_environment(monkeypatch):
    monkeypatch.setenv('BRIDGE_PRIVATE_KEY','fake')
    monkeypatch.setenv('ETHEREUM_RPC_URL','https://unused.invalid')
    monkeypatch.setenv('EVM_PRIVATE_KEY',KEY)
    connection = Ethereum(w3=fake_web3())
    with patch('aleo_bridge.client.build_aleo',return_value=FakeAleo(mappings=default_mappings())):
        b = Bridge.from_env(evm={'ethereum':connection},solana=None)
    assert b.ethereum is connection


def test_profile_ethereum_map_override_wins_over_environment(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setenv('ETHEREUM_RPC_URL','https://unused.invalid')
    monkeypatch.setenv('EVM_PRIVATE_KEY',KEY)
    connection = Ethereum(w3=fake_web3())
    profile = SimpleNamespace(endpoint='https://unused.invalid',network='mainnet',private_key='fake')
    with patch('aleo_bridge.client.build_aleo',return_value=FakeAleo(mappings=default_mappings())), \
         patch('aleo_bridge.client.Profile.load_or_create',return_value=profile), \
         patch('aleo_bridge.client._checkpoints_for_profile',return_value=None):
        b = Bridge.from_profile(evm={'ethereum':connection})
    assert b.ethereum is connection
