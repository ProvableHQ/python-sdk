import dataclasses

import pytest
from eth_account import Account

from aleo_bridge.encoding import aleo_address_to_bytes32, aleo_program_address, xreserve_hook_data
from aleo_bridge.errors import (BridgeError, ChainMismatchError, ConfigurationError, InsufficientBalanceError,
                                InvalidAmountError)
from aleo_bridge.eth import Ethereum
from aleo_bridge.registry import DEFAULT_REGISTRY
from aleo_bridge.types import EvmXReserveQuote
from tests.fakes.fake_web3 import fake_web3, make_bridge

KEY = "0x" + "11" * 32
ACCT = Account.from_key(KEY)
ALEO = "aleo1kypwp5m7qtk9mwazgcpg0tq8aal23mnrvwfvug65qgcg9xvsrqgspyjm6n"
SEPOLIA_USDC = "0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238"
SEPOLIA_XRESERVE = "0x008888878f94C0d87defdf0B07f46B93C1934442"
MAINNET_USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
MAINNET_XRESERVE = "0x8888888199b2Df864bf678259607d6D5EBb4e3Ce"


def sepolia(*, signed=True, balance=3_000_000, allowance=0, chain_id=11155111):
    w3 = fake_web3(chain_id=chain_id, token_balances={(SEPOLIA_USDC, ACCT.address): balance},
                   allowances={(SEPOLIA_USDC, ACCT.address, SEPOLIA_XRESERVE): allowance})
    conn = Ethereum(w3=w3, private_key=KEY) if signed else Ethereum(w3=w3)
    return make_bridge(environment="testnet", ethereum=conn).eth, w3


def test_record_mode_quote():
    eth, _ = sepolia()
    q = eth.quote_deposit_usdc(ALEO, amount="2", mint_mode="record")
    assert isinstance(q, EvmXReserveQuote) and q.kind == "evm-xreserve"
    assert q.plan.route_id == "xreserve:sepolia/usdc->aleo-testnet/usdcx" and q.plan.mint_mode == "record"
    assert q.plan.amount == "2" and q.plan.amount_atomic == 2_000_000 and q.plan.sender == ACCT.address
    assert q.amount_out == "2" and q.fees == ()
    assert q.hook_data == b"\x01" + bytes(64)
    assert q.remote_recipient_bytes32 == aleo_address_to_bytes32(ALEO)
    assert q.balance_atomic == 3_000_000 and q.allowance_atomic == 0 and q.approval_required is True
    assert q.max_fee_atomic == 100_000


def test_public_mode_is_default_and_allowance_can_cover():
    eth, _ = sepolia(allowance=5_000_000)
    q = eth.quote_deposit_usdc(ALEO, amount_atomic=2_000_000)
    assert q.plan.mint_mode == "public" and q.hook_data == bytes(65) and q.approval_required is False


def test_private_mode_targets_wrapper_program_and_commits_recipient():
    eth, _ = sepolia()
    q = eth.quote_deposit_usdc(ALEO, amount="2", mint_mode="private", secret_nonce="7scalar")
    wrapper = aleo_program_address("shielded_usdcx_wrapper.aleo", "testnet")
    assert q.remote_recipient_bytes32 == aleo_address_to_bytes32(wrapper)
    assert q.hook_data[0] == 2 and len(q.hook_data) == 65 and q.hook_data[1:33] != bytes(32)
    assert q.hook_data == xreserve_hook_data("private", ALEO, "testnet", "7scalar")
    assert q.hook_data != eth.quote_deposit_usdc(ALEO, amount="2", mint_mode="private").hook_data
    assert q.plan.recipient == ALEO            # the plan keeps the intended recipient, not the wrapper


def test_minimum_amount_and_balance_are_enforced():
    eth, _ = sepolia()
    with pytest.raises(InvalidAmountError, match="minimum deposit is 2000000"):
        eth.quote_deposit_usdc(ALEO, amount_atomic=1_999_999)
    with pytest.raises(InsufficientBalanceError, match="USDC"):
        eth.quote_deposit_usdc(ALEO, amount_atomic=3_000_001)


def test_wrong_chain_and_bad_mint_mode():
    eth, w3 = sepolia(chain_id=1)
    with pytest.raises(ChainMismatchError, match="expected 11155111"):
        eth.quote_deposit_usdc(ALEO, amount="2")
    assert "eth_call" not in w3.provider.methods
    eth, _ = sepolia()
    with pytest.raises(BridgeError, match="mint_mode"):
        eth.quote_deposit_usdc(ALEO, amount="2", mint_mode="shielded")


def test_read_only_needs_explicit_sender():
    eth, _ = sepolia(signed=False)
    with pytest.raises(ConfigurationError, match="sender"):
        eth.quote_deposit_usdc(ALEO, amount="2")
    q = eth.quote_deposit_usdc(ALEO, amount="2", sender=ACCT.address)
    assert q.balance_atomic == 3_000_000 and q.plan.sender == ACCT.address


def test_mainnet_environment_selects_ethereum_route():
    w3 = fake_web3(chain_id=1, token_balances={(MAINNET_USDC, ACCT.address): 2_000_000},
                   allowances={(MAINNET_USDC, ACCT.address, MAINNET_XRESERVE): 0})
    eth = make_bridge(environment="mainnet", ethereum=Ethereum(w3=w3, private_key=KEY)).eth
    q = eth.quote_deposit_usdc(ALEO, amount="2")
    assert q.plan.route_id == "xreserve:ethereum/usdc->aleo/usdcx" and q.approval_required is True


def _corrupted_xreserve_route(**overrides):
    route = DEFAULT_REGISTRY.route("xreserve:sepolia/usdc->aleo-testnet/usdcx")
    return dataclasses.replace(route, metadata={**route.metadata, **overrides})


def test_corrupted_xreserve_contract_is_refused_before_any_contract_read():
    eth, w3 = sepolia()
    route = _corrupted_xreserve_route(xReserveContract="not-an-address")
    with pytest.raises(ConfigurationError, match="xReserveContract"):
        eth.quote_deposit_usdc(ALEO, amount="2", route=route)
    assert "eth_call" not in w3.provider.methods


def test_non_hex_remote_token_bytes32_is_refused_before_any_contract_read():
    eth, w3 = sepolia()
    route = _corrupted_xreserve_route(remoteTokenBytes32="not-hex")
    with pytest.raises(ConfigurationError, match="remoteTokenBytes32"):
        eth.quote_deposit_usdc(ALEO, amount="2", route=route)
    assert "eth_call" not in w3.provider.methods


def test_short_remote_token_bytes32_is_refused_before_any_contract_read():
    eth, w3 = sepolia()
    route = _corrupted_xreserve_route(remoteTokenBytes32="0x" + "ab" * 16)   # 16 bytes, not 32
    with pytest.raises(ConfigurationError, match="remoteTokenBytes32"):
        eth.quote_deposit_usdc(ALEO, amount="2", route=route)
    assert "eth_call" not in w3.provider.methods


def test_bridge_program_without_aleo_suffix_is_refused_before_any_contract_read():
    eth, w3 = sepolia()
    route = _corrupted_xreserve_route(bridgeProgram="not_a_program")
    with pytest.raises(ConfigurationError, match="bridgeProgram"):
        eth.quote_deposit_usdc(ALEO, amount="2", route=route)
    assert "eth_call" not in w3.provider.methods


def test_negative_remote_domain_is_refused_before_any_contract_read():
    eth, w3 = sepolia()
    route = _corrupted_xreserve_route(remoteDomain=-1)
    with pytest.raises(ConfigurationError, match="remoteDomain"):
        eth.quote_deposit_usdc(ALEO, amount="2", route=route)
    assert "eth_call" not in w3.provider.methods


def test_non_digit_minimum_amount_atomic_is_refused_before_any_contract_read():
    eth, w3 = sepolia()
    route = _corrupted_xreserve_route(minimumAmountAtomic="2_000_000")
    with pytest.raises(ConfigurationError, match="minimumAmountAtomic"):
        eth.quote_deposit_usdc(ALEO, amount="2", route=route)
    assert "eth_call" not in w3.provider.methods
