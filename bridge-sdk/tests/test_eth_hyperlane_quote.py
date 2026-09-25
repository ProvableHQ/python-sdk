import dataclasses

import pytest
from eth_account import Account

from aleo_bridge.encoding import aleo_address_to_bytes32, bytes32_to_aleo_address
from aleo_bridge.errors import (AmbiguousRouteError, BridgeError, ChainMismatchError, ConfigurationError,
                                InvalidAmountError, InvalidRecipientError, RouteUnavailableError)
from aleo_bridge.eth import Ethereum
from aleo_bridge.registry import DEFAULT_REGISTRY
from aleo_bridge.types import EvmHyperlaneQuote
from tests.fakes.fake_web3 import ZERO_ADDRESS, fake_web3, make_bridge

KEY = "0x" + "11" * 32
ACCT = Account.from_key(KEY)
ALEO = "aleo1kypwp5m7qtk9mwazgcpg0tq8aal23mnrvwfvug65qgcg9xvsrqgspyjm6n"
ALEO_BYTES32 = "b102e0d37e02ec5dbba2460287ac07ef7ea8ee636392ce235402308299901811"
VEIL_RECIPIENT_BYTES32 = "20e3629764d5338f74bee96675801b1fb29d1fc68b177668f9175708bef84311"
ETH_ROUTER = "0x38D447694f5c1f773ae3132cf93bF30B7Ec1Fa5A"
WBTC, WBTC_ROUTER = "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599", "0x20CDC85778b732073F7EecEF3DF25c0d310f8772"
USDT, USDT_ROUTER = "0xdAC17F958D2ee523a2206206994597C13D831ec7", "0x3C2064D78e4578E8F936E3db42aEF044E33FBF31"


def eth_module(*, signed=True, **config):
    w3 = fake_web3(**config)
    conn = Ethereum(w3=w3, private_key=KEY) if signed else Ethereum(w3=w3)
    return make_bridge(ethereum=conn).eth, w3


def test_native_eth_quote_splits_fee_from_value():
    eth, _ = eth_module(quotes={ETH_ROUTER: [(ZERO_ADDRESS, 69_000_000_000_101)]})
    q = eth.quote_transfer_remote("ethereum/eth", ALEO, amount_atomic=100)
    assert isinstance(q, EvmHyperlaneQuote) and q.kind == "evm-hyperlane"
    assert q.plan.route_id == "hyperlane:ethereum/eth->aleo/eth" and q.plan.amount == "0.0000000000000001"
    assert q.plan.sender == ACCT.address and q.amount_out == "0.0000000000000001"
    assert q.native_value_atomic == 69_000_000_000_101 and q.native_fee_atomic == 69_000_000_000_001
    assert q.approval_required is None
    assert q.recipient_bytes32.hex() == ALEO_BYTES32
    assert len(q.fees) == 1 and q.fees[0].kind == "network" and q.fees[0].chain_id == "ethereum"
    assert q.fees[0].asset_id == "ethereum/eth" and q.fees[0].amount == "0.000069000000000001" and q.fees[0].estimated


def test_recipient_bytes32_matches_veil_vector():
    eth, _ = eth_module(quotes={ETH_ROUTER: [(ZERO_ADDRESS, 1_000)]})
    recipient = bytes32_to_aleo_address(bytes.fromhex(VEIL_RECIPIENT_BYTES32))
    q = eth.quote_transfer_remote("eth", recipient, amount_atomic=1)
    assert q.recipient_bytes32 == bytes.fromhex(VEIL_RECIPIENT_BYTES32)
    assert aleo_address_to_bytes32(recipient) == q.recipient_bytes32
    assert q.plan.recipient == recipient


def test_collateral_wbtc_quote_reads_allowance():
    eth, _ = eth_module(quotes={WBTC_ROUTER: [(ZERO_ADDRESS, 50_000), (WBTC, 100_000)]},
                        allowances={(WBTC, ACCT.address, WBTC_ROUTER): 0})
    q = eth.quote_transfer_remote("ethereum/wbtc", ALEO, amount="0.001")
    assert q.plan.amount_atomic == 100_000 and q.native_value_atomic == 50_000 and q.native_fee_atomic == 50_000
    assert q.approval_required is True
    eth, _ = eth_module(quotes={WBTC_ROUTER: [(ZERO_ADDRESS, 50_000), (WBTC, 100_000)]},
                        allowances={(WBTC, ACCT.address, WBTC_ROUTER): 100_000})
    assert eth.quote_transfer_remote("wbtc", ALEO, amount_atomic=100_000).approval_required is False


def test_read_only_connection_quotes_with_explicit_or_no_sender():
    eth, _ = eth_module(signed=False, quotes={USDT_ROUTER: [(ZERO_ADDRESS, 50_000), (USDT, 1_000_000)]},
                        allowances={(USDT, ACCT.address, USDT_ROUTER): 1})
    q = eth.quote_transfer_remote("usdt", ALEO, amount="1")
    assert q.approval_required is None and q.plan.sender is None
    q = eth.quote_transfer_remote("usdt", ALEO, amount="1", sender=ACCT.address)
    assert q.approval_required is True and q.plan.sender == ACCT.address


def test_quote_must_cover_amount():
    eth, _ = eth_module(quotes={ETH_ROUTER: [(ZERO_ADDRESS, 50)]})
    with pytest.raises(BridgeError, match="Native Hyperlane quote does not cover"):
        eth.quote_transfer_remote("eth", ALEO, amount_atomic=100)
    eth, _ = eth_module(quotes={WBTC_ROUTER: [(ZERO_ADDRESS, 50_000), (WBTC, 99_999)]})
    with pytest.raises(BridgeError, match="Collateral Hyperlane quote does not cover"):
        eth.quote_transfer_remote("wbtc", ALEO, amount_atomic=100_000)


def test_wrong_chain_is_refused_before_any_contract_read():
    eth, w3 = eth_module(chain_id=11155111, quotes={ETH_ROUTER: [(ZERO_ADDRESS, 10**15)]})
    with pytest.raises(ChainMismatchError, match="expected 1"):
        eth.quote_transfer_remote("eth", ALEO, amount_atomic=1)
    assert "eth_call" not in w3.provider.methods


def test_unavailable_unknown_and_explicit_routes():
    eth, _ = eth_module(quotes={ETH_ROUTER: [(ZERO_ADDRESS, 1_000)]})
    with pytest.raises(RouteUnavailableError):
        eth.quote_transfer_remote("ethereum/usad", ALEO, amount_atomic=1)
    with pytest.raises(BridgeError):
        eth.quote_transfer_remote("ethereum/doge", ALEO, amount_atomic=1)
    with pytest.raises(BridgeError, match="not a Hyperlane route"):
        eth.quote_transfer_remote("ethereum/usdc", ALEO, amount_atomic=1)
    route = DEFAULT_REGISTRY.route("hyperlane:ethereum/eth->aleo/eth")
    assert eth.quote_transfer_remote("eth", ALEO, amount_atomic=1, route=route).plan.route_id == route.id


def _corrupted_eth_route(**overrides):
    route = DEFAULT_REGISTRY.route("hyperlane:ethereum/eth->aleo/eth")
    return dataclasses.replace(route, metadata={**route.metadata, **overrides})


def test_corrupted_router_address_is_refused_before_any_contract_read():
    eth, w3 = eth_module(quotes={ETH_ROUTER: [(ZERO_ADDRESS, 10**15)]})
    route = _corrupted_eth_route(routerAddress="not-an-address")
    with pytest.raises(ConfigurationError, match="routerAddress"):
        eth.quote_transfer_remote("eth", ALEO, amount_atomic=1, route=route)
    assert "eth_call" not in w3.provider.methods


def test_bad_router_type_is_refused_before_any_contract_read():
    eth, w3 = eth_module(quotes={ETH_ROUTER: [(ZERO_ADDRESS, 10**15)]})
    route = _corrupted_eth_route(routerType="burn")
    with pytest.raises(ConfigurationError, match="routerType"):
        eth.quote_transfer_remote("eth", ALEO, amount_atomic=1, route=route)
    assert "eth_call" not in w3.provider.methods


def test_out_of_range_destination_domain_is_refused_before_any_contract_read():
    eth, w3 = eth_module(quotes={ETH_ROUTER: [(ZERO_ADDRESS, 10**15)]})
    route = _corrupted_eth_route(destinationDomain=2**32)
    with pytest.raises(ConfigurationError, match="destinationDomain"):
        eth.quote_transfer_remote("eth", ALEO, amount_atomic=1, route=route)
    assert "eth_call" not in w3.provider.methods


def test_bad_registry_commit_is_refused_before_any_contract_read():
    eth, w3 = eth_module(quotes={ETH_ROUTER: [(ZERO_ADDRESS, 10**15)]})
    route = _corrupted_eth_route(registryCommit="not-hex")
    with pytest.raises(ConfigurationError, match="registryCommit"):
        eth.quote_transfer_remote("eth", ALEO, amount_atomic=1, route=route)
    assert "eth_call" not in w3.provider.methods


def test_amount_and_recipient_validation():
    eth, _ = eth_module(quotes={ETH_ROUTER: [(ZERO_ADDRESS, 1_000)]})
    with pytest.raises(InvalidAmountError):
        eth.quote_transfer_remote("eth", ALEO, amount="1", amount_atomic=1)
    with pytest.raises(InvalidAmountError):
        eth.quote_transfer_remote("eth", ALEO)
    with pytest.raises(InvalidRecipientError):
        eth.quote_transfer_remote("eth", "aleo1notanaddress", amount_atomic=1)
    with pytest.raises(InvalidRecipientError):
        eth.quote_transfer_remote("eth", "0x0000000000000000000000000000000000000001", amount_atomic=1)


def test_quote_without_a_plan_or_a_recipient_names_the_missing_recipient():
    """``recipient`` is only optional when ``plan=`` supplies it; otherwise it must be named,
    not surface as the opaque TypeError ``_recipient_bytes32(None)`` used to raise."""
    eth, w3 = eth_module(quotes={ETH_ROUTER: [(ZERO_ADDRESS, 1_000)]})
    with pytest.raises(InvalidRecipientError, match="recipient is required when no plan is given"):
        eth.quote_transfer_remote("eth", amount_atomic=100)
    assert w3.provider.methods == []                    # refused before any contract read
