import pytest

from aleo_bridge.errors import (ConfigurationError, InvalidAmountError, RouteUnavailableError,
                                UnsupportedRouteError)
from aleo_bridge.lifecycle import quote
from aleo_bridge.types import AleoHyperlaneQuote, AleoXReserveQuote
from tests.fakes.fake_bridge import ALEO_RECIPIENT, EVM_ADDRESS, SOL_ADDRESS, FakeBridge


def test_evm_hyperlane_quote_dispatches_to_eth_and_carries_the_canonical_plan():
    b = FakeBridge()
    q = quote(b, source="ethereum/wbtc", destination="aleo/wbtc", amount="0.001", recipient=ALEO_RECIPIENT,
              sender=EVM_ADDRESS)
    assert q.kind == "evm-hyperlane" and q.plan.route_id == "hyperlane:ethereum/wbtc->aleo/wbtc"
    assert q.plan.sender == EVM_ADDRESS and q.native_value_atomic == 100_000 + 1000
    assert b.calls == [("eth.quote_transfer_remote", {"plan": q.plan})]


def test_evm_xreserve_quote_passes_mint_mode_and_secret_nonce():
    b = FakeBridge()
    q = quote(b, source="ethereum/usdc", destination="aleo/usdcx", amount="2", recipient=ALEO_RECIPIENT,
              mint_mode="private", secret_nonce="7scalar")
    assert q.kind == "evm-xreserve" and q.plan.mint_mode == "private"
    assert b.calls[-1] == ("eth.quote_deposit_usdc", {"plan": q.plan, "secret_nonce": "7scalar"})


def test_solana_hyperlane_quote():
    b = FakeBridge(solana=True)
    q = quote(b, source="solana/sol", destination="aleo/sol", amount="0.000000001", recipient=ALEO_RECIPIENT,
              sender=SOL_ADDRESS)
    assert q.kind == "solana-hyperlane" and q.total_lamports == 1 + 7_914_240
    assert b.calls == [("sol.quote_transfer_remote", {"plan": q.plan})]


def test_aleo_hyperlane_quote_reads_the_igp_only():
    b = FakeBridge(ethereum=False)
    q = quote(b, source="aleo/eth", destination="ethereum/eth", amount="0.000000000000000001",
              recipient=EVM_ADDRESS)
    assert isinstance(q, AleoHyperlaneQuote) and q.kind == "aleo-hyperlane"
    assert q.payment_microcredits == 8_174_147 and q.gas_limit == 44_000
    assert q.amount_out == "0.000000000000000001"
    assert [f.kind for f in q.fees] == ["protocol"] and q.fees[0].amount == "8.174147" and q.fees[0].estimated
    assert q.fees[0].asset_id == "aleo/aleo"
    assert b.calls == [("hyperlane.quote_gas_payment", "aleo/eth")]


def test_aleo_xreserve_quote_is_offline_and_deducts_the_withdrawal_fee():
    b = FakeBridge(ethereum=False)
    q = quote(b, source="aleo/usdcx", destination="ethereum/usdc", amount="2.000001", recipient=EVM_ADDRESS)
    assert isinstance(q, AleoXReserveQuote) and q.kind == "aleo-xreserve"
    assert q.amount_out == "0.000001" and q.withdrawal_fee_atomic == 2_000_000
    # I3: the registry's 2 USDCx literal is a quote ASSUMPTION, not an exact on-chain read — the
    # live testnet fee observed on 2026-09-18 was ~1.0035 USDC — so the fee is flagged estimated
    # and amount_out is a lower bound. The literal still drives the burn-minimum guard below.
    assert q.fees == (q.fees[0],) and q.fees[0].kind == "protocol" and q.fees[0].estimated is True
    assert (q.fees[0].chain_id, q.fees[0].asset_id, q.fees[0].amount) == ("aleo", "aleo/usdcx", "2")
    assert b.calls == []                                  # no network
    with pytest.raises(InvalidAmountError, match="exceed the 2 USDCx withdrawal fee"):
        quote(b, source="aleo/usdcx", destination="ethereum/usdc", amount="2", recipient=EVM_ADDRESS)


def test_missing_connection_and_unavailable_route():
    b = FakeBridge(ethereum=False)
    with pytest.raises(ConfigurationError, match="Ethereum connection"):
        quote(b, source="ethereum/usdc", destination="aleo/usdcx", amount="2", recipient=ALEO_RECIPIENT)
    with pytest.raises(RouteUnavailableError, match="metadata-required"):
        quote(b, source="aleo/aleo", destination="ethereum/aleo", amount="1", recipient=EVM_ADDRESS)
