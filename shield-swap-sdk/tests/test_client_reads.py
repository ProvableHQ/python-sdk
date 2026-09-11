import pytest

from aleo_shield_swap.client import ShieldSwap
from aleo_shield_swap.errors import (
    PoolNotFoundError,
    PoolNotInitializedError,
    SwapOutputNotFinalizedError,
)
from aleo_shield_swap.types import SwapHandle

from .conftest import StubAleo


def test_get_slot_returns_slotview(stub_aleo):
    dex = ShieldSwap(stub_aleo)
    slot = dex.get_slot("5field")
    assert slot.tick == 4055 and slot.tick_spacing == 60


def test_get_pool_returns_poolstate(stub_aleo):
    pool = ShieldSwap(stub_aleo).get_pool("5field")
    assert pool.token0 == "1field" and pool.fee == 3000
    assert pool.enabled is True


def test_missing_pool_raises(stub_aleo):
    with pytest.raises(PoolNotFoundError):
        ShieldSwap(stub_aleo).get_pool("9field")


def test_get_swap_output_absent_raises(stub_aleo):
    with pytest.raises(SwapOutputNotFinalizedError):
        ShieldSwap(stub_aleo).get_swap_output("9field")


def test_get_swap_output_accepts_handle(stub_aleo):
    handle = SwapHandle(swap_id="9field", blinding_factor=None, blinded_address=None,
                        token_in_id="1field", token_out_id="2field", pool_key="5field",
                        amount_in=1, transaction_id="at1x", program="shield_swap.aleo")
    with pytest.raises(SwapOutputNotFinalizedError):   # resolved to the id
        ShieldSwap(stub_aleo).get_swap_output(handle)
    with pytest.raises(ValueError, match="no swap_id"):
        ShieldSwap(stub_aleo).get_swap_output(
            SwapHandle(swap_id=None, blinding_factor=None, blinded_address=None,
                       token_in_id="1field", token_out_id="2field", pool_key="5field",
                       amount_in=1, transaction_id="at1x", program="shield_swap.aleo"))


def test_uninitialized_pool_distinct_from_missing(stub_aleo):
    from .conftest import POOL_TEXT, StubAleo
    aleo = StubAleo(mappings={"pools": {"5field": POOL_TEXT}, "slots": {}})
    with pytest.raises(PoolNotInitializedError):
        ShieldSwap(aleo).get_slot("5field")
    with pytest.raises(PoolNotFoundError):
        ShieldSwap(aleo).get_slot("6field")


def test_quoted_mapping_values_are_unwrapped():
    aleo = StubAleo(mappings={"initialized_pools": {"5field": '"true"'}})
    assert ShieldSwap(aleo).is_pool_initialized("5field") is True
    assert ShieldSwap(aleo).is_pool_initialized("6field") is False


def test_derive_passthroughs(stub_aleo):
    dex = ShieldSwap(stub_aleo)
    key = dex.derive_pool_key("1234567890123456789field", "9876543210987654321field", 3000)
    assert key == ("50041712585455958488907677199499969829064388375192540321564089"
                   "29642095152812field")


HEADER_TEXT = "{ executed_height: 4242u32, hop_count: 2u8 }"
HOP0_TEXT = ("{ pool: 5field, zero_for_one: true, amount_in: 1000u128, "
             "amount_out: 900u128, fee_paid: 30u128, protocol_fee: 6u128, "
             "sqrt_price_after: { hi: 1u128, lo: 7u128 }, liquidity_after: 555u128, "
             "tick_after: 4056i32 }")
HOP1_TEXT = ("{ pool: 6field, zero_for_one: false, amount_in: 900u128, "
             "amount_out: 850u128, fee_paid: 27u128, protocol_fee: 5u128, "
             "sqrt_price_after: { hi: 0u128, lo: 9u128 }, liquidity_after: 777u128, "
             "tick_after: -12i32 }")


def _execution_stub():
    return StubAleo(mappings={
        "swap_execution_headers": {"77field": HEADER_TEXT},
        # Keys are SwapExecutionKey struct literals, exactly as the chain
        # expects them — the client must format the key, not a bare id.
        "swap_execution_hops": {
            "{ swap_id: 77field, hop_index: 0u8 }": HOP0_TEXT,
            "{ swap_id: 77field, hop_index: 1u8 }": HOP1_TEXT,
        },
        "pool_creators": {"5field": "aleo1creator"},
    })


def test_get_swap_execution_reads_header_then_every_hop():
    """Per-hop fill receipts: header + hop_count hops, lp_fee derived as
    fee_paid - protocol_fee (fee_paid is gross of the protocol share)."""
    ex = ShieldSwap(_execution_stub()).get_swap_execution("77field")
    assert ex is not None
    assert ex.swap_id == "77field"
    assert ex.executed_height == 4242
    assert [h.pool for h in ex.hops] == ["5field", "6field"]
    assert ex.hops[0].amount_in == 1000 and ex.hops[0].amount_out == 900
    assert ex.hops[0].lp_fee == 24 and ex.hops[0].protocol_fee == 6
    assert ex.hops[0].sqrt_price_after == (1 << 128) | 7
    assert ex.hops[1].zero_for_one is False and ex.hops[1].tick_after == -12
    assert ex.hops[1].liquidity_after == 777


def test_get_swap_execution_accepts_handle_and_returns_none_when_unindexed():
    stub = _execution_stub()
    handle = SwapHandle(swap_id="77field", blinding_factor=None, blinded_address=None,
                        token_in_id="1field", token_out_id="2field", pool_key="5field",
                        amount_in=1, transaction_id="at1x", program="shield_swap.aleo")
    assert ShieldSwap(stub).get_swap_execution(handle).executed_height == 4242
    assert ShieldSwap(stub).get_swap_execution("99field") is None   # not finalized


def test_get_swap_execution_missing_hop_is_an_error():
    stub = StubAleo(mappings={
        "swap_execution_headers": {"77field": HEADER_TEXT},
        "swap_execution_hops": {"{ swap_id: 77field, hop_index: 0u8 }": HOP0_TEXT},
    })
    with pytest.raises(ValueError, match="hop 1"):
        ShieldSwap(stub).get_swap_execution("77field")


def test_get_pool_creator():
    dex = ShieldSwap(_execution_stub())
    assert dex.get_pool_creator("5field") == "aleo1creator"
    assert dex.get_pool_creator("6field") is None


def test_get_private_balances(stub_aleo):
    dex = ShieldSwap(stub_aleo)
    out = dex.get_private_balances(["tok.aleo"])
    assert out == {"tok.aleo": 2_000_000_000}


def test_program_handle_is_fetched_once_per_client():
    """Every mapping read used to call ``aleo.programs.get(...)``, which
    downloads and re-parses the 1.4 MB program source — a 47-position owned
    scan on testnet issued ~190 such fetches and stalled for hours.  The
    handle is immutable for a deployment; fetch it once per program."""
    from .conftest import POOL_TEXT
    stub = StubAleo(mappings={
        "pools": {"5field": POOL_TEXT},
        "pool_creators": {"5field": "aleo1c"},
    })
    dex = ShieldSwap(stub)
    dex.get_pool("5field"); dex.get_pool("5field")
    dex.get_pool_creator("5field"); dex.is_pool_initialized("5field")
    assert stub.fetched_programs.count("shield_swap.aleo") == 1


# ── Public balances are chain reads (the API's /balances route is retired) ──

class _Tok:
    def __init__(self, address, symbol, amm, underlying=None, decimals=6):
        self.address, self.symbol, self.decimals = address, symbol, decimals
        self.amm_token_program, self.underlying_program = amm, underlying


class _RegistryApi:
    def __init__(self, tokens):
        self._tokens = tokens

    def get_tokens(self):
        return self._tokens


def test_get_public_balances_reads_each_programs_balances_mapping():
    aleo = StubAleo(mappings={"balances": {"aleo1x": "5u128"}})
    dex = ShieldSwap(aleo)
    out = dex.get_public_balances(["a.aleo", "b.aleo", "a.aleo"], address="aleo1x")
    assert out == {"a.aleo": 5, "b.aleo": 5}
    assert aleo.fetched_programs == ["a.aleo", "b.aleo"]   # deduped, handle cached


def test_get_public_balances_absent_entry_is_zero_and_defaults_to_the_account():
    aleo = StubAleo(mappings={"balances": {}})
    assert ShieldSwap(aleo).get_public_balances(["a.aleo"]) == {"a.aleo": 0}
    aleo.default_account = None
    with pytest.raises(ValueError):
        ShieldSwap(aleo).get_public_balances(["a.aleo"])


def test_get_public_balances_rejects_a_non_arc20_value():
    aleo = StubAleo(mappings={"balances": {"aleo1x": "{ a: 1u8 }"}})
    with pytest.raises(ValueError, match="unsigned integer literal"):
        ShieldSwap(aleo).get_public_balances(["odd.aleo"], address="aleo1x")


def test_get_balances_joins_chain_public_with_record_private():
    # Public side: the AMM token program's mapping; private side: records in
    # the underlying program.  Unheld tokens are omitted.
    aleo = StubAleo(mappings={"balances": {"aleo1x": "7u128"}}, records=[])
    dex = ShieldSwap(aleo)
    dex.api = _RegistryApi([
        _Tok("1field", "USDCx", "wrapped_usdcx.aleo", "usdcx.aleo"),
        _Tok("2field", "ETH", "eth.aleo"),
    ])
    seen = {}
    dex.get_private_balances = lambda programs, account=None: (
        seen.setdefault("programs", list(programs)) and
        {"usdcx.aleo": 3, "eth.aleo": 0})
    out = dex.get_balances(address="aleo1x", account=aleo.default_account)
    # aleo1x is not the bound account, so private is skipped (0) for every token.
    assert out == {"1field": {"symbol": "USDCx", "decimals": 6, "public": 7,
                              "private": 0, "total": 7},
                   "2field": {"symbol": "ETH", "decimals": 6, "public": 7,
                              "private": 0, "total": 7}}
    assert "programs" not in seen
    aleo.default_account.address = "aleo1x"
    out = dex.get_balances()
    assert sorted(seen["programs"]) == ["eth.aleo", "usdcx.aleo"]
    assert out["1field"] == {"symbol": "USDCx", "decimals": 6, "public": 7,
                             "private": 3, "total": 10}
