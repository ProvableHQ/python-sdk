"""swap() against the stubbed facade — asserts the exact positional input
list (order from the TS reference src/actions/swap/swap.ts):
[record, blinding_factor, blinded_address, pool_key, zero_for_one,
 amount_in u128, amount_out_min u128, sqrt_price_limit u128, nonce u64,
 deadline u32, token0, token1]"""
import pytest

from aleo_shield_swap.client import ShieldSwap
from aleo_shield_swap.errors import InsufficientRecordsError
from aleo_shield_swap.tick_math import MIN_SQRT_PRICE
from aleo_shield_swap.types import SwapHandle

from .conftest import (
    BLINDED_ADDRESS_0,
    BLINDING_FACTOR_0,
    POOL_TEXT,
    RECORD_TEXT,
    SLOT_TEXT,
    StubAleo,
)


def _swap_call(stub_aleo, **over):
    dex = ShieldSwap(stub_aleo)
    kwargs = dict(pool_key="5field", token_in_id="1field", amount_in=10**9,
                  slippage_bps=50, nonce=123, token_in_program="tok.aleo",
                  expected_out=1_000_000)
    kwargs.update(over)
    return dex.swap(**kwargs)


def test_swap_builds_exact_inputs(stub_aleo):
    call = _swap_call(stub_aleo)
    fn, args = stub_aleo.last_call
    assert fn == "swap"
    assert args[0] == RECORD_TEXT                     # auto-selected record
    assert args[1] == BLINDING_FACTOR_0               # counter-0 identity
    assert args[2] == BLINDED_ADDRESS_0
    assert args[3] == "5field"
    assert args[4] is True                            # zero_for_one
    assert args[5] == f"{10**9}u128"
    assert args[6] == f"{1_000_000 * 9950 // 10000}u128"   # slippage applied
    assert args[7] == f"{MIN_SQRT_PRICE}u128"         # directional default
    assert args[8] == "123u64"
    assert args[9] == "1100u32"                       # height 1000 + 100
    assert args[10] == "1field" and args[11] == "2field"
    assert len(args) == 12
    # Dynamic dispatch: the DEX program and the token wrapper program must be
    # registered with the process before authorization.
    assert "shield_swap_v3.aleo" in stub_aleo.registered_programs
    assert "tok.aleo" in stub_aleo.registered_programs


def test_swap_transact_returns_complete_handle(stub_aleo):
    handle = _swap_call(stub_aleo).transact()
    assert isinstance(handle, SwapHandle)
    assert handle.swap_id == "77field"                # first field output
    assert handle.transaction_id == "at1stubtx"
    assert handle.blinding_factor == BLINDING_FACTOR_0
    assert handle.blinded_address == BLINDED_ADDRESS_0
    assert handle.token_out_id == "2field"
    assert handle.amount_in == 10**9
    assert stub_aleo.submitted                        # broadcast happened
    assert SwapHandle.from_json(handle.to_json()) == handle


def test_swap_delegate_recovers_swap_id_from_confirmed_tx(stub_aleo):
    handle = _swap_call(stub_aleo).delegate()
    assert handle.transaction_id == "at1delegated"
    assert handle.swap_id == "77field"                # from decode_transition
    assert stub_aleo.waited == ["at1delegated"]


def test_swap_simulate_passthrough(stub_aleo):
    assert _swap_call(stub_aleo).simulate() == "simulated"


def test_swap_explicit_record_skips_selection(stub_aleo):
    explicit = "{ owner: aleo1me.private, amount: 5000000000u128.private, _nonce: 9group.public }"
    _swap_call(stub_aleo, token_record=explicit)
    _, args = stub_aleo.last_call
    assert args[0] == explicit


def test_swap_no_covering_record_raises():
    stub = StubAleo(
        mappings={
            "pools": {"5field": POOL_TEXT},
            "slots": {"5field": SLOT_TEXT},
            "used_blinded_addresses": {},
        },
        records=[],
    )
    with pytest.raises(InsufficientRecordsError):
        _swap_call(stub)
