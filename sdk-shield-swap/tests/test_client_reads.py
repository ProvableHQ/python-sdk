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
    assert pool.scale0 == 10**9


def test_missing_pool_raises(stub_aleo):
    with pytest.raises(PoolNotFoundError):
        ShieldSwap(stub_aleo).get_pool("9field")


def test_get_swap_output_absent_raises(stub_aleo):
    with pytest.raises(SwapOutputNotFinalizedError):
        ShieldSwap(stub_aleo).get_swap_output("9field")


def test_get_swap_output_accepts_handle(stub_aleo):
    handle = SwapHandle(swap_id="9field", blinding_factor=None, blinded_address=None,
                        token_in_id="1field", token_out_id="2field", pool_key="5field",
                        amount_in=1, transaction_id="at1x", program="shield_swap_v3.aleo")
    with pytest.raises(SwapOutputNotFinalizedError):   # resolved to the id
        ShieldSwap(stub_aleo).get_swap_output(handle)
    with pytest.raises(ValueError, match="no swap_id"):
        ShieldSwap(stub_aleo).get_swap_output(
            SwapHandle(swap_id=None, blinding_factor=None, blinded_address=None,
                       token_in_id="1field", token_out_id="2field", pool_key="5field",
                       amount_in=1, transaction_id="at1x", program="shield_swap_v3.aleo"))


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


def test_get_private_balances(stub_aleo):
    dex = ShieldSwap(stub_aleo)
    out = dex.get_private_balances(["tok.aleo"])
    assert out == {"tok.aleo": 2_000_000_000}
