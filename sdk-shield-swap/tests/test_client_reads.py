import pytest

from aleo_shield_swap.client import ShieldSwap
from aleo_shield_swap.errors import PoolNotFoundError, SwapOutputNotFinalizedError

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
