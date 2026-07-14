from decimal import Decimal

from aleo_shield_swap._generated import Slot
from aleo_shield_swap.tick_math import Q64
from aleo_shield_swap.types import SlotView, SwapHandle


def _slot(**over):
    base = dict(tick=4055, tick_spacing=60, sqrt_price=Q64, fee_protocol=0,
                liquidity=0, fee_growth_global0_x_64=0, fee_growth_global1_x_64=0,
                fee_residual0_x_64=0, fee_residual1_x_64=0, max_liquidity_per_tick=0,
                protocol_fees0=0, protocol_fees1=0, next_init_below=0, next_init_above=0)
    base.update(over)
    return Slot(**base)


def test_swap_handle_json_roundtrip():
    h = SwapHandle(swap_id="1field", blinding_factor="2field", blinded_address="aleo1x",
                   token_in_id="3field", token_out_id="4field", pool_key="5field",
                   amount_in=10**18, transaction_id="at1abc", program="shield_swap_v3.aleo")
    assert SwapHandle.from_json(h.to_json()) == h


def test_slot_price_at_q64_is_one():
    v = SlotView(_slot())
    assert v.price(9, 9) == Decimal(1)
    assert v.price(6, 6) == Decimal(1)       # equal decimals cancel
    assert v.price(18, 6) == Decimal(1000)   # norm-capped: min(18,9)-min(6,9) = 3
    assert v.tick == 4055                    # attribute delegation
    assert v.raw is v._slot


def test_tick_range_alignment():
    v = SlotView(_slot(tick=4055, tick_spacing=60))
    lo, hi = v.tick_range(10)
    assert lo % 60 == 0 and hi % 60 == 0
    assert lo < 4055 < hi
