from decimal import Decimal

from aleo_shield_swap._generated import Slot, U256__8JquwLopp8 as U256
from aleo_shield_swap.types import SlotView, SwapHandle


def _u(value: int) -> U256:
    return U256(hi=value >> 128, lo=value & ((1 << 128) - 1))


def _slot(**over):
    base = dict(tick=4055, tick_spacing=60, sqrt_price=_u(1 << 128), fee_protocol=0,
                liquidity=0, fee_growth_global0_x_128=_u(0), fee_growth_global1_x_128=_u(0),
                max_liquidity_per_tick=0, protocol_fees0=0, protocol_fees1=0,
                next_init_below=0, next_init_above=0)
    base.update(over)
    return Slot(**base)


def test_swap_handle_json_roundtrip():
    h = SwapHandle(swap_id="1field", blinding_factor="2field", blinded_address="aleo1x",
                   token_in_id="3field", token_out_id="4field", pool_key="5field",
                   amount_in=10**18, transaction_id="at1abc", program="shield_swap.aleo")
    assert SwapHandle.from_json(h.to_json()) == h


def test_slot_price_at_q128_is_one():
    v = SlotView(_slot())
    assert v.price(9, 9) == Decimal(1)
    assert v.price(6, 6) == Decimal(1)       # equal decimals cancel
    assert v.price(18, 6) == Decimal(10) ** 12   # raw units: 10^(18-6)
    assert v.tick == 4055                    # attribute delegation
    assert v.raw is v._slot


def test_slot_price_x128_sqrt_two():
    # sqrt_price = 2.0 in Q128.128 → price 4.0 (token1 per token0, raw units)
    v = SlotView(_slot(sqrt_price=_u(2 << 128)))
    assert v.price(9, 6) == Decimal(4) * Decimal(10) ** 3


def test_tick_range_alignment():
    v = SlotView(_slot(tick=4055, tick_spacing=60))
    lo, hi = v.tick_range(10)
    assert lo % 60 == 0 and hi % 60 == 0
    assert lo < 4055 < hi
