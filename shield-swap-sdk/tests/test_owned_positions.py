"""get_owned_positions / get_owned_position — the record↔mapping join.

A position's identity lives in the private PositionNFT record and its amounts
live in the public mappings; these cover the join, the finalize lag, the pool
filter, and that the derived amounts agree with the view math.
"""
from __future__ import annotations

from aleo_shield_swap.client import ShieldSwap
from aleo_shield_swap.position_math import (
    amounts_for_liquidity,
    fee_owed,
    u256_of,
)
from aleo_shield_swap.tick_math import get_sqrt_price_at_tick_x128

from .conftest import POOL_TEXT, SLOT_TEXT, StubAleo

POSITION_RECORD = (
    "{ owner: aleo1me.private, withdrawal: aleo1payout.private, "
    "token_id: 42field.private, "
    "token0_id: 1field.private, token1_id: 2field.private, "
    "pool: 5field.private, tick_lower: -4080i32.private, "
    "tick_upper: 4080i32.private, liquidity: 500u128.private, "
    "_nonce: 3group.public }"
)
# A record from the same program that is NOT a PositionNFT (no tick_lower).
TOKEN_RECORD = ("{ owner: aleo1me.private, amount: 900u128.private, "
                "_nonce: 7group.public }")

POSITION_ENTRY = (
    "{ token_id: 42field, pool: 5field, tick_lower: -4080i32, "
    "tick_upper: 4080i32, liquidity: 500u128, "
    "fee_growth_inside0_last_x_128: { hi: 0u128, lo: 0u128 }, "
    "fee_growth_inside1_last_x_128: { hi: 0u128, lo: 0u128 }, "
    "tokens_owed0: 11u128, tokens_owed1: 22u128 }"
)


def _tick(pool: str, tick: int, fg0: int = 0, fg1: int = 0) -> str:
    return (f"{{ pool: {pool}, liquidity_net: 0i128, liquidity_gross: 500u128, "
            f"tick: {tick}i32, "
            f"fee_growth_outside0_x_128: {{ hi: 0u128, lo: {fg0}u128 }}, "
            f"fee_growth_outside1_x_128: {{ hi: 0u128, lo: {fg1}u128 }}, "
            "prev: 0i32, next: 0i32 }")


def _stub(*, positions=None, ticks=True, records=None):
    dex_keys = {
        "pools": {"5field": POOL_TEXT},
        "slots": {"5field": SLOT_TEXT},
        "positions": positions if positions is not None else {"42field": POSITION_ENTRY},
    }
    stub = StubAleo(mappings=dex_keys,
                    records=records if records is not None
                    else [{"record_plaintext": POSITION_RECORD}])
    if ticks:
        dex = ShieldSwap(stub)
        stub.programs._mappings["ticks"] = {
            dex.derive_tick_key("5field", -4080): _tick("5field", -4080),
            dex.derive_tick_key("5field", 4080): _tick("5field", 4080),
        }
    return stub


def test_joins_record_identity_with_chain_state():
    owned = ShieldSwap(_stub()).get_owned_positions()
    assert len(owned) == 1
    p = owned[0]
    # record side
    assert p.position_token_id == "42field"
    assert (p.pool_key, p.tick_lower, p.tick_upper) == ("5field", -4080, 4080)
    assert (p.token0_id, p.token1_id) == ("1field", "2field")
    assert p.withdrawal == "aleo1payout"
    assert p.record == POSITION_RECORD          # spendable, for position_record=
    # chain side
    assert p.state is not None
    assert p.state.liquidity == 500
    assert (p.state.tokens_owed0, p.state.tokens_owed1) == (11, 22)


def test_amounts_match_the_view_math():
    dex = ShieldSwap(_stub())
    p = dex.get_owned_positions()[0]
    slot = dex.get_slot("5field").raw
    expected = amounts_for_liquidity(
        u256_of(slot.sqrt_price),
        get_sqrt_price_at_tick_x128(-4080),
        get_sqrt_price_at_tick_x128(4080),
        500,
    )
    assert (p.state.amount0, p.state.amount1) == expected


def test_collectible_is_owed_plus_accrued():
    # outside counters at 0 and last-inside at 0, so accrued == inside growth
    dex = ShieldSwap(_stub())
    p = dex.get_owned_positions()[0]
    assert p.state.collectible0 >= p.state.tokens_owed0
    assert p.state.collectible1 >= p.state.tokens_owed1


def test_state_is_none_while_the_mint_finalizes():
    # the record exists but positions[token_id] is not written yet
    owned = ShieldSwap(_stub(positions={})).get_owned_positions()
    assert len(owned) == 1
    assert owned[0].state is None
    assert owned[0].position_token_id == "42field"   # identity still usable


def test_state_is_none_when_boundary_ticks_are_uninitialized():
    owned = ShieldSwap(_stub(ticks=False)).get_owned_positions()
    assert owned[0].state is None


def test_non_position_records_are_skipped():
    stub = _stub(records=[{"record_plaintext": TOKEN_RECORD},
                          {"record_plaintext": POSITION_RECORD}])
    owned = ShieldSwap(stub).get_owned_positions()
    assert [p.position_token_id for p in owned] == ["42field"]


def test_pool_key_filters():
    dex = ShieldSwap(_stub())
    assert len(dex.get_owned_positions(pool_key="5field")) == 1
    assert dex.get_owned_positions(pool_key="9field") == []


def test_get_owned_position_by_id():
    dex = ShieldSwap(_stub())
    assert dex.get_owned_position("42field").position_token_id == "42field"
    assert dex.get_owned_position("999field") is None


def test_no_records_is_empty_not_error():
    assert ShieldSwap(_stub(records=[])).get_owned_positions() == []


def test_fee_owed_contribution_is_reproducible():
    """collectible - tokens_owed must equal fee_owed over the inside growth."""
    dex = ShieldSwap(_stub())
    p = dex.get_owned_positions()[0]
    accrued0 = p.state.collectible0 - p.state.tokens_owed0
    # with all outside/last counters zero, inside growth == global growth
    slot = dex.get_slot("5field").raw
    assert accrued0 == fee_owed(u256_of(slot.fee_growth_global0_x_128), 0, 500)
