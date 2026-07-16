"""Liquidity verbs — exact input orders per the TS actions:
create_pool: [token0, token1, fee u16, sqrt_price u128, spacing u32, tick i32]
mint:        [nonce field, record0, record1, recipient, MintPositionRequest, token0, token1]
increase:    [position, record0, record1, a0 u128, a1 u128, a0min u128, a1min u128,
              token0, token1, lo_hint i32, hi_hint i32]
decrease:    [position, liquidity u128, a0min u128, a1min u128]
collect:     [position, a0req u128, a1req u128, token0, token1, recipient]
burn:        [position]"""
import pytest

from aleo_shield_swap.client import ShieldSwap
from aleo_shield_swap.errors import InsufficientRecordsError, InvalidFeeTierError
from aleo_shield_swap.tick_math import get_sqrt_price_at_tick

from .conftest import POOL_TEXT, SIGNER, SLOT_TEXT, StubAleo

POSITION_TEXT = ("{ owner: aleo1me.private, token_id: 42field.private, "
                 "token0_id: 1field.private, token1_id: 2field.private, "
                 "pool: 5field.private, tick_lower: -4080i32.private, "
                 "tick_upper: 4080i32.private, liquidity: 500u128.private, "
                 "_nonce: 3group.public }")

TOKEN0_RECORD = "{ owner: aleo1me.private, amount: 2000000000u128.private, _nonce: 7group.public }"


def _stub(**over):
    mappings = {
        "pools": {"5field": POOL_TEXT},
        "slots": {"5field": SLOT_TEXT},
        "fee_tiers": {"3000u16": "true"},
        "fee_to_tick_spacing": {"3000u16": "60u32"},
        "used_blinded_addresses": {},
    }
    mappings.update(over.pop("mappings", {}))
    records = over.pop("records", [{"record_plaintext": TOKEN0_RECORD},
                                   {"record_plaintext": POSITION_TEXT}])
    return StubAleo(mappings=mappings, records=records)


def test_create_pool_inputs_and_validation():
    stub = _stub()
    dex = ShieldSwap(stub)
    result = dex.create_pool(token0_id="1field", token1_id="2field",
                             fee=3000, initial_tick=0).transact()
    fn, args = stub.last_call
    assert fn == "create_pool"
    assert args == ["1field", "2field", "3000u16",
                    f"{get_sqrt_price_at_tick(0)}u128", "60u32", "0i32"]
    assert result.transaction_id == "at1stubtx"

    with pytest.raises(InvalidFeeTierError):
        dex.create_pool(token0_id="1field", token1_id="2field",
                        fee=123, initial_tick=0)
    assert stub.last_call[0] == "create_pool"     # no new call was prepared


def test_create_pool_rejects_disabled_fee_tier():
    # Registered-then-disabled tier: present in the mapping with value false.
    stub = _stub(mappings={"fee_tiers": {"3000u16": "false"}})
    with pytest.raises(InvalidFeeTierError):
        ShieldSwap(stub).create_pool(token0_id="1field", token1_id="2field",
                                     fee=3000, initial_tick=0)


def test_mint_inputs_rounding_and_request():
    stub = _stub()
    dex = ShieldSwap(stub)
    dex.mint(pool_key="5field", tick_lower=-4055, tick_upper=4055,
             amount0_desired=10**9, amount1_desired=100,
             token0_program="tok0.aleo", token1_program="tok1.aleo",
             nonce="9field")
    fn, args = stub.last_call
    assert fn == "mint"
    assert len(args) == 7
    assert args[0] == "9field"
    assert args[1] == TOKEN0_RECORD and args[2] == TOKEN0_RECORD
    assert args[3] == SIGNER                       # recipient defaults to signer
    # ticks rounded to spacing 60: -4055 -> -4080, 4055 -> 4020
    assert "tick_lower: -4080i32" in args[4]
    assert "tick_upper: 4020i32" in args[4]
    assert "amount0_desired: 1000000000u128" in args[4]
    # slot neighbors: below=3960, above=4080; lower hint for -4080 (< tick) is 3960?
    # pick_insert_hint(-4080 <= 4055) -> next_init_below = 3960
    assert "tick_lower_hint: 3960i32" in args[4]
    assert args[5] == "1field" and args[6] == "2field"


def test_mint_default_nonce_is_generated():
    stub = _stub()
    ShieldSwap(stub).mint(pool_key="5field", tick_lower=-4080, tick_upper=4080,
                          amount0_desired=10**9, amount1_desired=100,
                          token0_program="tok0.aleo", token1_program="tok1.aleo")
    _, args = stub.last_call
    assert args[0].endswith("field") and args[0] != "9field"   # random field nonce


def test_mint_empty_range_raises():
    dex = ShieldSwap(_stub())
    with pytest.raises(ValueError, match="Empty tick range"):
        dex.mint(pool_key="5field", tick_lower=10, tick_upper=50,   # both round to 0
                 amount0_desired=1, amount1_desired=1,
                 token0_program="tok0.aleo", token1_program="tok1.aleo")


def test_increase_liquidity_inputs():
    stub = _stub()
    ShieldSwap(stub).increase_liquidity(
        pool_key="5field", amount0_desired=10**9, amount1_desired=100,
        token0_program="tok0.aleo", token1_program="tok1.aleo")
    fn, args = stub.last_call
    assert fn == "increase_liquidity"
    assert args[0] == POSITION_TEXT                # auto-selected by pool
    assert args[3] == f"{10**9}u128" and args[4] == "100u128"
    assert args[5] == "0u128" and args[6] == "0u128"
    assert args[7] == "1field" and args[8] == "2field"
    assert args[9].endswith("i32") and args[10].endswith("i32")
    assert len(args) == 11


def test_decrease_collect_burn_inputs():
    stub = _stub()
    dex = ShieldSwap(stub)
    dex.decrease_liquidity(pool_key="5field", liquidity_to_remove=500)
    assert stub.last_call == ("decrease_liquidity",
                              [POSITION_TEXT, "500u128", "0u128", "0u128"])

    result = dex.collect(pool_key="5field", amount0_requested=7,
                         amount1_requested=8).transact()
    fn, args = stub.last_call
    assert (fn, args) == ("collect", [POSITION_TEXT, "7u128", "8u128",
                                      "1field", "2field", SIGNER])
    assert result.position_token_id is None        # collect re-issues privately

    dex.burn(pool_key="5field")
    assert stub.last_call == ("burn", [POSITION_TEXT])


def test_position_auto_select_requires_matching_pool():
    stub = _stub(records=[{"record_plaintext": TOKEN0_RECORD}])   # no position
    with pytest.raises(InsufficientRecordsError, match="PositionNFT"):
        ShieldSwap(stub).burn(pool_key="5field")


def test_mint_journals_position(stub_aleo, tmp_path):
    from aleo_shield_swap.journal import Journal
    from aleo_shield_swap.client import ShieldSwap

    dex = ShieldSwap(stub_aleo)
    dex.journal = Journal(tmp_path / "journal.jsonl")
    result = dex.mint(pool_key="5field", tick_lower=-60, tick_upper=60,
                      amount0_desired=10, amount1_desired=10,
                      token0_program="tok.aleo",
                      token1_program="tok.aleo").delegate()
    assert result.position_token_id
    assert dex.journal.open_positions() == [
        {"position_token_id": result.position_token_id, "pool_key": "5field"}]
