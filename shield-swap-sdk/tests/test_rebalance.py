"""Rebalance — close a position and re-mint its successor in one transaction
via shield_swap_rebalance_router.aleo (testnet).

Planner math mirrors veil's planRebalance; input order per the deployed
router: [nft, nonce, each funded side's record (+ sender proof when
wrapped), every wrapped side's receiver proof, RebalanceRequest,
RebalanceAssets, owner_proofs, withdrawal_proofs]."""
import pytest

from aleo_shield_swap import _generated as g
from aleo_shield_swap._core import default_merkle_proofs
from aleo_shield_swap._routing import REBALANCE_ROUTER_ID
from aleo_shield_swap.client import ShieldSwap
from aleo_shield_swap.position_math import amounts_for_liquidity
from aleo_shield_swap.rebalance import RebalancePlan, plan_rebalance
from aleo_shield_swap.tick_math import (MIN_TICK_SENTINEL, MAX_TICK_SENTINEL,
                                        get_sqrt_price_at_tick_x128 as sp)

from .conftest import POOL_TEXT, StubAleo

POOL = "5field"
POSITION_ID = "42field"
OLD_LOWER, OLD_UPPER = -4080, 4080
OLD_LIQUIDITY = 1_000_000
OWED0, OWED1 = 111, 222
SQRT_PRICE = 1 << 128                      # price exactly 1 keeps the math legible
ZERO = "{ hi: 0u128, lo: 0u128 }"

SLOT_TEXT = ("{ tick: 0i32, tick_spacing: 60u32, sqrt_price: { hi: 1u128, lo: 0u128 }, "
             "fee_protocol: 0u8, liquidity: 1000000u128, "
             f"fee_growth_global0_x_128: {ZERO}, fee_growth_global1_x_128: {ZERO}, "
             "max_liquidity_per_tick: 0u128, protocol_fees0: 0u128, protocol_fees1: 0u128, "
             f"next_init_below: {OLD_LOWER}i32, next_init_above: {OLD_UPPER}i32 }}")

POSITION_RECORD = ("{ owner: aleo1me.private, withdrawal: aleo1payout.private, "
                   f"token_id: {POSITION_ID}.private, token0_id: 1field.private, "
                   f"token1_id: 2field.private, pool: {POOL}.private, "
                   f"tick_lower: {OLD_LOWER}i32.private, tick_upper: {OLD_UPPER}i32.private, "
                   "_nonce: 3group.public }")
TOKEN_RECORD = "{ owner: aleo1me.private, amount: 5000000000000000000u128.private, _nonce: 1group.public }"


def _position_text(liquidity=OLD_LIQUIDITY, owed0=OWED0, owed1=OWED1, last0=ZERO, last1=ZERO):
    return (f"{{ token_id: {POSITION_ID}, pool: {POOL}, tick_lower: {OLD_LOWER}i32, "
            f"tick_upper: {OLD_UPPER}i32, liquidity: {liquidity}u128, "
            f"fee_growth_inside0_last_x_128: {last0}, fee_growth_inside1_last_x_128: {last1}, "
            f"tokens_owed0: {owed0}u128, tokens_owed1: {owed1}u128 }}")


def _tick_text(tick, prev, nxt, gross=OLD_LIQUIDITY, out0=ZERO, out1=ZERO):
    return (f"{{ pool: {POOL}, liquidity_net: 0i128, liquidity_gross: {gross}u128, "
            f"tick: {tick}i32, fee_growth_outside0_x_128: {out0}, "
            f"fee_growth_outside1_x_128: {out1}, prev: {prev}i32, next: {nxt}i32 }}")


def _stub(*, slot=SLOT_TEXT, position=None, wrapped=(), gross=OLD_LIQUIDITY, records=None):
    dex = ShieldSwap(StubAleo())                  # for derive_tick_key only
    key = lambda t: dex.derive_tick_key(POOL, t)  # noqa: E731
    # Initialized-tick list: MIN -> OLD_LOWER -> OLD_UPPER -> MAX.
    ticks = {
        key(MIN_TICK_SENTINEL): _tick_text(MIN_TICK_SENTINEL, MIN_TICK_SENTINEL, OLD_LOWER, gross=1),
        key(OLD_LOWER): _tick_text(OLD_LOWER, MIN_TICK_SENTINEL, OLD_UPPER, gross=gross),
        key(OLD_UPPER): _tick_text(OLD_UPPER, OLD_LOWER, MAX_TICK_SENTINEL, gross=gross),
    }
    return StubAleo(mappings={
        "pools": {POOL: POOL_TEXT},
        "slots": {POOL: slot},
        "positions": {POSITION_ID: position or _position_text()},
        "ticks": ticks,
        "from_wrapper_token_id": {t: "9field" for t in wrapped},
    }, records=records if records is not None else
        [{"record_plaintext": TOKEN_RECORD}, {"record_plaintext": POSITION_RECORD}])


def _expected(tick_lower, tick_upper, target):
    principal = amounts_for_liquidity(SQRT_PRICE, sp(OLD_LOWER), sp(OLD_UPPER), OLD_LIQUIDITY)
    recovered = (principal[0] + OWED0, principal[1] + OWED1)
    required = amounts_for_liquidity(SQRT_PRICE, sp(tick_lower), sp(tick_upper), target,
                                     round_up=True)
    return recovered, required


def _state(**over):
    """Pure-planner inputs, as the client would read them."""
    base = dict(
        pool_key=POOL, position_token_id=POSITION_ID,
        tick_lower=OLD_LOWER, tick_upper=OLD_UPPER,
        slot=g.Slot.from_plaintext(SLOT_TEXT),
        position=g.Position.from_plaintext(_position_text()),
        lower_tick=g.Tick.from_plaintext(_tick_text(OLD_LOWER, MIN_TICK_SENTINEL, OLD_UPPER)),
        upper_tick=g.Tick.from_plaintext(_tick_text(OLD_UPPER, OLD_LOWER, MAX_TICK_SENTINEL)),
        wrapped0=False, wrapped1=False,
    )
    base.update(over)
    return base


# ── Pure planner ─────────────────────────────────────────────────────────────

def test_plan_shrink_refunds_and_grow_funds():
    shrink = plan_rebalance(**_state(), liquidity_target=OLD_LIQUIDITY // 2)
    recovered, required = _expected(OLD_LOWER, OLD_UPPER, OLD_LIQUIDITY // 2)
    assert shrink.fees_accrued0 == 0 and shrink.fees_accrued1 == 0
    assert (shrink.recovered0, shrink.recovered1) == recovered
    assert (shrink.required0, shrink.required1) == required
    assert (shrink.funded0, shrink.funded1) == (0, 0)
    assert shrink.refund0 == recovered[0] - required[0]
    assert shrink.refund1 == recovered[1] - required[1]
    assert shrink.old_liquidity == OLD_LIQUIDITY
    assert shrink.function_name == "rebalance_plain_plain_none"

    grow = plan_rebalance(**_state(), liquidity_target=OLD_LIQUIDITY * 2)
    recovered, required = _expected(OLD_LOWER, OLD_UPPER, OLD_LIQUIDITY * 2)
    assert grow.funded0 == required[0] - recovered[0]
    assert grow.funded1 == required[1] - recovered[1]
    assert (grow.refund0, grow.refund1) == (0, 0)
    assert grow.function_name == "rebalance_plain_plain_both"


def test_plan_includes_fees_accrued_past_the_checkpoint():
    # Growth advanced 3 (token0) and 5 (token1) raw units per unit of
    # liquidity: the close settles these, so recovered must include them or
    # the contract's exactness assert reverts.
    slot = g.Slot.from_plaintext(SLOT_TEXT.replace(
        f"fee_growth_global0_x_128: {ZERO}", "fee_growth_global0_x_128: { hi: 3u128, lo: 0u128 }"
    ).replace(f"fee_growth_global1_x_128: {ZERO}", "fee_growth_global1_x_128: { hi: 5u128, lo: 0u128 }"))
    plan = plan_rebalance(**_state(slot=slot), liquidity_target=OLD_LIQUIDITY // 2)
    recovered, _ = _expected(OLD_LOWER, OLD_UPPER, OLD_LIQUIDITY // 2)
    assert plan.fees_accrued0 == 3 * OLD_LIQUIDITY
    assert plan.fees_accrued1 == 5 * OLD_LIQUIDITY
    assert plan.recovered0 == recovered[0] + 3 * OLD_LIQUIDITY
    assert plan.recovered1 == recovered[1] + 5 * OLD_LIQUIDITY


def test_plan_budget_mode_solves_the_largest_liquidity_the_budget_supports():
    zero = plan_rebalance(**_state(), max_funding0=0, max_funding1=0)
    assert zero.liquidity_target > 0
    assert (zero.funded0, zero.funded1) == (0, 0)
    assert zero.required0 <= zero.recovered0 and zero.required1 <= zero.recovered1

    funded = plan_rebalance(**_state(), max_funding0=10_000, max_funding1=10_000)
    assert funded.liquidity_target > zero.liquidity_target
    assert funded.funded0 <= 10_000 and funded.funded1 <= 10_000
    assert funded.function_name == "rebalance_plain_plain_both"


def test_plan_rejects_a_budget_that_supports_no_liquidity():
    tiny = g.Position.from_plaintext(_position_text(liquidity=1, owed0=0, owed1=0))
    with pytest.raises(ValueError, match="supports no liquidity"):
        plan_rebalance(**_state(position=tiny), max_funding0=0, max_funding1=0)


def test_plan_rejects_ambiguous_or_missing_sizing():
    with pytest.raises(ValueError, match="exactly one sizing mode"):
        plan_rebalance(**_state(), liquidity_target=1, max_funding0=0, max_funding1=0)
    with pytest.raises(ValueError, match="exactly one sizing mode"):
        plan_rebalance(**_state())
    with pytest.raises(ValueError, match="both max_funding0 and max_funding1"):
        plan_rebalance(**_state(), max_funding0=0)


def test_plan_aligns_ticks_and_rejects_an_empty_range():
    plan = plan_rebalance(**_state(tick_lower=-4079, tick_upper=4079), liquidity_target=5)
    assert (plan.tick_lower, plan.tick_upper) == (-4080, 4020)     # floors to spacing 60
    with pytest.raises(ValueError, match="Empty tick range"):
        plan_rebalance(**_state(tick_lower=10, tick_upper=50), liquidity_target=5)


def test_plan_exact_fill_invariants_hold():
    """The contract asserts recovered + funded == required + refund per side
    and funded * refund == 0; every plan must satisfy both by construction."""
    for kwargs in (dict(liquidity_target=OLD_LIQUIDITY // 3),
                   dict(liquidity_target=OLD_LIQUIDITY * 3),
                   dict(max_funding0=0, max_funding1=0),
                   dict(max_funding0=123_456, max_funding1=0)):
        p = plan_rebalance(**_state(), **kwargs)
        assert p.recovered0 + p.funded0 == p.required0 + p.refund0
        assert p.recovered1 + p.funded1 == p.required1 + p.refund1
        assert p.funded0 == 0 or p.refund0 == 0
        assert p.funded1 == 0 or p.refund1 == 0


# ── Client: plan from chain reads, execute through the router ────────────────

def test_client_plan_reads_slot_position_and_boundary_ticks():
    dex = ShieldSwap(_stub())
    plan = dex.plan_rebalance(pool_key=POOL, position_token_id=POSITION_ID,
                              tick_lower=OLD_LOWER, tick_upper=OLD_UPPER,
                              liquidity_target=OLD_LIQUIDITY // 2)
    expected = plan_rebalance(**_state(), liquidity_target=OLD_LIQUIDITY // 2)
    assert plan == expected


def test_client_plan_missing_position_raises():
    stub = _stub()
    stub.programs._mappings["positions"] = {}
    with pytest.raises(ValueError, match="Position does not exist"):
        ShieldSwap(stub).plan_rebalance(pool_key=POOL, position_token_id=POSITION_ID,
                                        tick_lower=OLD_LOWER, tick_upper=OLD_UPPER,
                                        liquidity_target=1)


def test_rebalance_plain_pair_no_funding_exact_slots():
    stub = _stub()
    dex = ShieldSwap(stub)
    result = dex.rebalance_position(pool_key=POOL, position_token_id=POSITION_ID,
                                    tick_lower=OLD_LOWER, tick_upper=OLD_UPPER,
                                    liquidity_target=OLD_LIQUIDITY // 2,
                                    nonce="9field").transact()
    fn, args = stub.last_call
    assert stub.last_program == REBALANCE_ROUTER_ID
    assert fn == "rebalance_plain_plain_none"
    # [nft, nonce, request, assets, owner_proofs, withdrawal_proofs]
    assert len(args) == 6
    assert args[0] == POSITION_RECORD and args[1] == "9field"
    plan = result.plan
    request = g.CoreRebalanceRequest(
        old_liquidity=OLD_LIQUIDITY, recovered0=plan.recovered0, recovered1=plan.recovered1,
        funded0=0, funded1=0, refund0=plan.refund0, refund1=plan.refund1,
        liquidity_target=OLD_LIQUIDITY // 2,
        mint=g.MintPositionRequest(
            pool=POOL, tick_lower=OLD_LOWER, tick_upper=OLD_UPPER,
            amount0_desired=plan.required0, amount1_desired=plan.required1,
            amount0_min=plan.required0, amount1_min=plan.required1,
            # The close unlinks both old boundary ticks (their whole gross
            # liquidity is this position), so hints step back to survivors:
            # MIN sentinel for the lower bound, then the freshly inserted lower
            # tick for the upper bound.
            tick_lower_hint=MIN_TICK_SENTINEL, tick_upper_hint=OLD_LOWER),
        deadline=1020,                      # height 1000 + the 20-block default
    ).to_plaintext()
    assert args[2] == request
    assert args[3] == g.RebalanceAssets(
        token0=g.RebalanceAsset(token_id="1field", underlying_id="1field"),
        token1=g.RebalanceAsset(token_id="2field", underlying_id="2field"),
    ).to_plaintext()
    assert args[4] == args[5] == default_merkle_proofs()
    assert result.transaction_id == "at1stubtx"
    assert result.position_token_id == "77field"        # first public field output
    assert result.plan.function_name == "rebalance_plain_plain_none"


def test_rebalance_surviving_ticks_are_kept_as_hints():
    # Another position shares the boundary ticks (gross > old liquidity), so
    # the close leaves them linked and the hints are the plain predecessors.
    stub = _stub(gross=OLD_LIQUIDITY + 5)
    ShieldSwap(stub).rebalance_position(pool_key=POOL, position_token_id=POSITION_ID,
                                        tick_lower=OLD_LOWER, tick_upper=OLD_UPPER,
                                        liquidity_target=OLD_LIQUIDITY // 2).transact()
    _, args = stub.last_call
    request = g.CoreRebalanceRequest.from_plaintext(args[2])
    # An initialized tick is its own predecessor (validation is skipped).
    assert (request.mint.tick_lower_hint, request.mint.tick_upper_hint) == (OLD_LOWER, OLD_UPPER)


def test_rebalance_funded_wrapped_side_interleaves_record_and_proofs():
    stub = _stub(wrapped={"1field"})
    dex = ShieldSwap(stub)
    dex._token_program = lambda tid: {"1field": "under_zero.aleo", "2field": "tok1.aleo"}[tid]
    dex._amm_token_program = lambda tid: "wrap_zero.aleo" if tid == "1field" else None
    result = dex.rebalance_position(pool_key=POOL, position_token_id=POSITION_ID,
                                    tick_lower=OLD_LOWER, tick_upper=OLD_UPPER,
                                    liquidity_target=OLD_LIQUIDITY * 2,
                                    token1_record=TOKEN_RECORD, nonce="9field").transact()
    fn, args = stub.last_call
    assert fn == "rebalance_wrapped_plain_both"
    # [nft, nonce, rec0, sender_proof0, rec1, receiver_proof0, request, assets, owner, withdrawal]
    assert len(args) == 10
    assert args[2] == TOKEN_RECORD and args[3] == default_merkle_proofs()
    assert args[4] == TOKEN_RECORD and args[5] == default_merkle_proofs()
    # The settlement asset comes from the chain's from_wrapper_token_id entry.
    assert g.RebalanceAssets.from_plaintext(args[7]).token0.underlying_id == "9field"
    assert result.plan.funded0 > 0 and result.plan.funded1 > 0
    # The router, the wrapper, and both settlement token programs are registered.
    assert {"shield_swap_rebalance_router.aleo", "wrap_zero.aleo",
            "under_zero.aleo", "tok1.aleo"} <= set(stub.registered_programs)


def test_rebalance_wrapped_unfunded_side_still_takes_a_receiver_proof():
    stub = _stub(wrapped={"2field"})
    dex = ShieldSwap(stub)
    dex._token_program = lambda tid: "tok.aleo"
    dex._amm_token_program = lambda tid: "wrap_one.aleo" if tid == "2field" else None
    dex.rebalance_position(pool_key=POOL, position_token_id=POSITION_ID,
                           tick_lower=OLD_LOWER, tick_upper=OLD_UPPER,
                           liquidity_target=OLD_LIQUIDITY // 2).transact()
    fn, args = stub.last_call
    # [nft, nonce, receiver_proof1, request, assets, owner, withdrawal]
    assert (fn, len(args)) == ("rebalance_plain_wrapped_none", 7)
    assert args[2] == default_merkle_proofs()


def test_rebalance_accepts_a_prebuilt_plan_verbatim():
    stub = _stub()
    dex = ShieldSwap(stub)
    plan = plan_rebalance(**_state(), liquidity_target=OLD_LIQUIDITY // 2)
    hand_built = RebalancePlan(**{**plan.__dict__, "recovered0": plan.recovered0 + 7,
                                  "refund0": plan.refund0 + 7})
    result = dex.rebalance_position(plan=hand_built, nonce="9field").transact()
    _, args = stub.last_call
    request = g.CoreRebalanceRequest.from_plaintext(args[2])
    assert request.recovered0 == plan.recovered0 + 7      # submitted as given
    assert result.plan == hand_built


def test_rebalance_requires_sizing_without_a_plan():
    with pytest.raises(ValueError, match="exactly one sizing mode"):
        ShieldSwap(_stub()).rebalance_position(pool_key=POOL, position_token_id=POSITION_ID,
                                               tick_lower=OLD_LOWER, tick_upper=OLD_UPPER)


def test_rebalance_result_journals_the_successor_position(tmp_path):
    from aleo_shield_swap.journal import Journal
    stub = _stub()
    dex = ShieldSwap(stub)
    dex.journal = Journal(tmp_path / "j.jsonl")
    dex.rebalance_position(pool_key=POOL, position_token_id=POSITION_ID,
                           tick_lower=OLD_LOWER, tick_upper=OLD_UPPER,
                           liquidity_target=OLD_LIQUIDITY // 2).transact()
    assert any(e.get("type") == "position" and e.get("position_token_id") == "77field"
               for e in dex.journal.events())
