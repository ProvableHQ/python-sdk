"""Write tier: the full liquidity lifecycle on the REAL testnet — veil's
``liveLiquidity.e2e.test.ts`` plus ``ownedPositions.e2e.test.ts``, and the
Python-only live rebalance (the router is deployed on testnet).

mint → owned-position views → increase → rebalance (zero budget) →
decrease all → collect → burn.  Spends real testnet balances (proportional
budgets: a thousandth of each held side), DPS pays proving fees.  A run that
aborts leaves the position on chain.

Run: ``python -m pytest tests/integration/test_liquidity_lifecycle_live.py -m "live and slow"``
"""
from __future__ import annotations

import time

import pytest

from aleo_shield_swap.position_math import amounts_for_liquidity, liquidity_for_amounts
from aleo_shield_swap.tick_math import (MIN_TICK_SENTINEL, get_sqrt_price_at_tick_x128,
                                        round_tick_to_spacing, u256_to_int)

from .conftest import poll_until, with_retry, write_tier

pytestmark = [pytest.mark.live, pytest.mark.slow]


class _State:
    """Order-dependent lifecycle state; the first failure aborts the rest so
    no fee is burned on a doomed follow-up write."""

    aborted = None
    pool = None
    slot = None
    tick_lower = tick_upper = None
    budget0 = budget1 = None
    amount0 = amount1 = None
    liquidity = None
    position_token_id = None
    record = None


@pytest.fixture(scope="module")
def st():
    return _State()


@pytest.fixture(autouse=True)
def _abort_guard(request, st):
    if st.aborted:
        pytest.fail(f'aborted: "{st.aborted}" failed, and the rest of the lifecycle depends on it')
    yield
    report = getattr(request.node, "rep_call", None)      # set by conftest's makereport hook
    if report is not None and report.failed:
        st.aborted = request.node.name


def _wait_for_position(dex, token_id, predicate, what, seconds=60):
    """Mapping writes propagate asynchronously — poll the ``positions`` entry."""
    deadline = time.time() + seconds
    while True:
        pos = dex._position_state(token_id)
        if predicate(pos):
            return pos
        if time.time() > deadline:
            pytest.fail(f"position did not {what} within {seconds}s (last: {pos})")
        time.sleep(3)


def _wait_for_fresh_record(dex, token_id, stale=None, seconds=180):
    """The spent record can still be served by the scanner for a while; a
    write built on it carries a consumed serial and is dropped at
    verification — so wait for a record that differs from *stale*."""
    deadline = time.time() + seconds
    last = None
    while time.time() < deadline:
        try:
            owned = dex.get_owned_position(token_id)
        except Exception as exc:               # scanner 401/5xx — retry inside the window
            last = exc
            owned = None
        if owned is not None and owned.record != stale:
            return owned
        time.sleep(5)
    pytest.fail(f"no fresh PositionNFT record for {token_id} within {seconds}s ({last})")


@write_tier
def test_finds_a_pool_funded_on_both_sides_and_derives_a_range(account_dex, st):
    dex = account_dex
    held = {tid: v["private"] for tid, v in dex.get_balances().items()}
    candidates = []
    for entry in dex.api.get_pools():
        if held.get(entry.token0, 0) > 0 and held.get(entry.token1, 0) > 0:
            candidates.append((dex.get_slot(entry.key).liquidity, entry))
    assert candidates, "no live pool is funded on both sides for this account"
    # The deepest pool: a thin one moves price sharply between plan and finalize.
    _, st.pool = max(candidates, key=lambda c: c[0])
    st.slot = dex.get_slot(st.pool.key)
    spacing = st.slot.tick_spacing
    onchain = dex.get_pool(st.pool.key)
    assert dex._mapping_value("fee_to_tick_spacing", f"{onchain.fee}u16") == f"{spacing}u32"

    center = round_tick_to_spacing(st.slot.tick, spacing)
    st.tick_lower, st.tick_upper = center - spacing * 10, center + spacing * 10
    assert st.tick_lower < st.slot.tick < st.tick_upper

    st.budget0, st.budget1 = held[st.pool.token0] // 1000, held[st.pool.token1] // 1000
    price = u256_to_int(st.slot.sqrt_price)
    lo, hi = get_sqrt_price_at_tick_x128(st.tick_lower), get_sqrt_price_at_tick_x128(st.tick_upper)
    st.liquidity = liquidity_for_amounts(price, lo, hi, st.budget0, st.budget1)
    assert st.liquidity > 0, "budget is dust for this range — fund the account further"
    st.amount0, st.amount1 = amounts_for_liquidity(price, lo, hi, st.liquidity, round_up=True)
    assert st.amount0 + st.amount1 > 0
    assert st.amount0 <= st.budget0 and st.amount1 <= st.budget1


@write_tier
def test_insert_hints_are_true_predecessors(account_dex, st):
    dex = account_dex
    for target in (st.tick_lower, st.tick_upper):
        hint = dex.find_tick_predecessor(st.pool.key, target)
        assert hint <= target
        if hint == target:
            continue                            # already initialized — validation skipped
        if hint != MIN_TICK_SENTINEL:
            tick = dex._tick_info(st.pool.key, hint)
            assert tick is not None, f"hint {hint} is not an initialized tick"
            assert tick.next >= target
        else:
            head = dex._tick_info(st.pool.key, MIN_TICK_SENTINEL)
            assert head is None or head.next >= target


@write_tier
def test_mint_position(account_dex, st):
    dex = account_dex
    result = with_retry(lambda: dex.mint(
        pool_key=st.pool.key, tick_lower=st.tick_lower, tick_upper=st.tick_upper,
        amount0_desired=st.amount0, amount1_desired=st.amount1).delegate())
    assert result.position_token_id and result.transaction_id
    st.position_token_id = result.position_token_id
    pos = _wait_for_position(dex, st.position_token_id, lambda p: p is not None,
                             "appear in the positions mapping")
    assert pos.liquidity > 0
    assert (pos.tick_lower, pos.tick_upper) == (st.tick_lower, st.tick_upper)
    drift = abs(pos.liquidity - st.liquidity) / st.liquidity
    assert drift < 0.01, f"predicted {st.liquidity}, chain minted {pos.liquidity}"
    st.liquidity = pos.liquidity


@write_tier
def test_owned_position_views_carry_the_new_position(account_dex, st):
    dex = account_dex
    owned = _wait_for_fresh_record(dex, st.position_token_id)
    st.record = owned.record
    assert owned.pool_key == st.pool.key
    assert (owned.tick_lower, owned.tick_upper) == (st.tick_lower, st.tick_upper)
    assert owned.withdrawal.startswith("aleo1")
    assert owned.state is not None and owned.state.liquidity == st.liquidity
    # Every joined view agrees with the public mapping (ownedPositions.e2e).
    for p in dex.get_owned_positions():
        mapped = dex._position_state(p.position_token_id)
        if p.state is None or mapped is None:
            continue
        assert p.state.liquidity == mapped.liquidity
        assert (p.state.tokens_owed0, p.state.tokens_owed1) == (mapped.tokens_owed0, mapped.tokens_owed1)
        assert p.state.collectible0 >= p.state.tokens_owed0
        assert p.state.collectible1 >= p.state.tokens_owed1
        if p.state.liquidity == 0:
            assert (p.state.amount0, p.state.amount1) == (0, 0)
    assert dex.get_owned_position("1field") is None


@write_tier
def test_increase_liquidity(account_dex, st):
    dex = account_dex
    with_retry(lambda: dex.increase_liquidity(
        pool_key=st.pool.key, amount0_desired=st.amount0, amount1_desired=st.amount1,
        position_record=st.record).delegate())
    pos = _wait_for_position(dex, st.position_token_id,
                             lambda p: p is not None and p.liquidity > st.liquidity,
                             "show the added liquidity")
    st.liquidity = pos.liquidity
    st.record = _wait_for_fresh_record(dex, st.position_token_id, stale=st.record).record


@write_tier
def test_rebalance_shifts_the_range_on_recovered_funds(account_dex, st):
    """Live rebalance through shield_swap_rebalance_router.aleo: one spacing
    up, zero budget.  The contract asserts every planned amount, so success
    here is the planner agreeing with the chain at the execution price."""
    dex = account_dex
    spacing = st.slot.tick_spacing
    old_id = st.position_token_id
    plan = dex.plan_rebalance(pool_key=st.pool.key, position_token_id=old_id,
                              tick_lower=st.tick_lower + spacing,
                              tick_upper=st.tick_upper + spacing,
                              max_funding0=0, max_funding1=0)
    assert plan.old_liquidity == st.liquidity
    assert plan.funded0 == plan.funded1 == 0
    assert plan.function_name.startswith("rebalance_") and plan.function_name.endswith("_none")
    result = with_retry(lambda: dex.rebalance_position(
        pool_key=st.pool.key, position_token_id=old_id, position_record=st.record,
        tick_lower=st.tick_lower + spacing, tick_upper=st.tick_upper + spacing,
        max_funding0=0, max_funding1=0).delegate(), attempts=2, delay=10)
    assert result.position_token_id and result.position_token_id != old_id
    st.position_token_id = result.position_token_id
    st.tick_lower, st.tick_upper = result.plan.tick_lower, result.plan.tick_upper
    pos = _wait_for_position(dex, st.position_token_id, lambda p: p is not None,
                             "appear after the rebalance")
    assert pos.liquidity == result.plan.liquidity_target
    assert (pos.tick_lower, pos.tick_upper) == (st.tick_lower, st.tick_upper)
    assert poll_until(lambda: dex._position_state(old_id) is None, 20, 3)
    st.liquidity = pos.liquidity
    st.record = _wait_for_fresh_record(dex, st.position_token_id).record


@write_tier
def test_decrease_the_whole_position(account_dex, st):
    dex = account_dex
    with_retry(lambda: dex.decrease_liquidity(
        pool_key=st.pool.key, liquidity_to_remove=st.liquidity,
        position_record=st.record).delegate())
    pos = _wait_for_position(dex, st.position_token_id,
                             lambda p: p is not None and p.liquidity == 0,
                             "drop to zero liquidity")
    assert pos.tokens_owed0 + pos.tokens_owed1 > 0
    st.record = _wait_for_fresh_record(dex, st.position_token_id, stale=st.record).record


@write_tier
def test_collect_the_owed_balances(account_dex, st):
    dex = account_dex
    owed = dex._position_state(st.position_token_id)
    with_retry(lambda: dex.collect(
        pool_key=st.pool.key, amount0_requested=owed.tokens_owed0,
        amount1_requested=owed.tokens_owed1, position_record=st.record).delegate())
    _wait_for_position(dex, st.position_token_id,
                       lambda p: p is not None and p.tokens_owed0 == 0 and p.tokens_owed1 == 0,
                       "clear its owed balances")
    st.record = _wait_for_fresh_record(dex, st.position_token_id, stale=st.record).record


@write_tier
def test_burn_the_emptied_position(account_dex, st):
    dex = account_dex
    with_retry(lambda: dex.burn(pool_key=st.pool.key, position_record=st.record).delegate())
    _wait_for_position(dex, st.position_token_id, lambda p: p is None,
                       "disappear from the positions mapping")
    # The scanner's own view is deliberately not asserted: it marks the
    # record spent on its own schedule.
