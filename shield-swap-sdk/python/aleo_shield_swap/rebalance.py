"""Rebalance planning — the exact close-and-remint quote the contract asserts.

A rebalance (``shield_swap_rebalance_router.aleo``) burns a position, settles
everything it returns, optionally adds funds, and mints the successor range
in ONE transaction.  The contract re-derives every amount at finalize and
asserts equality, so the plan is only submittable at the pool price it was
built against: rebuild after any delay rather than caching one.

Pure functions only — the client feeds them chain state (see
``ShieldSwap.plan_rebalance``); a caller with its own indexer can feed them
directly.  Mirrors veil's ``planRebalance`` field for field.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import _generated as g
from ._routing import rebalance_route
from .position_math import (
    amounts_for_liquidity,
    fee_growth_inside,
    fee_owed,
    liquidity_for_amounts,
    u256_of,
)
from .tick_math import get_sqrt_price_at_tick_x128, round_tick_to_spacing

#: The budget round-trip (floor the liquidity, ceil the amounts) can overshoot
#: a budget by a rounding unit; each retry lowers the target by one.
BUDGET_CLAMP_RETRIES = 8

#: Default request lifetime.  Short on purpose: a plan is only valid at the
#: price it was built against, so a stale transaction almost certainly reverts
#: on the exactness asserts anyway — the deadline fails it cheaply instead.
REBALANCE_DEADLINE_OFFSET_BLOCKS = 20


@dataclass(frozen=True)
class RebalancePlan:
    """The exact rebalance a position can submit against the planned state.

    All amounts are raw base units.  The contract asserts, per side,
    ``recovered + funded == required + refund`` and ``funded * refund == 0``;
    every plan the planner produces satisfies both by construction, and a
    hand-built plan must too.

    ``fees_accrued0/1`` are advisory (already folded into ``recovered``);
    execution never reads them.
    """

    pool_key: str
    position_token_id: str
    #: Successor range, spacing-aligned.
    tick_lower: int
    tick_upper: int
    #: The position's live liquidity — the close removes all of it.
    old_liquidity: int
    #: Fees earned since the position's checkpoint, settled by the close.
    fees_accrued0: int
    fees_accrued1: int
    #: Everything the close returns: principal at the live price + owed + accrued.
    recovered0: int
    recovered1: int
    #: What the successor range needs at the live price (rounded up).
    required0: int
    required1: int
    #: What the caller must add — ``max(required - recovered, 0)``.
    funded0: int
    funded1: int
    #: Surplus paid to the position's withdrawal address.
    refund0: int
    refund1: int
    #: The successor position's exact liquidity.
    liquidity_target: int
    #: The router entrypoint the plan selects.
    function_name: str


@dataclass(frozen=True)
class RebalanceResult:
    """Outcome of a rebalance: the successor position and what was submitted."""

    position_token_id: Optional[str]
    transaction_id: str
    plan: RebalancePlan


def _sizing(liquidity_target: Optional[int], max_funding0: Optional[int],
            max_funding1: Optional[int]) -> None:
    has_target = liquidity_target is not None
    has_budget = max_funding0 is not None or max_funding1 is not None
    if has_target == has_budget:
        raise ValueError("Pass exactly one sizing mode: liquidity_target=, or "
                         "max_funding0= and max_funding1= together")
    if has_target:
        if liquidity_target <= 0:              # type: ignore[operator]
            raise ValueError("liquidity_target must be greater than zero")
        return
    if max_funding0 is None or max_funding1 is None:
        raise ValueError("Budget sizing needs both max_funding0 and max_funding1 "
                         "(0 is a valid budget)")
    if max_funding0 < 0 or max_funding1 < 0:
        raise ValueError("Funding budgets must not be negative")


def plan_rebalance(*, pool_key: str, position_token_id: str,
                   tick_lower: int, tick_upper: int,
                   slot: g.Slot, position: g.Position,
                   lower_tick: g.Tick, upper_tick: g.Tick,
                   wrapped0: bool, wrapped1: bool,
                   liquidity_target: Optional[int] = None,
                   max_funding0: Optional[int] = None,
                   max_funding1: Optional[int] = None) -> RebalancePlan:
    """Build the exact rebalance from pre-read chain state.  Pure.

    Computes everything the transaction asserts: the full recovery of the old
    range (principal at the live price, the settled ``tokens_owed``, and the
    fees accrued since the checkpoints), the successor range's deposit, and
    per side either the funding to add or the surplus to refund.

    Args:
        pool_key: The pool the position belongs to.
        position_token_id: The position to close.
        tick_lower: Successor lower bound, before spacing alignment.
        tick_upper: Successor upper bound.
        slot: The pool's live slot (price, tick, spacing, fee accumulators).
        position: The ``positions`` entry being closed.
        lower_tick: The ``ticks`` entry at the position's CURRENT lower bound.
        upper_tick: The entry at its current upper bound.
        wrapped0: Whether token0 is a wrapper (routes and proofs follow).
        wrapped1: Whether token1 is a wrapper.
        liquidity_target: Exact successor liquidity — one sizing mode.
        max_funding0: Token0 budget on top of what the close returns — the
            other mode, together with *max_funding1*; the planner solves for
            the largest liquidity the budget supports.  ``0`` rebalances on
            recovered funds alone.
        max_funding1: Token1 budget.

    Raises:
        ValueError: On an empty aligned range, both or neither sizing modes,
            or a budget that supports no liquidity in the range.
    """
    _sizing(liquidity_target, max_funding0, max_funding1)

    spacing = int(slot.tick_spacing)
    lo = round_tick_to_spacing(tick_lower, spacing)
    hi = round_tick_to_spacing(tick_upper, spacing)
    if lo >= hi:
        raise ValueError(f"Empty tick range after spacing alignment: [{lo}, {hi})")

    sqrt_price = u256_of(slot.sqrt_price)
    old_liquidity = int(position.liquidity)
    old_lower, old_upper = int(position.tick_lower), int(position.tick_upper)

    # The close settles principal, the owed balances, AND the fees accrued
    # since the checkpoints — the contract asserts the position ends at
    # exactly zero owed, so all three must be in `recovered`.
    inside0, inside1 = fee_growth_inside(
        (u256_of(lower_tick.fee_growth_outside0_x_128),
         u256_of(lower_tick.fee_growth_outside1_x_128)), old_lower,
        (u256_of(upper_tick.fee_growth_outside0_x_128),
         u256_of(upper_tick.fee_growth_outside1_x_128)), old_upper,
        int(slot.tick),
        (u256_of(slot.fee_growth_global0_x_128),
         u256_of(slot.fee_growth_global1_x_128)),
    )
    fees0 = fee_owed(inside0, u256_of(position.fee_growth_inside0_last_x_128), old_liquidity)
    fees1 = fee_owed(inside1, u256_of(position.fee_growth_inside1_last_x_128), old_liquidity)
    principal0, principal1 = amounts_for_liquidity(
        sqrt_price, get_sqrt_price_at_tick_x128(old_lower),
        get_sqrt_price_at_tick_x128(old_upper), old_liquidity)
    recovered0 = principal0 + int(position.tokens_owed0) + fees0
    recovered1 = principal1 + int(position.tokens_owed1) + fees1

    sqrt_lo, sqrt_hi = get_sqrt_price_at_tick_x128(lo), get_sqrt_price_at_tick_x128(hi)

    def required_for(liquidity: int) -> tuple[int, int]:
        # The deposit rounds up: the contract takes at most these amounts, and
        # anything the recovered balances do not cover is funding.
        return amounts_for_liquidity(sqrt_price, sqrt_lo, sqrt_hi, liquidity, round_up=True)

    if liquidity_target is not None:
        target = liquidity_target
        required = required_for(target)
    else:
        assert max_funding0 is not None and max_funding1 is not None
        target = liquidity_for_amounts(sqrt_price, sqrt_lo, sqrt_hi,
                                       recovered0 + max_funding0, recovered1 + max_funding1)
        # The solve floors and the deposit ceils, so the first target can
        # exceed the budget by a rounding unit; step down until it fits.
        for _ in range(BUDGET_CLAMP_RETRIES + 1):
            if target <= 0:
                raise ValueError("The funding budget supports no liquidity in this "
                                 "range — raise it or narrow the range")
            required = required_for(target)
            if (required[0] <= recovered0 + max_funding0
                    and required[1] <= recovered1 + max_funding1):
                break
            target -= 1
        else:
            raise ValueError("Budget sizing did not converge — pass an explicit "
                             "liquidity_target")

    funded0 = max(required[0] - recovered0, 0)
    funded1 = max(required[1] - recovered1, 0)
    refund0 = max(recovered0 - required[0], 0)
    refund1 = max(recovered1 - required[1], 0)
    route = rebalance_route(wrapped0, wrapped1, funded0 > 0, funded1 > 0)

    return RebalancePlan(
        pool_key=pool_key, position_token_id=position_token_id,
        tick_lower=lo, tick_upper=hi, old_liquidity=old_liquidity,
        fees_accrued0=fees0, fees_accrued1=fees1,
        recovered0=recovered0, recovered1=recovered1,
        required0=required[0], required1=required[1],
        funded0=funded0, funded1=funded1, refund0=refund0, refund1=refund1,
        liquidity_target=target, function_name=route.function,
    )
