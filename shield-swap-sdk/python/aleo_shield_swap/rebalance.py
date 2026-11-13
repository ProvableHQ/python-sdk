"""Plan a complete position replacement from a supplied chain snapshot.

Exact bounds enforce quoted recovery and deposits. Flexible bounds use actual
recovery; a zero target derives liquidity from token budgets at execution.
Planning never signs or submits a transaction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

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

#: Default request lifetime, in blocks, to limit exposure to state changes.
REBALANCE_DEADLINE_OFFSET_BLOCKS = 20


@dataclass(frozen=True)
class RebalancePlan:
    """Describe submitted constraints and estimates for a replacement position.

    Amounts use integer token base units. Flexible surplus stays collectible
    on the successor and earns no liquidity fees until reinvested.

    Attributes:
        pool_key: Pool shared by the old and replacement positions.
        position_token_id: Position to close completely.
        tick_lower: Spacing-aligned lower bound of the replacement range.
        tick_upper: Spacing-aligned upper bound of the replacement range.
        old_liquidity: Full liquidity removed from the old position.
        fees_accrued0: Estimated token0 fees since the last checkpoint.
        fees_accrued1: Estimated token1 fees since the last checkpoint.
        recovered0: Quoted token0 principal plus owed and accrued fees.
        recovered1: Quoted token1 recovery; exact quotes enforce both values.
        required0: Estimated token0 consumed, also the cap if bounds are absent.
        required1: Estimated token1 consumed, also the cap if bounds are absent.
        funded0: Fixed extra token0 supplied from private records.
        funded1: Fixed extra token1 supplied from private records.
        refund0: Fixed immediate token0 payout to the withdrawal address.
        refund1: Fixed immediate token1 payout to the withdrawal address.
        liquidity_target: Positive exact liquidity, or zero for execution-time sizing.
        function_name: Router entry point determined by tokens and funded sides.
        amount0_desired: Token0 consumption cap; None uses required0.
        amount1_desired: Token1 consumption cap; None uses required1.
        amount0_min: Token0 consumption floor; None uses its cap.
        amount1_min: Token1 consumption floor; None uses its cap.
        estimated_liquidity: Advisory liquidity when the target is zero. Never
            submitted as the target or a guaranteed minimum liquidity.
    """

    pool_key: str
    position_token_id: str
    tick_lower: int
    tick_upper: int
    old_liquidity: int
    fees_accrued0: int
    fees_accrued1: int
    recovered0: int
    recovered1: int
    required0: int
    required1: int
    funded0: int
    funded1: int
    refund0: int
    refund1: int
    liquidity_target: int
    function_name: str
    amount0_desired: Optional[int] = None
    amount1_desired: Optional[int] = None
    amount0_min: Optional[int] = None
    amount1_min: Optional[int] = None
    estimated_liquidity: Optional[int] = None


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
        if liquidity_target < 0:              # type: ignore[operator]
            raise ValueError("liquidity_target must not be negative")
        return
    if max_funding0 is None or max_funding1 is None:
        raise ValueError("Budget sizing needs both max_funding0 and max_funding1 "
                         "(0 is a valid budget)")
    if max_funding0 < 0 or max_funding1 < 0:
        raise ValueError("Funding budgets must not be negative")


_AMOUNT_FIELDS = ("amount0_desired", "amount1_desired", "amount0_min", "amount1_min",
                  "funded0", "funded1", "refund0", "refund1")
_PLAN_INPUT_FIELDS = ("pool_key", "position_token_id", "tick_lower", "tick_upper",
                      "liquidity_target", "max_funding0", "max_funding1", *_AMOUNT_FIELDS)


def _validate_amounts(params: Mapping[str, Any]) -> None:
    """Reject invalid contract integers, bounds, transfers, and exact accounting."""
    fields = (*_AMOUNT_FIELDS, "liquidity_target", "old_liquidity", "recovered0",
              "recovered1", "required0", "required1", "max_funding0", "max_funding1")
    for key in fields:
        value = params.get(key)
        if value is not None and (type(value) is not int or not 0 <= value < 1 << 128):
            raise ValueError(f"{key} must be a u128 integer (0 through 2**128 - 1)")
    if params.get("liquidity_target") is not None and any(
            params.get(key) is not None for key in ("max_funding0", "max_funding1")):
        raise ValueError("Pass exactly one sizing mode: liquidity_target, or max_funding0 and max_funding1")
    if (params.get("amount0_desired") is None) != (params.get("amount1_desired") is None):
        raise ValueError("Supply amount0_desired and amount1_desired together")
    if params.get("amount0_desired") is None and any(
            params.get(key) is not None for key in ("amount0_min", "amount1_min")):
        raise ValueError("Token minimums require amount0_desired and amount1_desired")
    desired = [params.get(f"amount{i}_desired") for i in range(2)]
    desired = [cap if cap is not None else params.get(f"required{i}")
               for i, cap in enumerate(desired)]
    minimum = [params.get(f"amount{i}_min") for i in range(2)]
    minimum = [value if value is not None else desired[i] for i, value in enumerate(minimum)]
    flexible = params.get("liquidity_target") == 0 or any(
        lo is not None and cap is not None and lo < cap for lo, cap in zip(minimum, desired))
    for i, cap in enumerate(desired):
        if cap is not None and minimum[i] > cap:
            raise ValueError(f"Token{i} minimum exceeds its desired amount")
        funded, refund = params.get(f"funded{i}"), params.get(f"refund{i}")
        if (funded or 0) > 0 and (refund or 0) > 0:
            raise ValueError(f"Token{i} cannot have both funding and a refund")
        recovered, required = params.get(f"recovered{i}"), params.get(f"required{i}")
        if not flexible and cap is not None and required is not None and cap != required:
            raise ValueError(f"Exact token{i} amount must match the liquidity target's quoted deposit")
        if (not flexible and cap is not None and recovered is not None
                and funded is not None and refund is not None):
            if recovered + funded != cap + refund:
                raise ValueError(f"Exact token{i} recovery and funding must equal its desired amount and refund")


def _validate_planning(params: Mapping[str, Any]) -> None:
    _validate_amounts(params)
    _sizing(params.get("liquidity_target"), params.get("max_funding0"), params.get("max_funding1"))
    explicit = params.get("amount0_desired") is not None
    if explicit and params.get("liquidity_target") is None:
        raise ValueError("Explicit token bounds require liquidity_target; use 0 for execution-time sizing")
    if params.get("liquidity_target") == 0 and not explicit:
        raise ValueError("A zero liquidity_target requires amount0_desired and amount1_desired")
    if not explicit and any(params.get(key) is not None for key in ("funded0", "funded1", "refund0", "refund1")):
        raise ValueError("Fixed funding and refunds require explicit token bounds")


def _validate_plan_arguments(plan: RebalancePlan, params: Mapping[str, Any]) -> None:
    if any(params.get(key) is not None for key in _PLAN_INPUT_FIELDS):
        raise ValueError("A supplied plan cannot be combined with planning overrides")
    _validate_amounts(asdict(plan))
    if plan.liquidity_target == 0 and plan.amount0_desired is None:
        raise ValueError("A zero liquidity_target requires amount0_desired and amount1_desired")


def _mode_parameters(mode: str, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Apply a named helper's guarantee without altering a prebuilt plan."""
    result = dict(kwargs)
    plan = result.get("plan")
    if plan is not None:
        _validate_plan_arguments(plan, result)
        params = asdict(plan)
    else:
        params = result
    if mode != "exact_amounts":
        if any(params.get(key) is not None for key in ("max_funding0", "max_funding1")):
            raise ValueError("Use fixed funding (funded0/funded1), not max_funding, with explicit token bounds")
        if mode == "from_amounts":
            if params.get("liquidity_target") not in (None, 0):
                raise ValueError("Amount-driven rebalancing requires a zero liquidity_target")
            params["liquidity_target"] = 0
        elif params.get("liquidity_target") is None or params["liquidity_target"] <= 0:
            raise ValueError("Exact liquidity requires a positive liquidity_target")
        if any(params.get(f"amount{i}_desired") is None for i in range(2)):
            raise ValueError("Supply amount0_desired and amount1_desired for bounded rebalancing")
        # Omitted minima on an existing plan mean exact caps, not helper defaults.
        if plan is None:
            for i in range(2):
                if params.get(f"amount{i}_min") is None:
                    params[f"amount{i}_min"] = 0
    _validate_amounts(params)
    minima = [params.get(f"amount{i}_min") for i in range(2)]
    caps = [params.get(f"amount{i}_desired") for i in range(2)]
    caps = [cap if cap is not None else params.get(f"required{i}") for i, cap in enumerate(caps)]
    minima = [lo if lo is not None else caps[i] for i, lo in enumerate(minima)]
    if mode == "exact_amounts":
        target = params.get("liquidity_target")
        if target is not None and target <= 0:
            raise ValueError("Exact amounts require a positive liquidity_target")
        if minima != caps:
            raise ValueError("Exact amounts require minimum and desired amounts to be equal")
    elif mode == "to_liquidity" and minima == caps:
        raise ValueError("Flexible recovery requires a wider bound on at least one token")
    return result


def plan_rebalance(*, pool_key: str, position_token_id: str,
                   tick_lower: int, tick_upper: int,
                   slot: g.Slot, position: g.Position,
                   lower_tick: g.Tick, upper_tick: g.Tick,
                   wrapped0: bool, wrapped1: bool,
                   liquidity_target: Optional[int] = None,
                   max_funding0: Optional[int] = None,
                   max_funding1: Optional[int] = None,
                   amount0_desired: Optional[int] = None,
                   amount1_desired: Optional[int] = None,
                   amount0_min: Optional[int] = None,
                   amount1_min: Optional[int] = None,
                   funded0: Optional[int] = None, funded1: Optional[int] = None,
                   refund0: Optional[int] = None, refund1: Optional[int] = None) -> RebalancePlan:
    """Plan a replacement position from pre-read pool, position, and tick state.

    Without token bounds, a positive ``liquidity_target`` or paired
    ``max_funding0/1`` budgets produce an exact quote. Explicit desired amounts
    cap consumption; minimums default to those caps. Widening either bound
    uses actual recovery, with fixed funding/refunds defaulting to zero.
    A zero target estimates liquidity from the capped available funds, but
    leaves the submitted target at zero for execution-time sizing.

    All amount arguments accept integers from 0 through 2**128 - 1.
    ``max_funding`` budgets cannot be combined with explicit bounds. Bounds
    constrain token consumption, not minimum liquidity. Raises ValueError for
    invalid bounds, conflicting transfers, or budgets supporting no liquidity.
    No network reads, signing, or submission occurs.
    """
    _validate_planning(locals())

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

    explicit = amount0_desired is not None
    amount_mode = liquidity_target == 0
    available0 = recovered0 + (funded0 or 0) - (refund0 or 0)
    available1 = recovered1 + (funded1 or 0) - (refund1 or 0)
    if explicit and (available0 < 0 or available1 < 0):
        raise ValueError("Fixed refunds exceed the quoted recovery and funding")
    if liquidity_target is not None and not amount_mode:
        target = liquidity_target
        required = required_for(target)
    else:
        if amount_mode:
            assert amount0_desired is not None and amount1_desired is not None
            budget0, budget1 = min(available0, amount0_desired), min(available1, amount1_desired)
        else:
            assert max_funding0 is not None and max_funding1 is not None
            budget0, budget1 = recovered0 + max_funding0, recovered1 + max_funding1
        target = liquidity_for_amounts(sqrt_price, sqrt_lo, sqrt_hi, budget0, budget1)
        for _ in range(BUDGET_CLAMP_RETRIES + 1):
            if target <= 0:
                raise ValueError("The funding budget supports no liquidity in this "
                                 "range — raise it or narrow the range")
            required = required_for(target)
            if required[0] <= budget0 and required[1] <= budget1:
                break
            target -= 1
        else:
            raise ValueError("Budget sizing did not converge — pass an explicit liquidity_target")

    min0 = amount0_min if amount0_min is not None else amount0_desired
    min1 = amount1_min if amount1_min is not None else amount1_desired
    flexible = amount_mode or (explicit and (
        (min0 is not None and amount0_desired is not None and min0 < amount0_desired) or
        (min1 is not None and amount1_desired is not None and min1 < amount1_desired)))
    funded0 = funded0 if funded0 is not None else (0 if flexible else max(required[0] - recovered0, 0))
    funded1 = funded1 if funded1 is not None else (0 if flexible else max(required[1] - recovered1, 0))
    refund0 = refund0 if refund0 is not None else (0 if flexible else max(recovered0 - required[0], 0))
    refund1 = refund1 if refund1 is not None else (0 if flexible else max(recovered1 - required[1], 0))
    route = rebalance_route(wrapped0, wrapped1, funded0 > 0, funded1 > 0)

    plan = RebalancePlan(
        pool_key=pool_key, position_token_id=position_token_id,
        tick_lower=lo, tick_upper=hi, old_liquidity=old_liquidity,
        fees_accrued0=fees0, fees_accrued1=fees1,
        recovered0=recovered0, recovered1=recovered1,
        required0=required[0], required1=required[1],
        funded0=funded0, funded1=funded1, refund0=refund0, refund1=refund1,
        liquidity_target=0 if amount_mode else target, function_name=route.function,
        amount0_desired=amount0_desired, amount1_desired=amount1_desired,
        amount0_min=min0, amount1_min=min1,
        estimated_liquidity=target if amount_mode else None,
    )
    _validate_amounts(asdict(plan))
    return plan


def plan_rebalance_exact_amounts(**kwargs: Any) -> RebalancePlan:
    """Plan exact quoted recovery and deposits; price or fee changes can reject execution.

    Accepts the state and constraints of :func:`plan_rebalance`.
    No network reads, signing, or submission occurs.
    """
    return plan_rebalance(**_mode_parameters("exact_amounts", kwargs))


def plan_rebalance_to_liquidity(**kwargs: Any) -> RebalancePlan:
    """Plan exact successor liquidity using actual recovery and bounded token consumption.

    Accepts the state and constraints of :func:`plan_rebalance`.
    No network reads, signing, or submission occurs.
    """
    return plan_rebalance(**_mode_parameters("to_liquidity", kwargs))


def plan_rebalance_from_amounts(**kwargs: Any) -> RebalancePlan:
    """Estimate successor liquidity from token caps; the contract sizes it at execution.

    Accepts the state and constraints of :func:`plan_rebalance`.
    No network reads, signing, or submission occurs.
    """
    return plan_rebalance(**_mode_parameters("from_amounts", kwargs))
