"""Plan the joins and splits that reshape an inventory, without any network access.

The planner is deterministic: the same records, target, and helpers always
produce the same steps, so a plan built for review can be rebuilt at
execution time and compared. It follows one bounded strategy: keep records
that already fit, split large records directly into the missing lots when
the amounts divide exactly, and otherwise consolidate and then split. Every
amount stays within the asset's declared width.
"""
from __future__ import annotations

from typing import Literal, Sequence

from .assets import max_amount, split_deduction
from .join_programs import validate_record_join_program, validate_token_join_router
from .types import (
    MAX_PLAN_TRANSACTIONS, MAX_TARGET_RECORDS, InventoryPlan, InventoryPlanError, InventoryStep, InventoryTarget,
    PlannedRecord, RecordAsset, RecordJoinProgram, StepKind, TokenJoinRouter,
)


def positive_integer(value: int, name: str, maximum: int = MAX_TARGET_RECORDS) -> None:
    """Require an integer from 1 through *maximum*.

    Raises:
        InventoryPlanError: The value is not an integer in range.
    """
    if isinstance(value, bool) or value < 1 or value > maximum:
        raise InventoryPlanError(f"{name} must be an integer from 1 to {maximum}")


def build_inventory_plan(
    *,
    scope: str,
    asset: RecordAsset,
    target: InventoryTarget,
    inputs: Sequence[PlannedRecord],
    max_transactions: int | None = None,
    token_join: TokenJoinRouter | Literal[False] | None = None,
    join_programs: Sequence[RecordJoinProgram] | None = None,
) -> InventoryPlan:
    """Compute the joins and splits that take *inputs* to *target*.

    Reads nothing from the network and reserves nothing. The result lists
    each step's inputs, predicted outputs, and credits deduction so it can be
    reviewed before execution.

    Args:
        scope: Chain and account the plan belongs to.
        asset: The resolved asset; its width bounds every amount.
        target: Desired count, minimum amount, and distribution.
        inputs: Eligible records as nonces and amounts.
        max_transactions: Most steps allowed, 1 through 10000. Defaults to 100.
        token_join: Explicit dynamic router for ARC tokens; ``False`` or
            ``None`` means none was configured.
        join_programs: Deployed helpers available for batch joins.

    Raises:
        InventoryPlanError: A bound is invalid, the target cannot be reached
            from the inputs, an intermediate join would overflow the width, or
            the plan needs more steps than allowed.
    """
    if token_join:
        validate_token_join_router(token_join)
    router = token_join if (asset.program != "credits.aleo" and token_join) else None
    helpers: tuple[RecordJoinProgram, ...] = ()
    if join_programs:
        for helper in join_programs:
            validate_record_join_program(helper)
        helpers = tuple(join_programs)
    max_join = validate_token_join_router(router) if router else 2
    positive_integer(target.records, "records")
    limit = 100 if max_transactions is None else max_transactions
    positive_integer(limit, "max_transactions", MAX_PLAN_TRANSACTIONS)
    minimum = target.min_record_amount
    tolerance = target.tolerance_bps
    ceiling = max_amount(asset)
    if isinstance(minimum, bool) or minimum < 1 or minimum > ceiling:
        raise InventoryPlanError("Invalid minimum record amount")
    if isinstance(tolerance, bool) or tolerance < 0 or tolerance > 10_000:
        raise InventoryPlanError("Invalid balance tolerance")
    if target.distribution not in ("preserve", "balanced"):
        raise InventoryPlanError("Invalid inventory distribution")
    records = tuple(PlannedRecord(item.id, item.amount) for item in inputs)
    if len({item.id for item in records}) != len(records):
        raise InventoryPlanError("Duplicate inventory records")
    if any(item.amount <= 0 or item.amount > ceiling for item in records):
        raise InventoryPlanError("Invalid record amount")

    working: list[PlannedRecord] = list(records)
    steps: list[InventoryStep] = []
    total = sum(item.amount for item in working)
    deduction = split_deduction(asset)
    count = target.records

    def balanced() -> bool:
        # Compare against the ideal equal share in integer arithmetic so
        # fractional base units cannot introduce rounding drift.
        for item in working:
            difference = abs(item.amount * count - total)
            # One base unit of rounding never warrants another transaction.
            if difference > count and difference * 10_000 > total * tolerance:
                return False
        return True

    satisfied = (
        len(working) == count and all(item.amount >= minimum for item in working)
        and (target.distribution != "balanced" or balanced())
    )

    def finish() -> InventoryPlan:
        return InventoryPlan(
            scope=scope, asset=asset, target=target, inputs=records, steps=tuple(steps),
            deduction=sum(step.deduction for step in steps), max_transactions=limit,
            token_join=router, join_programs=helpers or None,
        )

    def push(kind: StepKind, consumed: Sequence[PlannedRecord], amounts: Sequence[int], amount: int | None = None) -> None:
        # Replace consumed records with symbolic outputs that later steps can
        # reference; execution resolves them to confirmed records.
        nonlocal working
        if len(steps) >= limit:
            raise InventoryPlanError("Inventory plan exceeds max_transactions")
        if any(value < minimum or value > ceiling for value in amounts):
            raise InventoryPlanError("Inventory target is infeasible within token amount bounds")
        outputs = tuple(PlannedRecord(f"step:{len(steps)}:{index}", value) for index, value in enumerate(amounts))
        steps.append(InventoryStep(
            kind=kind, inputs=tuple(item.id for item in consumed), outputs=outputs, amount=amount,
            deduction=deduction if kind == "split" else 0,
        ))
        consumed_ids = {item.id for item in consumed}
        working = [item for item in working if item.id not in consumed_ids] + list(outputs)

    # Preserve exact lots when larger records split directly into the missing
    # lots; this avoids a consolidate-and-split round trip for [25, 25, 50].
    if not satisfied and target.distribution == "balanced" and len(working) < count:
        # Each split adds one record and may deduct credits; compute the final
        # lot size after every deduction before deciding whether lots can stay.
        final_total = total - deduction * (count - len(working))
        unit = final_total // count
        if (
            unit >= minimum and final_total % count == 0
            and all((item.amount + deduction) % (unit + deduction) == 0 for item in working)
        ):
            for original in list(working):
                record = original
                while record.amount > unit:
                    push("split", [record], [unit, record.amount - unit - deduction], unit)
                    record = steps[-1].outputs[1]
            return finish()

    if not satisfied:
        if not working or total < count * minimum:
            raise InventoryPlanError("Insufficient available inventory for target")
        # Repack only when existing records cannot be preserved: balanced
        # layouts or undersized records.
        capacity = sum((item.amount + deduction) // (minimum + deduction) for item in working)
        repack = (
            target.distribution == "balanced" or any(item.amount < minimum for item in working) or capacity < count
        )
        reduce_to = 1 if repack else count
        # Join the smallest records first, with ids breaking ties so rebuilding
        # the same plan produces the same dependency sequence.
        while len(working) > reduce_to:
            working.sort(key=lambda item: (item.amount, item.id))
            needed = len(working) - reduce_to + 1
            # Prefer the largest helper that preserves the requested count. A
            # final pair uses the native join unless a router was configured.
            helper_count = max([2, *(c for helper in helpers for c in helper.counts if c <= needed)])
            batch = min(max_join if router else helper_count, needed)
            amount = sum(item.amount for item in working[:batch])
            # Shrink the batch if its amount would overflow or its exact arity is unavailable.
            while batch > 2 and (
                amount > ceiling or (not router and not any(batch in helper.counts for helper in helpers))
            ):
                batch -= 1
                amount -= working[batch].amount
            helper = next((h for h in helpers if batch in h.counts), None) if (not router and batch > 2) else None
            consumed = working[:batch]
            # Intermediate joins can be below the output minimum while accumulating dust.
            if amount > ceiling:
                raise InventoryPlanError("Join would overflow the asset amount width")
            if len(steps) >= limit:
                raise InventoryPlanError("Inventory plan exceeds max_transactions")
            output = PlannedRecord(f"step:{len(steps)}:0", amount)
            steps.append(InventoryStep(
                kind="join", inputs=tuple(item.id for item in consumed), outputs=(output,), join_program=helper,
            ))
            working = working[batch:] + [output]
        # Split the largest remaining record and leave enough in its remainder
        # for any dependent splits, including their credits deductions.
        while len(working) < count:
            working.sort(key=lambda item: (-item.amount, item.id))
            record = working[0]
            outputs_needed = count - len(working) + 1 if repack else 2
            usable = record.amount - deduction * (outputs_needed - 1)
            amount = usable // outputs_needed
            push("split", [record], [amount, record.amount - amount - deduction], amount)
        if any(item.amount < minimum for item in working):
            raise InventoryPlanError("Insufficient balance after split deductions")
    return finish()


__all__ = ["build_inventory_plan", "positive_integer"]
