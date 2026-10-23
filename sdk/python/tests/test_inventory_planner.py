"""Planner known answers.

_source: ProvableHQ/veil packages/core/test/inventory/inventory.test.ts
("inventory planning") and configuration/autojoin suites, PR #172 @ 2e324d5.
Amounts and step shapes are copied as-is; the planner is a line-for-line port
with one extension, the whole-lot split (python-sdk#83), which only changes
plans Veil would reject.
"""
from __future__ import annotations

import pytest

from aleo.inventory import (
    InventoryError, InventoryPlan, InventoryPlanError, InventoryTarget, PlannedRecord, RecordAsset, RecordJoinProgram,
    ResolvedRecordAsset, TokenJoinRouter, build_inventory_plan,
)

SCOPE = '["testnet","aleo1owner"]'
CREDITS = ResolvedRecordAsset("credits.aleo", standard="credits", record_name="credits", amount_field="microcredits", amount_type="u64")
ARC20 = ResolvedRecordAsset("test_arc20_eth.aleo", standard="arc20", record_name="Token", amount_field="amount", amount_type="u128")
ROUTER = TokenJoinRouter("test_aj_arc20_2_15.aleo")


def plan(amounts: list[int], count: int, asset: RecordAsset = ARC20, distribution: str = "balanced", **kw: object) -> InventoryPlan:
    return build_inventory_plan(
        scope=SCOPE, asset=asset, target=InventoryTarget(records=count, distribution=distribution),  # type: ignore[arg-type]
        inputs=[PlannedRecord(f"{i}group", amount) for i, amount in enumerate(amounts)], **kw,  # type: ignore[arg-type]
    )


def test_splits_one_arc_token_record_into_exact_balanced_lots() -> None:
    result = plan([100], 4)
    assert len(result.steps) == 3
    assert [step.outputs[0].amount for step in result.steps] == [25, 25, 25]
    assert result.deduction == 0


def test_accounts_for_every_native_credits_split_deduction() -> None:
    result = plan([1_030_000], 4, CREDITS)
    assert result.deduction == 30_000
    assert [step.amount for step in result.steps] == [250_000, 250_000, 250_000]


def test_does_nothing_for_balanced_inventory_and_tolerates_atomic_rounding() -> None:
    assert plan([25, 25, 25, 25], 4).steps == ()
    assert plan([33, 33, 34], 3).steps == ()


def test_preserves_existing_records_for_count_only_policies() -> None:
    result = plan([10, 15, 25, 50], 3, ARC20, "preserve")
    assert len(result.steps) == 1
    assert result.steps[0].inputs == ("0group", "1group")


def test_preserves_exact_balanced_lots_and_repacks_small_records() -> None:
    assert len(plan([25, 25, 50], 4).steps) == 1
    assert len(plan([250_000, 250_000, 510_000], 4, CREDITS).steps) == 1
    result = build_inventory_plan(
        scope=SCOPE, asset=ARC20, inputs=[PlannedRecord("1group", 3), PlannedRecord("2group", 3)],
        target=InventoryTarget(records=3, min_record_amount=2),
    )
    assert [step.kind for step in result.steps] == ["join", "split", "split"]
    assert [record.amount for record in result.steps[-1].outputs] == [2, 2]


def test_splits_whole_lots_when_halving_would_strand_value_below_the_minimum() -> None:
    """Python extends Veil here: the same inputs raise in Veil's planner (python-sdk#83)."""
    result = build_inventory_plan(
        scope=SCOPE, asset=ARC20, inputs=[PlannedRecord("1group", 9), PlannedRecord("2group", 3)],
        target=InventoryTarget(records=4, min_record_amount=3, distribution="preserve"),
    )
    assert [[o.amount for o in step.outputs] for step in result.steps] == [[3, 6], [3, 3]]
    # Credits: 20,003 microcredits fund two 10,000 deductions and three lots of 1,
    # but a first halving would leave neither half able to pay the second split.
    credits = build_inventory_plan(
        scope=SCOPE, asset=CREDITS, inputs=[PlannedRecord("1group", 20_003)],
        target=InventoryTarget(records=3, min_record_amount=1, distribution="preserve"),
    )
    assert [[o.amount for o in step.outputs] for step in credits.steps] == [[1, 10_002], [1, 1]]
    assert credits.deduction == 20_000
    # Halving still wins whenever it can finish, so existing plans are unchanged.
    assert [[o.amount for o in step.outputs] for step in plan([100], 2, ARC20, "preserve").steps] == [[50, 50]]


def test_rejects_impossible_counts_deductions_overflow_and_operation_limits() -> None:
    with pytest.raises(InventoryPlanError, match="infeasible"):
        plan([10_001], 2, CREDITS)
    with pytest.raises(InventoryPlanError, match="records"):
        plan([10], 0)
    with pytest.raises(InventoryPlanError, match="overflow"):
        plan([(1 << 128) - 1, 1], 1)
    with pytest.raises(InventoryPlanError, match="max_transactions"):
        build_inventory_plan(
            scope=SCOPE, asset=ARC20, inputs=[PlannedRecord("1group", 100)], target=InventoryTarget(records=4),
            max_transactions=2,
        )


@pytest.mark.parametrize("count", [2, 15, 16, 30])
def test_consolidates_records_in_bounded_router_transactions(count: int) -> None:
    result = plan([100] * count, 1, ARC20, "preserve", token_join=ROUTER, max_transactions=3)
    assert result.token_join == ROUTER
    assert len(result.steps) == -(-(count - 1) // 14)
    assert all(2 <= len(step.inputs) <= 15 for step in result.steps)


def test_router_is_dropped_for_credits_and_native_pairs_remain() -> None:
    result = plan([100, 100, 100], 1, CREDITS, "preserve", token_join=ROUTER)
    assert result.token_join is None
    assert [len(step.inputs) for step in result.steps] == [2, 2]


def test_honors_custom_batch_ceilings() -> None:
    result = plan([100] * 4, 1, ARC20, "preserve", token_join=TokenJoinRouter(ROUTER.program, max_records=3))
    assert [len(step.inputs) for step in result.steps] == [3, 2]
    with pytest.raises(InventoryError, match="2 to 15"):
        plan([100] * 4, 1, ARC20, "preserve", token_join=TokenJoinRouter(ROUTER.program, max_records=16))


def test_preserves_target_lots_with_router_then_splits_natively() -> None:
    preserve = plan([100] * 16, 4, ARC20, "preserve", token_join=ROUTER)
    assert len(preserve.steps) == 1 and len(preserve.steps[0].inputs) == 13
    balanced = plan([100] * 16, 4, ARC20, "balanced", token_join=ROUTER)
    assert [step.kind for step in balanced.steps] == ["join", "join", "split", "split", "split"]


STATIC = tuple(
    RecordJoinProgram(f"autojoin_credits_{lo}_{hi}.aleo", "records", tuple(range(lo, hi + 1)))
    for lo, hi in ((2, 10), (11, 14), (15, 16))
)


@pytest.mark.parametrize("count,helper", [(3, "autojoin_credits_2_10.aleo"), (11, "autojoin_credits_11_14.aleo"),
                                          (15, "autojoin_credits_15_16.aleo"), (16, "autojoin_credits_15_16.aleo")])
def test_uses_one_static_helper_transaction(count: int, helper: str) -> None:
    result = plan([100] * count, 1, CREDITS, "preserve", join_programs=STATIC, max_transactions=1)
    assert len(result.steps) == 1
    assert result.steps[0].join_program is not None and result.steps[0].join_program.program == helper
    assert len(result.steps[0].inputs) == count


def test_chains_a_sixteen_record_helper_into_a_native_join() -> None:
    result = plan([100] * 17, 1, CREDITS, "preserve", join_programs=STATIC, max_transactions=2)
    assert [(step.join_program.program if step.join_program else None, len(step.inputs)) for step in result.steps] == [
        ("autojoin_credits_15_16.aleo", 16), (None, 2),
    ]


def test_falls_back_to_native_pairs_without_helpers() -> None:
    result = plan([100] * 3, 1, CREDITS, "preserve")
    assert [len(step.inputs) for step in result.steps] == [2, 2]
    assert all(step.join_program is None for step in result.steps)


def test_plan_round_trips_through_dict() -> None:
    result = plan([100] * 16, 4, ARC20, "balanced", token_join=ROUTER, join_programs=STATIC)
    restored = InventoryPlan.from_dict(result.to_dict())
    assert restored.steps == result.steps
    assert restored.token_join == ROUTER
    assert restored.join_programs == STATIC
    assert restored.asset == RecordAsset(ARC20.program, "arc20", "Token", "amount", "u128")
    assert restored.target == result.target


def test_rebuilding_a_plan_detects_edits() -> None:
    original = plan([1_000_000], 2, CREDITS, "preserve")
    rebuilt = build_inventory_plan(
        scope=original.scope, asset=original.asset, target=original.target, inputs=original.inputs,
        max_transactions=original.max_transactions, token_join=original.token_join, join_programs=original.join_programs,
    )
    assert rebuilt == original
    edited = InventoryPlan.from_dict({**original.to_dict(), "steps": [{**s, "amount": "1"} for s in original.to_dict()["steps"]]})
    assert edited != original
