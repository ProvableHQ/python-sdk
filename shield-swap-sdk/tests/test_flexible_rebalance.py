"""Flexible rebalance constraints, planning and submitted request regression tests."""
from dataclasses import replace

import pytest

from aleo_shield_swap import _generated as g
from aleo_shield_swap.client import ShieldSwap
from aleo_shield_swap.rebalance import plan_rebalance
from .test_rebalance import _state, _stub, POOL, POSITION_ID, OLD_LOWER, OLD_UPPER, TOKEN_RECORD

COMMON = dict(pool_key=POOL, position_token_id=POSITION_ID,
              tick_lower=OLD_LOWER, tick_upper=OLD_UPPER)


def test_zero_target_survives_equal_bounds():
    p = plan_rebalance(**_state(), liquidity_target=0, amount0_desired=10_000, amount1_desired=10_000)
    assert p.liquidity_target == 0 and p.estimated_liquidity > 0
    assert p.required0 <= 10_000 and p.required1 <= 10_000
    assert p.amount0_min == p.amount1_min == 10_000
    assert (p.funded0, p.funded1, p.refund0, p.refund1) == (0, 0, 0, 0)


def test_amount_budgets_include_fixed_transfers():
    p = plan_rebalance(**_state(), liquidity_target=0, amount0_desired=1_000_000,
                       amount1_desired=1_000_000, amount0_min=0, amount1_min=0,
                       funded0=123, refund1=456)
    assert (p.funded0, p.funded1, p.refund0, p.refund1) == (123, 0, 0, 456)
    assert p.required0 <= p.recovered0 + 123 and p.required1 <= p.recovered1 - 456
    assert p.function_name == "rebalance_plain_plain_one"


def test_one_sided_amount_budget():
    p = plan_rebalance(**_state(tick_lower=600, tick_upper=1200), liquidity_target=0,
                       amount0_desired=5_000, amount1_desired=0, amount0_min=0, amount1_min=0)
    assert p.estimated_liquidity > 0
    assert 0 < p.required0 <= 5_000 and p.required1 == 0
    assert p.function_name == "rebalance_plain_plain_none"


def test_explicit_exact_bounds_derive_quote_transfers():
    exact = plan_rebalance(**_state(), liquidity_target=500_000)
    p = plan_rebalance(**_state(), liquidity_target=500_000,
                       amount0_desired=exact.required0, amount1_desired=exact.required1)
    assert (p.refund0, p.refund1) == (exact.refund0, exact.refund1)
    assert p.funded0 == p.funded1 == 0


@pytest.mark.parametrize("overrides, message", [
    ({"amount0_desired": -1}, "u128"),
    ({"amount0_desired": 1 << 128}, "u128"),
    ({"funded0": True}, "u128"),
    ({"amount0_min": 10_001}, "minimum"),
    ({"funded0": 1, "refund0": 1}, "funding.*refund"),
    ({"refund0": 1_000_000}, "refunds exceed"),
    ({"amount1_desired": None}, "together"),
])
def test_invalid_constraints(overrides, message):
    args = dict(liquidity_target=0, amount0_desired=10_000, amount1_desired=10_000,
                amount0_min=0, amount1_min=0)
    args.update(overrides)
    with pytest.raises(ValueError, match=message):
        plan_rebalance(**_state(), **args)


@pytest.mark.parametrize("args, message", [
    ({"liquidity_target": 0}, "zero.*requires"),
    ({"liquidity_target": 1, "funded0": 1}, "explicit token bounds"),
    ({"max_funding0": 0, "max_funding1": 0,
      "amount0_desired": 10, "amount1_desired": 10}, "require liquidity_target"),
    ({"liquidity_target": 1, "amount0_min": 0}, "minimums require"),
    ({"liquidity_target": 500_000, "amount0_desired": 1, "amount1_desired": 1}, "quoted deposit"),
])
def test_ambiguous_bounds(args, message):
    with pytest.raises(ValueError, match=message):
        plan_rebalance(**_state(), **args)


def test_helpers_preserve_modes_and_prebuilt_bounds():
    dex = ShieldSwap(_stub())
    exact = dex.plan_rebalance_exact_amounts(**COMMON, liquidity_target=500_000)
    dex.rebalance_position_exact_amounts(plan=exact)
    with pytest.raises(ValueError, match="positive"):
        dex.plan_rebalance_exact_amounts(**COMMON, liquidity_target=0)
    p = dex.plan_rebalance_to_liquidity(**COMMON, liquidity_target=500_000,
        amount0_desired=150_000, amount1_desired=160_000,
        amount0_min=90_000, amount1_min=80_000, funded0=123, refund1=456)
    result = dex.rebalance_position_to_liquidity(plan=p, token0_record=TOKEN_RECORD).transact()
    req = next(g.CoreRebalanceRequest.from_plaintext(x) for x in dex._aleo.last_call[1]
               if x.startswith("{ old_liquidity:"))
    assert req.liquidity_target == 500_000
    assert (req.mint.amount0_desired, req.mint.amount1_desired) == (150_000, 160_000)
    assert (req.mint.amount0_min, req.mint.amount1_min) == (90_000, 80_000)
    assert req.funded0 == 123 and req.refund1 == 456 and result.plan == p
    with pytest.raises(ValueError, match="equal"):
        dex.rebalance_position_exact_amounts(plan=p)
    with pytest.raises(ValueError, match="wider"):
        dex.rebalance_position_to_liquidity(plan=replace(exact, amount0_desired=exact.required0,
                                                        amount1_desired=exact.required1))
    p = dex.plan_rebalance_from_amounts(**COMMON, amount0_desired=10_000, amount1_desired=10_000)
    assert p.amount0_min == p.amount1_min == 0
    dex.rebalance_position_from_amounts(plan=p).transact()
    req = g.CoreRebalanceRequest.from_plaintext(dex._aleo.last_call[1][2])
    assert req.liquidity_target == 0
    assert req.mint.amount0_desired == 10_000 and req.mint.amount0_min == 0
    with pytest.raises(ValueError, match="zero"):
        dex.plan_rebalance_from_amounts(**COMMON, liquidity_target=1,
                                        amount0_desired=10_000, amount1_desired=10_000)
    with pytest.raises(ValueError, match="fixed funding"):
        dex.plan_rebalance_from_amounts(**COMMON, max_funding0=0, max_funding1=0,
                                        amount0_desired=10_000, amount1_desired=10_000)


def test_prebuilt_plan_rejects_overrides_and_bad_accounting():
    p = plan_rebalance(**_state(), liquidity_target=500_000)
    dex = ShieldSwap(_stub())
    with pytest.raises(ValueError, match="plan.*overrid"):
        dex.rebalance_position(plan=p, liquidity_target=1)
    with pytest.raises(ValueError, match="recovery.*funding"):
        dex.rebalance_position(plan=replace(p, refund0=p.refund0 + 1))


@pytest.mark.asyncio
@pytest.mark.parametrize("mode, constraints", [
    ("exact_amounts", {"liquidity_target": 500_000}),
    ("to_liquidity", {"liquidity_target": 500_000, "amount0_desired": 150_000, "amount1_desired": 160_000}),
    ("from_amounts", {"amount0_desired": 10_000, "amount1_desired": 10_000}),
])
async def test_async_and_exported_planners_match_sync(mode, constraints):
    import aleo_shield_swap as sdk
    from .test_async_client import AsyncStubAleo
    stub = _stub()
    sync = getattr(ShieldSwap(stub), f"plan_rebalance_{mode}")(**COMMON, **constraints)
    async_dex = sdk.AsyncShieldSwap(AsyncStubAleo(mappings=stub.programs._mappings))
    planned = await getattr(async_dex, f"plan_rebalance_{mode}")(**COMMON, **constraints)
    assert planned == sync
    assert getattr(sdk, f"plan_rebalance_{mode}")(**_state(), **constraints) == sync


@pytest.mark.parametrize("mode, constraints", [
    ("exact_amounts", {"liquidity_target": "500000"}),
    ("to_liquidity", {"liquidity_target": "500000", "amount0_desired": "150000", "amount1_desired": "160000"}),
    ("from_amounts", {"amount0_desired": str((1 << 100) + 123), "amount1_desired": "10000"}),
    ("", {"liquidity_target": "0", "amount0_desired": "10000", "amount1_desired": "10000", "amount0_min": "0", "amount1_min": "0"}),
])
def test_agent_modes_preserve_constraints_and_integer_precision(mode, constraints):
    from aleo_shield_swap.agent import dispatch_tool, shield_swap_tools
    suffix = f"_{mode}" if mode else ""
    tools = {t["name"]: t for t in shield_swap_tools()}
    plan_name, execute_name = "plan_rebalance" + suffix, "rebalance_position" + suffix
    assert plan_name in tools and execute_name in tools
    assert set(constraints) <= tools[plan_name]["input_schema"]["properties"].keys()
    dex = ShieldSwap(_stub())
    args = {**COMMON, **constraints}
    planned = dispatch_tool(dex, plan_name, args)
    result = dispatch_tool(dex, execute_name, args)
    assert result["position_token_id"] == "77field"
    assert result["plan"] == planned
    if mode == "from_amounts":
        assert planned["amount0_desired"] == str((1 << 100) + 123)
        assert planned["liquidity_target"] == "0"
        assert isinstance(planned["estimated_liquidity"], str)
    if mode == "to_liquidity":
        assert planned["amount0_min"] == "0" and planned["refund0"] == "0"


@pytest.mark.parametrize("mode", ["from_amounts", "to_liquidity"])
def test_bounded_helpers_reject_invalid_values_before_chain_reads(mode):
    dex = ShieldSwap(_stub())
    with pytest.raises(ValueError, match="u128"):
        getattr(dex, f"plan_rebalance_{mode}")(**COMMON, liquidity_target=0 if mode == "from_amounts" else 1,
                                               amount0_desired=True, amount1_desired=10_000)


@pytest.mark.asyncio
async def test_mcp_exposes_and_executes_amount_planner():
    import json
    pytest.importorskip("mcp")
    from aleo_shield_swap.mcp import tool_definitions, call_tool
    tools = {tool.name: tool for tool in tool_definitions()}
    assert "amount0_desired" in tools["plan_rebalance_from_amounts"].inputSchema["required"]
    result = await call_tool(ShieldSwap(_stub()), "plan_rebalance_from_amounts",
                            {**COMMON, "amount0_desired": "10000", "amount1_desired": "10000"})
    plan = json.loads(result[0].text)
    assert plan["liquidity_target"] == "0" and int(plan["estimated_liquidity"]) > 0


@pytest.mark.parametrize("value", [True, 1.5, "1.5"])
def test_agent_rejects_amounts_that_would_be_coerced_or_truncated(value):
    from aleo_shield_swap.agent import dispatch_tool
    with pytest.raises(ValueError, match="integer|decimal"):
        dispatch_tool(ShieldSwap(_stub()), "plan_rebalance_from_amounts",
                      {**COMMON, "amount0_desired": value, "amount1_desired": "10000"})
