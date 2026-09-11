import json

import pytest

from aleo_shield_swap.agent import dispatch_tool, shield_swap_tools
from aleo_shield_swap.types import (CollectReport, MintResult, OnboardReport,
                                    StageOutcome, SwapBatchReport, SwapHandle,
                                    TxResult)

CURATED = {"setup_account", "redeem_referral_code", "request_airdrop", "status",
           "get_swap_execution", "plan_rebalance", "rebalance_position",
           "get_pools", "get_balances", "get_positions", "swap_many",
           "mint_position", "adjust_liquidity", "collect_all"}


def test_tool_surface_is_curated():
    tools = shield_swap_tools()
    assert {t["name"] for t in tools} == CURATED
    for t in tools:
        assert t["description"], f"{t['name']} needs a teaching description"
        assert t["input_schema"]["type"] == "object"
        json.dumps(t)                          # fully serializable
        # Nothing may teach the agent to demand an invite: access is granted
        # by authentication alone and a referral code is optional.
        assert "invite" not in t["description"].lower(), t["name"]
    setup = next(t for t in tools if t["name"] == "setup_account")
    assert "referral_code" in setup["input_schema"]["properties"]
    assert setup["input_schema"].get("required", []) == []


def test_dispatch_setup_account_serializes_report():
    class _Dex:
        def onboard(self, referral_code=None):
            assert referral_code == "C"
            return OnboardReport("aleo1x", [StageOutcome("authenticate", "ran")],
                                 funded=True)

    out = dispatch_tool(_Dex(), "setup_account", {"referral_code": "C"})
    assert out["funded"] is True
    assert out["outcomes"][0]["name"] == "authenticate"
    json.dumps(out)


def test_dispatch_swap_many_and_collect_all():
    handle = SwapHandle(swap_id="s0", blinding_factor="bf",
                        blinded_address="ba", token_in_id="t0",
                        token_out_id="t1", pool_key="pk", amount_in=5,
                        transaction_id="tx", program="p")

    class _Dex:
        def swap_many(self, *, pool_key, token_in_id, amount_in, count,
                      slippage_bps=50):
            assert (pool_key, count) == ("pk", 2)
            return SwapBatchReport(handles=[handle, handle], failures=[])

        def collect_all(self):
            return CollectReport(claimed=[{"swap_id": "s0"}],
                                 still_pending=[], fees=[])

    out = dispatch_tool(_Dex(), "swap_many",
                        {"pool_key": "pk", "token_in_id": "t0",
                         "amount_in": 5, "count": 2})
    assert len(out["handles"]) == 2 and out["handles"][0]["swap_id"] == "s0"
    out2 = dispatch_tool(_Dex(), "collect_all", {})
    assert out2["claimed"] == [{"swap_id": "s0"}] and out2["fees"] == []
    json.dumps(out)
    json.dumps(out2)


def test_dispatch_adjust_liquidity_signs():
    calls = []

    class _Call:
        def delegate(self):
            return TxResult("p1", "tx")

    class _Dex:
        def increase_liquidity(self, **kw):
            calls.append(("inc", kw))
            return _Call()

        def decrease_liquidity(self, **kw):
            calls.append(("dec", kw))
            return _Call()

    dispatch_tool(_Dex(), "adjust_liquidity",
                  {"pool_key": "k", "liquidity_delta": -10})
    dispatch_tool(_Dex(), "adjust_liquidity",
                  {"pool_key": "k", "liquidity_delta": 7})
    assert calls[0][0] == "dec" and calls[0][1]["liquidity_to_remove"] == 10
    assert calls[1][0] == "inc" and calls[1][1]["amount0_desired"] == 7


def test_dispatch_mint_position_serializes():
    class _Call:
        def delegate(self):
            return MintResult("11field", "txm")

    class _Dex:
        def mint(self, **kw):
            return _Call()

    out = dispatch_tool(_Dex(), "mint_position",
                        {"pool_key": "pk", "tick_lower": -60, "tick_upper": 60,
                         "amount0_desired": 1, "amount1_desired": 1})
    assert out["position_token_id"] == "11field"   # journaling lives in client.mint


def test_dispatch_request_airdrop_defaults_to_profile():
    class _Api:
        def request_airdrop(self, address):
            return {"job_id": "j1", "status": "running", "address": address}

    class _Profile:
        address = "aleo1me"

    class _Dex:
        api = _Api()
        profile = _Profile()

    out = dispatch_tool(_Dex(), "request_airdrop", {})
    assert out["address"] == "aleo1me"


def test_dispatch_unknown_tool():
    with pytest.raises(ValueError, match="Unknown"):
        dispatch_tool(object(), "nope", {})


def test_rebalance_and_execution_tools_are_curated():
    names = {t["name"] for t in shield_swap_tools()}
    assert {"plan_rebalance", "rebalance_position", "get_swap_execution"} <= names
    plan = next(t for t in shield_swap_tools() if t["name"] == "plan_rebalance")
    assert set(plan["input_schema"]["required"]) == {"pool_key", "position_token_id",
                                                     "tick_lower", "tick_upper"}
    assert {"liquidity_target", "max_funding0", "max_funding1"} <= set(plan["input_schema"]["properties"])
    assert "testnet" in plan["description"].lower()          # not on mainnet yet


def test_dispatch_plan_and_rebalance_serialize():
    from aleo_shield_swap.rebalance import RebalancePlan, RebalanceResult
    plan = RebalancePlan(pool_key="5field", position_token_id="42field", tick_lower=-60,
                         tick_upper=60, old_liquidity=10, fees_accrued0=0, fees_accrued1=0,
                         recovered0=5, recovered1=6, required0=7, required1=4,
                         funded0=2, funded1=0, refund0=0, refund1=2, liquidity_target=12,
                         function_name="rebalance_plain_plain_one")

    class _Call:
        def delegate(self):
            return RebalanceResult("77field", "at1x", plan)

    class _Dex:
        def plan_rebalance(self, **kw):
            assert kw == {"pool_key": "5field", "position_token_id": "42field",
                          "tick_lower": -60, "tick_upper": 60,
                          "liquidity_target": 12, "max_funding0": None, "max_funding1": None}
            return plan

        def rebalance_position(self, **kw):
            assert kw["max_funding0"] == 0 and kw["max_funding1"] == 0
            assert kw["liquidity_target"] is None
            return _Call()

    out = dispatch_tool(_Dex(), "plan_rebalance",
                        {"pool_key": "5field", "position_token_id": "42field",
                         "tick_lower": -60, "tick_upper": 60, "liquidity_target": "12"})
    # Amounts are raw u128 values reported as STRINGS (exact for JSON consumers
    # limited to 2^53), as the tool description promises; ticks stay numeric.
    assert out["funded0"] == "2" and out["liquidity_target"] == "12"
    assert out["tick_lower"] == -60 and out["function_name"] == "rebalance_plain_plain_one"
    json.dumps(out)
    out = dispatch_tool(_Dex(), "rebalance_position",
                        {"pool_key": "5field", "position_token_id": "42field",
                         "tick_lower": -60, "tick_upper": 60,
                         "max_funding0": "0", "max_funding1": "0"})
    assert out["position_token_id"] == "77field" and out["plan"]["liquidity_target"] == "12"
    json.dumps(out)


def test_dispatch_get_swap_execution():
    from aleo_shield_swap.types import HopFill, SwapExecution

    class _Dex:
        def get_swap_execution(self, swap_id):
            assert swap_id == "77field"
            return SwapExecution("77field", 4242, [HopFill("5field", True, 10, 9, 1, 0, 1,
                                                           1 << 128, 5, 3)])

    out = dispatch_tool(_Dex(), "get_swap_execution", {"swap_id": "77field"})
    assert out["executed_height"] == 4242 and out["hops"][0]["lp_fee"] == 1
    json.dumps(out)
    assert dispatch_tool(type("D", (), {"get_swap_execution": lambda self, s: None})(),
                         "get_swap_execution", {"swap_id": "1field"}) is None


def test_get_pools_tool_reads_the_api_pool_document():
    """The API pool document carries the fee as ``fee_percent`` (basis
    points) and has no ``fee`` attribute — the tool must not reach for one
    (it did, and broke against the live API until the live suite caught it)."""
    from types import SimpleNamespace as NS
    from aleo_shield_swap.agent import dispatch_tool
    entry = NS(key="5field", token0="1field", token1="2field", fee_percent="30",
               token0_info=NS(symbol="ETH"), token1_info=None)
    dex = NS(api=NS(get_pools=lambda: [entry]))
    out = dispatch_tool(dex, "get_pools", {})
    assert out == [{"key": "5field", "token0": "1field", "token1": "2field",
                    "fee_bps": 30, "token0_symbol": "ETH", "token1_symbol": None}]
    json.dumps(out)
