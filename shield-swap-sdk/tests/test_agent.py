import json

import pytest

from aleo_shield_swap.agent import dispatch_tool, shield_swap_tools
from aleo_shield_swap.types import (CollectReport, MintResult, OnboardReport,
                                    StageOutcome, SwapBatchReport, SwapHandle,
                                    TxResult)

CURATED = {"setup_account", "redeem_invite", "request_airdrop", "status",
           "get_pools", "get_balances", "get_positions", "swap_many",
           "mint_position", "adjust_liquidity", "collect_all"}


def test_tool_surface_is_curated():
    tools = shield_swap_tools()
    assert {t["name"] for t in tools} == CURATED
    for t in tools:
        assert t["description"], f"{t['name']} needs a teaching description"
        assert t["input_schema"]["type"] == "object"
        json.dumps(t)                          # fully serializable


def test_dispatch_setup_account_serializes_report():
    class _Dex:
        def onboard(self, invite_code=None):
            assert invite_code == "C"
            return OnboardReport("aleo1x", [StageOutcome("authenticate", "ran")],
                                 funded=True)

    out = dispatch_tool(_Dex(), "setup_account", {"invite_code": "C"})
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
