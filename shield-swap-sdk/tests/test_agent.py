import json

import pytest

from aleo_shield_swap.agent import dispatch_tool, shield_swap_tools
from aleo_shield_swap.client import ShieldSwap


def test_tool_definitions_shape():
    tools = shield_swap_tools()
    names = {t["name"] for t in tools}
    assert {"get_pools", "get_route", "get_slot", "get_balances",
            "swap", "claim_swap_output", "mint"} <= names
    for t in tools:
        assert t["description"]
        assert t["input_schema"]["type"] == "object"
        json.dumps(t)                          # fully serializable


def test_dispatch_get_slot(stub_aleo):
    out = dispatch_tool(ShieldSwap(stub_aleo), "get_slot", {"pool_key": "5field"})
    assert out["tick"] == 4055
    json.dumps(out)


def test_dispatch_swap_returns_serialized_handle(stub_aleo):
    out = dispatch_tool(ShieldSwap(stub_aleo), "swap",
                        {"pool_key": "5field", "token_in_id": "1field",
                         "amount_in": 10**9, "expected_out": 1_000_000,
                         "token_in_program": "tok.aleo"})
    assert out["swap_id"] == "77field"         # delegate path, decoded tx
    assert out["blinding_factor"]
    json.dumps(out)


def test_dispatch_unknown_tool(stub_aleo):
    with pytest.raises(ValueError, match="Unknown"):
        dispatch_tool(ShieldSwap(stub_aleo), "nope", {})
