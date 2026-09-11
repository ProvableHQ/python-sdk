"""The agent tool surface against live testnet — the read-only tools an
agent runs before it spends anything, dispatched exactly as an MCP/Claude
tool call would be (name + JSON args → JSON-serializable result).  The
write tools (swap_many, mint, collect_all, rebalance) are proved end to end
by ``test_agent_lifecycle_live.py``."""
from __future__ import annotations

import json

import pytest

from aleo_shield_swap.agent import dispatch_tool, shield_swap_tools

from .conftest import account_tier

pytestmark = pytest.mark.live


def test_tool_definitions_are_well_formed():
    tools = shield_swap_tools()
    names = [t["name"] for t in tools]
    assert len(names) == len(set(names)) and "status" in names and "swap_many" in names
    for tool in tools:
        schema = tool["input_schema"]
        assert schema["type"] == "object"
        assert set(schema.get("required", [])) <= set(schema["properties"])
        json.dumps(tool)                            # Claude API `tools=` shape


def test_read_only_tools_dispatch_against_live_state(live_dex_module):
    pools = dispatch_tool(live_dex_module, "get_pools", {})
    assert pools and {"key", "token0", "token1", "fee_bps"} <= set(pools[0])
    assert all(isinstance(p["fee_bps"], int) for p in pools)
    json.dumps(pools)
    # A finalized-swap receipt for an id nobody swapped under is null, not an error.
    assert dispatch_tool(live_dex_module, "get_swap_execution", {"swap_id": "0field"}) is None
    with pytest.raises(ValueError):
        dispatch_tool(live_dex_module, "not_a_tool", {})


@account_tier
def test_account_tools_reorient_from_live_chain_and_api(account_dex):
    status = dispatch_tool(account_dex, "status", {})
    json.dumps(status)
    assert status["authenticated"] is True and status["has_access"] is True
    assert status["address"].startswith("aleo1")
    balances = dispatch_tool(account_dex, "get_balances", {})
    json.dumps(balances)
    assert balances                                   # the e2e account holds tokens
    for entry in balances.values():
        assert entry["total"] == entry["public"] + entry["private"]
    positions = dispatch_tool(account_dex, "get_positions", {})
    json.dumps(positions)
    assert isinstance(positions, list)
