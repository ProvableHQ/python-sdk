import json

import pytest

mcp = pytest.importorskip("mcp")

from aleo_shield_swap.agent import shield_swap_tools  # noqa: E402
from aleo_shield_swap.client import ShieldSwap  # noqa: E402
from aleo_shield_swap.mcp import build_server, call_tool, tool_definitions  # noqa: E402


def test_tool_definitions_carry_exact_schemas():
    tools = tool_definitions()
    by_name = {t.name: t for t in tools}
    expected = {t["name"]: t for t in shield_swap_tools()}
    assert set(by_name) == set(expected)
    # The precise agent schema must survive — not FastMCP signature inference.
    swap_many = by_name["swap_many"]
    assert swap_many.inputSchema == expected["swap_many"]["input_schema"]
    assert "pool_key" in swap_many.inputSchema["properties"]
    assert "count" in swap_many.inputSchema["required"]


def test_build_server_constructs():
    assert build_server(dex=None).name == "shield-swap"


async def test_call_tool_dispatches_and_serializes(stub_aleo):
    dex = ShieldSwap(stub_aleo)
    dex.api.get_pools = lambda: []          # no network in unit tests
    out = await call_tool(dex, "get_pools", {})
    assert out[0].type == "text"
    assert json.loads(out[0].text) == []
