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
    swap = by_name["swap"]
    assert swap.inputSchema == expected["swap"]["input_schema"]
    assert "pool_key" in swap.inputSchema["properties"]
    assert "amount_in" in swap.inputSchema["required"]


def test_build_server_constructs():
    assert build_server(dex=None).name == "shield-swap"


async def test_call_tool_dispatches_and_serializes(stub_aleo):
    out = await call_tool(ShieldSwap(stub_aleo), "get_slot", {"pool_key": "5field"})
    assert out[0].type == "text"
    assert json.loads(out[0].text)["tick"] == 4055
