"""MCP stdio server over the agent tools — schema fidelity, dispatch, and the confirm gate.

Every test here exercises the module's functions directly (``tool_definitions`` / ``call_tool`` /
the handlers ``build_server`` registers) against a :class:`FakeBridge`. None of them start a real
stdio server: that would require a live client on the other end of the pipe, which is exactly what
the low-level ``mcp.server.Server`` API lets us skip in tests.
"""
import json
import sys

import pytest

mcp = pytest.importorskip("mcp")

from mcp import types  # noqa: E402

from aleo_bridge.agent import bridge_tools  # noqa: E402
from aleo_bridge.errors import MissingExtraError  # noqa: E402
from aleo_bridge.mcp import build_server, call_tool, tool_definitions  # noqa: E402
from tests.fakes.fake_bridge import ALEO_RECIPIENT, FakeBridge  # noqa: E402


def test_tool_definitions_carry_exact_schemas():
    defs = {t.name: t for t in tool_definitions()}
    expected = {t["name"]: t for t in bridge_tools()}
    assert set(defs) == set(expected)
    assert defs["bridge_execute"].inputSchema == expected["bridge_execute"]["input_schema"]
    assert "confirm" in defs["bridge_execute"].inputSchema["properties"]
    for name, tool in defs.items():
        assert tool.description == expected[name]["description"]


def test_build_server_constructs():
    assert build_server(FakeBridge()).name == "aleo-bridge"


async def test_call_tool_dispatches_and_serializes():
    b = FakeBridge()
    out = await call_tool(b, "bridge_quote", {"source": "ethereum/usdc", "destination": "aleo/usdcx",
                                              "amount": "2", "recipient": ALEO_RECIPIENT})
    assert out[0].type == "text"
    assert json.loads(out[0].text)["kind"] == "evm-xreserve"
    gated = await call_tool(b, "bridge_execute", {"source": "ethereum/usdc", "destination": "aleo/usdcx",
                                                  "amount": "2", "recipient": ALEO_RECIPIENT})
    assert json.loads(gated[0].text)["confirmation_required"] is True and b.events == []


async def test_build_server_handlers_list_and_dispatch_through_the_real_server_wiring():
    """Reach the exact coroutines ``@server.list_tools()``/``@server.call_tool()`` registered —
    not just the module-level helpers they delegate to — and confirm the write gate still holds
    when a tool call is routed through the server object."""
    b = FakeBridge()
    server = build_server(b)

    list_result = await server.request_handlers[types.ListToolsRequest](types.ListToolsRequest())
    names = {t.name for t in list_result.root.tools}
    assert names == {t["name"] for t in bridge_tools()}

    call_result = await server.request_handlers[types.CallToolRequest](
        types.CallToolRequest(params=types.CallToolRequestParams(
            name="bridge_execute",
            arguments={"source": "ethereum/usdc", "destination": "aleo/usdcx", "amount": "2",
                      "recipient": ALEO_RECIPIENT})))
    payload = json.loads(call_result.root.content[0].text)
    assert payload["confirmation_required"] is True
    assert b.events == []


def test_missing_extra_error_without_mcp_installed(monkeypatch):
    """``tool_definitions``/``build_server`` import ``mcp`` lazily: without the extra installed,
    calling them fails with the SDK's own ``MissingExtraError`` (naming the real ``mcp`` extra from
    ``pyproject.toml``), never a bare ``ImportError`` from deep inside the module."""
    for mod in ("mcp", "mcp.types", "mcp.server", "mcp.server.stdio"):
        monkeypatch.setitem(sys.modules, mod, None)

    from aleo_bridge import mcp as bridge_mcp

    with pytest.raises(MissingExtraError) as exc_info:
        bridge_mcp.tool_definitions()
    assert exc_info.value.extra == "mcp"
    assert "aleo-bridge-sdk[mcp]" in str(exc_info.value)

    with pytest.raises(MissingExtraError):
        bridge_mcp.build_server(FakeBridge())
