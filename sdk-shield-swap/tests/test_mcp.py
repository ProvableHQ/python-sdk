import pytest

mcp = pytest.importorskip("mcp")


async def test_server_registers_all_agent_tools():
    from aleo_shield_swap.agent import shield_swap_tools
    from aleo_shield_swap.mcp import build_server

    server = build_server(dex=None)
    tools = await server.list_tools()
    registered = {t.name for t in tools}
    expected = {t["name"] for t in shield_swap_tools()}
    assert expected <= registered
