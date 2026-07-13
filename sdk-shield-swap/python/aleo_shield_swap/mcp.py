"""MCP server exposing shield_swap as tools (the ``[mcp]`` extra).

Run: ``python -m aleo_shield_swap.mcp``

Uses the low-level ``mcp.server.Server`` (not FastMCP) so each tool
advertises the exact JSON schema from :func:`~aleo_shield_swap.agent
.shield_swap_tools` — FastMCP infers schemas from handler signatures, which
would collapse every tool to one opaque ``args`` object.  Tools run against
the synchronous :class:`~aleo_shield_swap.client.ShieldSwap` (the full verb
surface) in a worker thread, keeping the event loop free.

Environment:
    ALEO_ENDPOINT     API origin (default ``https://api.provable.com`` —
                      the provider derives ``/v2`` reads, ``/prove``, and
                      ``/scanner`` from it)
    ALEO_PRIVATE_KEY  Signer for write tools (omit for read-only)
    ALEO_NETWORK      ``testnet`` (default) or ``mainnet``
    ALEO_E2E_API_KEY / ALEO_E2E_CONSUMER_ID
                      Delegated-proving + hosted-scanner credentials
"""
from __future__ import annotations

import json
import os
from typing import Any

try:
    from mcp.server import Server
    from mcp.types import TextContent, Tool
except ImportError as exc:  # pragma: no cover - env-dependent
    raise ImportError(
        "The MCP server requires the mcp package — install the extra: "
        "pip install 'aleo-shield-swap[mcp]'"
    ) from exc

from .agent import dispatch_tool, shield_swap_tools


def tool_definitions() -> list[Tool]:
    """The agent tools as MCP ``Tool`` objects with their exact schemas."""
    return [Tool(name=t["name"], description=t["description"],
                 inputSchema=t["input_schema"])
            for t in shield_swap_tools()]


async def call_tool(dex: Any, name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Execute one tool in a worker thread; result as JSON text content."""
    from anyio import to_thread

    result = await to_thread.run_sync(lambda: dispatch_tool(dex, name, arguments))
    return [TextContent(type="text", text=json.dumps(result))]


def build_server(dex: Any, *, name: str = "shield-swap") -> Server:
    """An MCP server with every agent tool registered against *dex*."""
    server: Server = Server(name)

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return tool_definitions()

    @server.call_tool()
    async def _call_tool(tool_name: str, arguments: dict[str, Any]) -> list[TextContent]:
        return await call_tool(dex, tool_name, arguments)

    return server


def _build_dex() -> Any:
    from aleo import Aleo, HTTPProvider

    from .client import ShieldSwap

    endpoint = os.environ.get("ALEO_ENDPOINT", "https://api.provable.com")
    network = os.environ.get("ALEO_NETWORK", "testnet")
    api_key = os.environ.get("ALEO_E2E_API_KEY")
    aleo = Aleo(HTTPProvider(endpoint, network=network, api_key=api_key))
    consumer = os.environ.get("ALEO_E2E_CONSUMER_ID")
    if consumer:
        aleo.network_client.consumer_id = consumer
    pk = os.environ.get("ALEO_PRIVATE_KEY")
    if pk:
        aleo.default_account = aleo.account.from_private_key(pk)
    return ShieldSwap(aleo)


def main() -> None:
    import anyio
    from mcp.server.stdio import stdio_server

    server = build_server(_build_dex())

    async def _run() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(_run)


if __name__ == "__main__":
    main()
