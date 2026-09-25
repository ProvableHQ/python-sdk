"""MCP server exposing the bridge lifecycle as tools (the ``[mcp]`` extra).

Run: ``python -m aleo_bridge.mcp``

Uses the low-level ``mcp.server.Server`` (not FastMCP) so each tool advertises the exact JSON
schema from :func:`~aleo_bridge.agent.bridge_tools`. Tools run against the synchronous
:class:`~aleo_bridge.client.Bridge` in a worker thread. Writes stay behind ``confirm: true``
exactly as in ``agent.py`` — the MCP transport adds no privileges of its own.

Every ``mcp.*`` import here is lazy: importing this module (and ``import aleo_bridge``) never
requires the ``mcp`` package, and the first call into it that actually needs ``mcp`` raises the
SDK's own :class:`~aleo_bridge.errors.MissingExtraError` naming the real extra
(``pip install 'aleo-bridge-sdk[mcp]'``) instead of a bare ``ImportError`` from deep inside this
module.

Environment (all read by ``Bridge.from_env`` — nothing here reads the environment directly, and
nothing here logs key material):
    BRIDGE_PRIVATE_KEY                       Aleo signer (required)
    ALEO_ENDPOINT / ALEO_NETWORK / ALEO_API_KEY / ALEO_CONSUMER_ID
    ETHEREUM_RPC_URL / EVM_PRIVATE_KEY       Ethereum connection (both or neither)
    SOLANA_RPC_URL / SOLANA_PRIVATE_KEY      Solana connection
    BRIDGE_CHECKPOINT_DIR                    optional FileCheckpointStore directory
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from .agent import bridge_tools, dispatch_tool
from .errors import MissingExtraError

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from mcp.server import Server
    from mcp.types import TextContent, Tool

_FEATURE = "the MCP server"


def _mcp_types() -> tuple[Any, Any]:
    """``(Tool, TextContent)``, or :class:`MissingExtraError` when the extra is not installed."""
    try:
        from mcp.types import TextContent, Tool
    except ImportError as exc:
        raise MissingExtraError("mcp", _FEATURE) from exc
    return Tool, TextContent


def _mcp_server_cls() -> Any:
    """``mcp.server.Server``, or :class:`MissingExtraError` when the extra is not installed."""
    try:
        from mcp.server import Server
    except ImportError as exc:
        raise MissingExtraError("mcp", _FEATURE) from exc
    return Server


def tool_definitions() -> "list[Tool]":
    """The agent tools as MCP ``Tool`` objects with their exact schemas."""
    Tool, _TextContent = _mcp_types()
    return [Tool(name=t["name"], description=t["description"], inputSchema=t["input_schema"])
            for t in bridge_tools()]


async def call_tool(bridge: Any, name: str, arguments: dict[str, Any]) -> "list[TextContent]":
    """Execute one tool in a worker thread; result as JSON text content.

    Runs the synchronous ``dispatch_tool`` off the event loop so a slow chain read never blocks
    other in-flight MCP requests. ``dispatch_tool`` already renders every ``BridgeError`` as a
    JSON-serializable dict and holds writes behind ``confirm: true`` — this function adds nothing
    beyond the thread hop and the JSON encoding.
    """
    _Tool, TextContent = _mcp_types()
    from anyio import to_thread

    result = await to_thread.run_sync(lambda: dispatch_tool(bridge, name, arguments))
    return [TextContent(type="text", text=json.dumps(result))]


def build_server(bridge: Any, *, name: str = "aleo-bridge") -> "Server":
    """An MCP server with every agent tool registered against *bridge*.

    No second tool list: both handlers below delegate straight to :func:`tool_definitions` /
    :func:`call_tool`, which read the same table :func:`~aleo_bridge.agent.bridge_tools` and
    :func:`~aleo_bridge.agent.dispatch_tool` use everywhere else.
    """
    Server = _mcp_server_cls()
    server: Any = Server(name)

    @server.list_tools()
    async def _list_tools() -> "list[Tool]":
        return tool_definitions()

    @server.call_tool()
    async def _call_tool(tool_name: str, arguments: dict[str, Any]) -> "list[TextContent]":
        return await call_tool(bridge, tool_name, arguments)

    return server


def serve(bridge: Any) -> None:
    """Serve the bridge tools over stdio until the client disconnects (blocking)."""
    try:
        import anyio
        from mcp.server.stdio import stdio_server
    except ImportError as exc:
        raise MissingExtraError("mcp", _FEATURE) from exc

    server = build_server(bridge)

    async def _run() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(_run)


def main() -> None:
    """``python -m aleo_bridge.mcp``: bind the bridge from the environment and serve.

    ``Bridge.from_env()`` already reads ``BRIDGE_PRIVATE_KEY``, the EVM/Solana keys and their
    aliases, and ``BRIDGE_CHECKPOINT_DIR`` — this function never re-reads or logs any of them.
    """
    from .client import Bridge

    serve(Bridge.from_env())


if __name__ == "__main__":
    main()


__all__ = ["tool_definitions", "call_tool", "build_server", "serve", "main"]
