"""Use Shield Swap from an MCP client.

Install ``shield-swap-sdk[mcp]`` and run
``python -m aleo_shield_swap.mcp``. By default, the server uses the local
profile to retain the account, API credentials, and unfinished swaps
between sessions.

Set up the account before starting the server. Import an existing account
to keep its address and funds, or create one for a separate address.
Use macOS Keychain or encrypted storage to protect the private key; see
:meth:`~aleo_shield_swap.profile.Profile.load_or_create`. Without setup,
the first run can create an account with its key stored as plaintext.

An encrypted account needs ``ALEO_ACCOUNT_PASSWORD`` or a password saved
in the macOS Keychain: a server connected over stdio cannot answer a
password prompt.

Environment:
    SHIELD_SWAP_HOME  Profile directory; defaults to ``~/.shield-swap``.
    ALEO_ACCOUNT     Account name; defaults to the profile's saved name,
                     then ``default``.
    ALEO_ACCOUNTS_DIR Account directory; defaults to ``~/.aleo/accounts``.
    ALEO_PRIVATE_KEY Use this key directly, without the profile's journal
                     or credentials. Supply it again on each server start.
    ALEO_ENDPOINT    API service root for a direct-key client; defaults to
                     ``https://edge.provable.com/api``.
    ALEO_NETWORK     ``testnet`` (default) or ``mainnet`` for a direct-key
                     client. A profile keeps its recorded network.
    ALEO_E2E_API_KEY / ALEO_E2E_CONSUMER_ID
                     Optional service credentials for a direct-key client.
                     A profile uses its saved credentials instead.
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
        "pip install 'shield-swap-sdk[mcp]'"
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

    endpoint = os.environ.get("ALEO_ENDPOINT", "https://edge.provable.com/api")
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
    # Keep pending swaps and credentials available when the MCP client restarts.
    return ShieldSwap.from_profile()


def main() -> None:
    """Serve Shield Swap tools until the MCP client disconnects.

    Requires ``shield-swap-sdk[mcp]``. Prepare the account and any password
    before starting; see the module's setup instructions. Without
    ``ALEO_PRIVATE_KEY``, the server uses the saved profile and journal.
    Setting that variable starts a client without profile persistence.
    """
    import anyio
    from mcp.server.stdio import stdio_server

    server = build_server(_build_dex())

    async def _run() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(_run)


if __name__ == "__main__":
    main()
