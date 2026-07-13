"""MCP server exposing shield_swap as tools (the ``[mcp]`` extra).

Run: ``python -m aleo_shield_swap.mcp``

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

import os
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - env-dependent
    raise ImportError(
        "The MCP server requires the mcp package — install the extra: "
        "pip install 'aleo-shield-swap[mcp]'"
    ) from exc

from .agent import dispatch_tool, shield_swap_tools


def build_server(dex: Any, *, name: str = "shield-swap") -> "FastMCP":
    """A FastMCP server with every agent tool registered against *dex*.

    Tools run in a worker thread (the ShieldSwap client is synchronous), so
    the event loop stays free.
    """
    from anyio import to_thread

    server = FastMCP(name)
    for tool in shield_swap_tools():
        def make_handler(tool_name: str):
            async def handler(args: dict[str, Any]) -> Any:
                return await to_thread.run_sync(
                    lambda: dispatch_tool(dex, tool_name, args)
                )
            return handler

        server.add_tool(
            make_handler(tool["name"]),
            name=tool["name"],
            description=tool["description"],
        )
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
    build_server(_build_dex()).run()


if __name__ == "__main__":
    main()
