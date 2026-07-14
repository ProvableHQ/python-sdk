"""Framework-neutral agent tools over a ShieldSwap client.

``shield_swap_tools()`` returns tool definitions in the Claude API ``tools=``
shape (name / description / input_schema) — they plug into any framework
that speaks JSON-schema tools.  ``dispatch_tool(dex, name, args)`` executes
one against a :class:`~aleo_shield_swap.client.ShieldSwap` (write verbs run
``.delegate()``) and returns a JSON-serializable result; handles serialize
as dicts so an agent can persist and resume the two-step swap flow.
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Callable

from .types import SwapHandle

_S = {"type": "string"}
_I = {"type": "integer"}


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required}


def _serialize(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    raw = getattr(value, "raw", None)                  # SlotView and friends
    if raw is not None and is_dataclass(raw) and not isinstance(raw, type):
        return asdict(raw)
    return value


def _h_get_pools(dex: Any, args: dict[str, Any]) -> Any:
    return [{"key": p.key, "token0": p.token0, "token1": p.token1,
             "fee": p.fee,
             "token0_symbol": p.token0_info.symbol if p.token0_info else None,
             "token1_symbol": p.token1_info.symbol if p.token1_info else None}
            for p in dex.api.get_pools()]


def _h_get_route(dex: Any, args: dict[str, Any]) -> Any:
    return _serialize(dex.api.get_route(
        token_in=args["token_in"], token_out=args["token_out"],
        amount_in=args.get("amount_in")))


def _h_get_slot(dex: Any, args: dict[str, Any]) -> Any:
    return _serialize(dex.get_slot(args["pool_key"]))


def _h_get_balances(dex: Any, args: dict[str, Any]) -> Any:
    return dex.get_balances(address=args.get("address"))


def _h_swap(dex: Any, args: dict[str, Any]) -> Any:
    call = dex.swap(
        pool_key=args["pool_key"], token_in_id=args["token_in_id"],
        amount_in=int(args["amount_in"]),
        slippage_bps=int(args.get("slippage_bps", 50)),
        expected_out=(int(args["expected_out"])
                      if args.get("expected_out") is not None else None),
        token_in_program=args.get("token_in_program"))
    return asdict(call.delegate())


def _h_claim(dex: Any, args: dict[str, Any]) -> Any:
    handle = SwapHandle.from_json(json.dumps(args["handle"]))
    return asdict(dex.claim_swap_output(handle).delegate())


def _h_increase(dex: Any, args: dict[str, Any]) -> Any:
    call = dex.increase_liquidity(
        pool_key=args["pool_key"],
        amount0_desired=int(args["amount0_desired"]),
        amount1_desired=int(args["amount1_desired"]),
        token0_program=args.get("token0_program"),
        token1_program=args.get("token1_program"))
    return asdict(call.delegate())


def _h_decrease(dex: Any, args: dict[str, Any]) -> Any:
    call = dex.decrease_liquidity(
        pool_key=args["pool_key"],
        liquidity_to_remove=int(args["liquidity_to_remove"]))
    return asdict(call.delegate())


def _h_collect(dex: Any, args: dict[str, Any]) -> Any:
    call = dex.collect(
        pool_key=args["pool_key"],
        amount0_requested=int(args["amount0_requested"]),
        amount1_requested=int(args["amount1_requested"]))
    return asdict(call.delegate())


def _h_burn(dex: Any, args: dict[str, Any]) -> Any:
    return asdict(dex.burn(pool_key=args["pool_key"]).delegate())


def _h_create_pool(dex: Any, args: dict[str, Any]) -> Any:
    call = dex.create_pool(
        token0_id=args["token0_id"], token1_id=args["token1_id"],
        fee=int(args["fee"]), initial_tick=int(args["initial_tick"]))
    return asdict(call.delegate())


def _h_mint(dex: Any, args: dict[str, Any]) -> Any:
    call = dex.mint(
        pool_key=args["pool_key"],
        tick_lower=int(args["tick_lower"]), tick_upper=int(args["tick_upper"]),
        amount0_desired=int(args["amount0_desired"]),
        amount1_desired=int(args["amount1_desired"]),
        token0_program=args.get("token0_program"),
        token1_program=args.get("token1_program"))
    return asdict(call.delegate())


_TOOLS: list[tuple[str, str, dict[str, Any], Callable[[Any, dict[str, Any]], Any]]] = [
    ("get_pools", "List shield_swap pools with their token pairs and fee tiers.",
     _schema({}, []), _h_get_pools),
    ("get_route", "Quote a trade route between two tokens; returns the estimated output.",
     _schema({"token_in": _S, "token_out": _S, "amount_in": _I},
             ["token_in", "token_out"]), _h_get_route),
    ("get_slot", "Live pool state: sqrt price, current tick, in-range liquidity.",
     _schema({"pool_key": _S}, ["pool_key"]), _h_get_slot),
    ("get_balances", "Public + private + total balances per token for an address.",
     _schema({"address": _S}, []), _h_get_balances),
    ("swap", "Request a private swap (phase 1 of 2). Returns the swap handle — "
             "persist it; claim_swap_output consumes it after finalization.",
     _schema({"pool_key": _S, "token_in_id": _S, "amount_in": _I,
              "slippage_bps": _I, "expected_out": _I, "token_in_program": _S},
             ["pool_key", "token_in_id", "amount_in"]), _h_swap),
    ("claim_swap_output", "Claim a private swap's output (phase 2). Takes the "
                          "handle returned by swap; retry if not finalized yet.",
     _schema({"handle": {"type": "object"}}, ["handle"]), _h_claim),
    ("mint", "Mint a concentrated-liquidity position over a tick range.",
     _schema({"pool_key": _S, "tick_lower": _I, "tick_upper": _I,
              "amount0_desired": _I, "amount1_desired": _I,
              "token0_program": _S, "token1_program": _S},
             ["pool_key", "tick_lower", "tick_upper",
              "amount0_desired", "amount1_desired"]), _h_mint),
    ("increase_liquidity", "Add funds to an existing position (range fixed at mint).",
     _schema({"pool_key": _S, "amount0_desired": _I, "amount1_desired": _I,
              "token0_program": _S, "token1_program": _S},
             ["pool_key", "amount0_desired", "amount1_desired"]), _h_increase),
    ("decrease_liquidity", "Remove liquidity from a position; owed amounts become collectable.",
     _schema({"pool_key": _S, "liquidity_to_remove": _I},
             ["pool_key", "liquidity_to_remove"]), _h_decrease),
    ("collect", "Collect owed token amounts from a position.",
     _schema({"pool_key": _S, "amount0_requested": _I, "amount1_requested": _I},
             ["pool_key", "amount0_requested", "amount1_requested"]), _h_collect),
    ("burn", "Burn an empty position NFT.",
     _schema({"pool_key": _S}, ["pool_key"]), _h_burn),
    ("create_pool", "Create a new pool for a token pair at a registered fee tier.",
     _schema({"token0_id": _S, "token1_id": _S, "fee": _I, "initial_tick": _I},
             ["token0_id", "token1_id", "fee", "initial_tick"]), _h_create_pool),
]


def shield_swap_tools() -> list[dict[str, Any]]:
    """Tool definitions (Claude API ``tools=`` shape)."""
    return [{"name": name, "description": desc, "input_schema": schema}
            for name, desc, schema, _ in _TOOLS]


def dispatch_tool(dex: Any, name: str, args: dict[str, Any]) -> Any:
    """Execute one tool against *dex*; returns a JSON-serializable result."""
    for tool_name, _, _, handler in _TOOLS:
        if tool_name == name:
            return handler(dex, args)
    raise ValueError(f"Unknown shield_swap tool: {name!r}")
