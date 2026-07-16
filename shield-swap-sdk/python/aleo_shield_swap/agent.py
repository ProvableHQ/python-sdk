"""Framework-neutral agent tools over a ShieldSwap client.

``shield_swap_tools()`` returns tool definitions in the Claude API ``tools=``
shape (name / description / input_schema) — they plug into any framework
that speaks JSON-schema tools.  ``dispatch_tool(dex, name, args)`` executes
one against a :class:`~aleo_shield_swap.client.ShieldSwap` (write verbs run
``.delegate()``) and returns a JSON-serializable result.  The surface is the
curated lifecycle set — swap handles and counters live in the profile
journal, so agents never carry state between calls; the long tail of verbs
is reachable by writing Python against the client instead.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Callable

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


def _h_get_balances(dex: Any, args: dict[str, Any]) -> Any:
    return dex.get_balances(address=args.get("address"))


def _h_setup_account(dex: Any, args: dict[str, Any]) -> Any:
    return _serialize(dex.onboard(invite_code=args.get("invite_code")))


def _h_redeem_invite(dex: Any, args: dict[str, Any]) -> Any:
    return _serialize(dex.api.redeem_code(args["code"]))


def _h_request_airdrop(dex: Any, args: dict[str, Any]) -> Any:
    address = args.get("address") or (dex.profile.address if dex.profile
                                      else None)
    if not address:
        raise ValueError("No address: pass address= or bind a profile.")
    return _serialize(dex.api.request_airdrop(address))


def _h_status(dex: Any, args: dict[str, Any]) -> Any:
    return _serialize(dex.status())


def _h_get_positions(dex: Any, args: dict[str, Any]) -> Any:
    return _serialize(dex.get_positions())


def _h_swap_many(dex: Any, args: dict[str, Any]) -> Any:
    return _serialize(dex.swap_many(
        pool_key=args["pool_key"], token_in_id=args["token_in_id"],
        amount_in=int(args["amount_in"]), count=int(args["count"]),
        slippage_bps=int(args.get("slippage_bps", 50))))


def _h_mint_position(dex: Any, args: dict[str, Any]) -> Any:
    result = dex.mint(
        pool_key=args["pool_key"],
        tick_lower=int(args["tick_lower"]), tick_upper=int(args["tick_upper"]),
        amount0_desired=int(args["amount0_desired"]),
        amount1_desired=int(args["amount1_desired"]),
        token0_program=args.get("token0_program"),
        token1_program=args.get("token1_program")).delegate()
    if dex.journal is not None and result.position_token_id:
        dex.journal.record_position(result.position_token_id,
                                    args["pool_key"], result.transaction_id)
    return _serialize(result)


def _h_adjust_liquidity(dex: Any, args: dict[str, Any]) -> Any:
    delta = int(args["liquidity_delta"])
    if delta >= 0:
        call = dex.increase_liquidity(pool_key=args["pool_key"],
                                      amount0_desired=delta,
                                      amount1_desired=delta)
    else:
        call = dex.decrease_liquidity(pool_key=args["pool_key"],
                                      liquidity_to_remove=-delta)
    return _serialize(call.delegate())


def _h_collect_all(dex: Any, args: dict[str, Any]) -> Any:
    return _serialize(dex.collect_all())


_TOOLS: list[tuple[str, str, dict[str, Any], Callable[[Any, dict[str, Any]], Any]]] = [
    ("setup_account",
     "Register this machine's shield-swap profile end to end (auth, invite "
     "redeem, credentials, airdrop, funded check). Pass invite_code on the "
     "first run; re-running is a safe no-op that reports what was skipped.",
     _schema({"invite_code": _S}, []), _h_setup_account),
    ("redeem_invite",
     "Redeem an invite code for the authenticated account (setup_account "
     "does this for you; use this only for manual control).",
     _schema({"code": _S}, ["code"]), _h_redeem_invite),
    ("request_airdrop",
     "Queue the test-token airdrop (private records; one claim per address "
     "per 15 minutes). Defaults to the profile's own address.",
     _schema({"address": _S}, []), _h_request_airdrop),
    ("status",
     "Re-orient: registration state, balances, open positions, pending swap "
     "claims, counter cursor. Run this FIRST in any session.",
     _schema({}, []), _h_status),
    ("get_pools", "List shield_swap pools with their token pairs and fee tiers.",
     _schema({}, []), _h_get_pools),
    ("get_balances", "Public + private + total balances per token for an address.",
     _schema({"address": _S}, []), _h_get_balances),
    ("get_positions",
     "Open liquidity positions — journaled ones plus any recovered by "
     "scanning the account's records.",
     _schema({}, []), _h_get_positions),
    ("swap_many",
     "Fire N private swaps with journal-reserved counters; handles are "
     "journaled for collect_all. Requires a funded, registered profile.",
     _schema({"pool_key": _S, "token_in_id": _S, "amount_in": _I,
              "count": _I, "slippage_bps": _I},
             ["pool_key", "token_in_id", "amount_in", "count"]), _h_swap_many),
    ("mint_position",
     "Mint a concentrated-liquidity position over a tick range (journaled).",
     _schema({"pool_key": _S, "tick_lower": _I, "tick_upper": _I,
              "amount0_desired": _I, "amount1_desired": _I,
              "token0_program": _S, "token1_program": _S},
             ["pool_key", "tick_lower", "tick_upper",
              "amount0_desired", "amount1_desired"]), _h_mint_position),
    ("adjust_liquidity",
     "Resize a position: positive liquidity_delta adds that much of each "
     "token (increase), negative removes liquidity (decrease; owed amounts "
     "become collectable).",
     _schema({"pool_key": _S, "liquidity_delta": _I},
             ["pool_key", "liquidity_delta"]), _h_adjust_liquidity),
    ("collect_all",
     "Claim every finalized swap and collect owed LP fees, from the journal. "
     "Safe to run any time; reports what is still pending.",
     _schema({}, []), _h_collect_all),
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
