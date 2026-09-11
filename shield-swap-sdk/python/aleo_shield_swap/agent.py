"""Framework-neutral agent tools over a ShieldSwap client.

``shield_swap_tools()`` returns tool definitions in the Claude API ``tools=``
shape (name / description / input_schema) — they plug into any framework
that accepts JSON-schema tools.  ``dispatch_tool(dex, name, args)`` executes
one against a :class:`~aleo_shield_swap.client.ShieldSwap` (write methods run
``.delegate()``) and returns a JSON-serializable result.  The surface is the
curated lifecycle set — swap handles and counters live in the profile
journal, so agents never carry state between calls; the long tail of methods
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
    # The API's pool document carries the fee as ``fee_percent`` — a legacy
    # name for the fee in basis points ("2" means 0.02%).  There is no ``fee``
    # attribute on it; that lives on the chain-side PoolState.
    return [{"key": p.key, "token0": p.token0, "token1": p.token1,
             "fee_bps": int(p.fee_percent),
             "token0_symbol": p.token0_info.symbol if p.token0_info else None,
             "token1_symbol": p.token1_info.symbol if p.token1_info else None}
            for p in dex.api.get_pools()]


def _h_get_balances(dex: Any, args: dict[str, Any]) -> Any:
    return dex.get_balances(address=args.get("address"))


def _h_setup_account(dex: Any, args: dict[str, Any]) -> Any:
    return _serialize(dex.onboard(referral_code=args.get("referral_code")))


def _h_redeem_referral_code(dex: Any, args: dict[str, Any]) -> Any:
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
    return _serialize(result)          # the client journals the position


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


def _h_get_swap_execution(dex: Any, args: dict[str, Any]) -> Any:
    return _serialize(dex.get_swap_execution(args["swap_id"]))


def _opt_int(args: dict[str, Any], key: str) -> Any:
    value = args.get(key)
    return int(value) if value is not None else None


def _rebalance_kwargs(args: dict[str, Any]) -> dict[str, Any]:
    return dict(pool_key=args["pool_key"], position_token_id=args["position_token_id"],
                tick_lower=int(args["tick_lower"]), tick_upper=int(args["tick_upper"]),
                liquidity_target=_opt_int(args, "liquidity_target"),
                max_funding0=_opt_int(args, "max_funding0"),
                max_funding1=_opt_int(args, "max_funding1"))


#: RebalancePlan fields that are raw u128 amounts — reported as strings so
#: JSON consumers that cannot represent integers above 2^53 keep them exact
#: (the tool descriptions promise "raw base units as strings").
_PLAN_AMOUNT_FIELDS = ("old_liquidity", "fees_accrued0", "fees_accrued1",
                       "recovered0", "recovered1", "required0", "required1",
                       "funded0", "funded1", "refund0", "refund1", "liquidity_target")


def _plan_json(plan: Any) -> dict[str, Any]:
    out = _serialize(plan)
    for key in _PLAN_AMOUNT_FIELDS:
        if out.get(key) is not None:
            out[key] = str(out[key])
    return out


def _h_plan_rebalance(dex: Any, args: dict[str, Any]) -> Any:
    return _plan_json(dex.plan_rebalance(**_rebalance_kwargs(args)))


def _h_rebalance_position(dex: Any, args: dict[str, Any]) -> Any:
    result = _serialize(dex.rebalance_position(**_rebalance_kwargs(args)).delegate())
    if isinstance(result.get("plan"), dict):
        for key in _PLAN_AMOUNT_FIELDS:
            if result["plan"].get(key) is not None:
                result["plan"][key] = str(result["plan"][key])
    return result


_TOOLS: list[tuple[str, str, dict[str, Any], Callable[[Any, dict[str, Any]], Any]]] = [
    ("setup_account",
     "Register this machine's shield-swap profile end to end (auth, "
     "credentials, airdrop, funded check). Nothing is required from the "
     "user: access is granted by authentication alone. referral_code is "
     "optional — pass one only if the user has a friend's code to credit. "
     "Re-running is a safe no-op that reports what was skipped.",
     _schema({"referral_code": _S}, []), _h_setup_account),
    ("redeem_referral_code",
     "Credit a referrer by redeeming their referral code (optional, once "
     "per account; setup_account does this when given referral_code).",
     _schema({"code": _S}, ["code"]), _h_redeem_referral_code),
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
    ("get_swap_execution",
     "The chain's fill receipt for a swap: executed height and, per pool hop, "
     "amounts in/out, gross fee, protocol fee, LP fee, and post-trade price/"
     "tick/liquidity. Survives the claim (unlike the swap output). Returns "
     "null until the swap finalizes.",
     _schema({"swap_id": _S}, ["swap_id"]), _h_get_swap_execution),
    ("plan_rebalance",
     "Quote moving a position to a new tick range in one transaction (testnet "
     "only — the rebalance router is not on mainnet). Size with EXACTLY one "
     "of liquidity_target (exact successor liquidity) or max_funding0 AND "
     "max_funding1 (extra raw units the user will add; 0 and 0 = reuse only "
     "what the old position returns). Reports recovered, required, funded, "
     "and refund amounts per token; show them to the user before executing. "
     "Amounts are raw base units as strings.",
     _schema({"pool_key": _S, "position_token_id": _S, "tick_lower": _I,
              "tick_upper": _I, "liquidity_target": _S, "max_funding0": _S,
              "max_funding1": _S},
             ["pool_key", "position_token_id", "tick_lower", "tick_upper"]),
     _h_plan_rebalance),
    ("rebalance_position",
     "Execute the rebalance plan_rebalance described: burn the old position, "
     "settle its principal and fees, add any funding, and mint the successor "
     "range — atomically. Same sizing arguments as plan_rebalance (the plan is "
     "rebuilt at submit time). Reverts if the pool price moved since planning; "
     "on revert, re-plan and resubmit. Returns the new position id, the "
     "transaction id, and the submitted plan.",
     _schema({"pool_key": _S, "position_token_id": _S, "tick_lower": _I,
              "tick_upper": _I, "liquidity_target": _S, "max_funding0": _S,
              "max_funding1": _S},
             ["pool_key", "position_token_id", "tick_lower", "tick_upper"]),
     _h_rebalance_position),
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
