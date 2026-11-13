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
from functools import partial

from .rebalance import _AMOUNT_FIELDS
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


def _opt_int(args: dict[str, Any], key: str) -> int | None:
    value = args.get(key)
    if value is None:
        return None
    if type(value) is int:
        return value
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        return int(value)
    raise ValueError(f"{key} must be an integer or an unsigned decimal integer string")


def _rebalance_kwargs(args: dict[str, Any]) -> dict[str, Any]:
    result = dict(pool_key=args["pool_key"], position_token_id=args["position_token_id"],
                tick_lower=int(args["tick_lower"]), tick_upper=int(args["tick_upper"]),
                liquidity_target=_opt_int(args, "liquidity_target"),
                max_funding0=_opt_int(args, "max_funding0"),
                max_funding1=_opt_int(args, "max_funding1"))
    result.update({key: _opt_int(args, key) for key in _AMOUNT_FIELDS if args.get(key) is not None})
    return result


#: RebalancePlan fields that are raw u128 amounts — reported as strings so
#: JSON consumers that cannot represent integers above 2^53 keep them exact
#: (the tool descriptions promise "raw base units as strings").
_PLAN_AMOUNT_FIELDS = ("old_liquidity", "fees_accrued0", "fees_accrued1",
                       "recovered0", "recovered1", "required0", "required1",
                       "funded0", "funded1", "refund0", "refund1", "liquidity_target",
                       "amount0_desired", "amount1_desired", "amount0_min", "amount1_min",
                       "estimated_liquidity")


def _plan_json(plan: Any) -> dict[str, Any]:
    out = _serialize(plan)
    for key in _PLAN_AMOUNT_FIELDS:
        if out.get(key) is not None:
            out[key] = str(out[key])
    return out


def _h_plan_rebalance(dex: Any, args: dict[str, Any], *, mode: str = "") -> Any:
    method = getattr(dex, "plan_rebalance" + (f"_{mode}" if mode else ""))
    return _plan_json(method(**_rebalance_kwargs(args)))


def _h_rebalance_position(dex: Any, args: dict[str, Any], *, mode: str = "") -> Any:
    method = getattr(dex, "rebalance_position" + (f"_{mode}" if mode else ""))
    result = _serialize(method(**_rebalance_kwargs(args)).delegate())
    if isinstance(result.get("plan"), dict):
        for key in _PLAN_AMOUNT_FIELDS:
            if result["plan"].get(key) is not None:
                result["plan"][key] = str(result["plan"][key])
    return result


_REBALANCE_REQUIRED = ["pool_key", "position_token_id", "tick_lower", "tick_upper"]
_REBALANCE_PROPERTIES = {
    "pool_key": _S, "position_token_id": _S, "tick_lower": _I, "tick_upper": _I,
    "liquidity_target": {**_S, "description": "Positive exact liquidity; zero sizes from tokens at execution."},
    "max_funding0": {**_S, "description": "Quote-time extra token0 budget; requires max_funding1 and no explicit caps."},
    "max_funding1": {**_S, "description": "Quote-time extra token1 budget; requires max_funding0 and no explicit caps."},
    **{f"amount{i}_desired": {**_S, "description": f"Maximum token{i} consumed, in integer base units; supply both caps."} for i in range(2)},
    **{f"amount{i}_min": {**_S, "description": f"Minimum token{i} consumed; generic/exact helpers default to the cap, flexible helpers to zero."} for i in range(2)},
    **{f"funded{i}": {**_S, "description": f"Fixed extra token{i} funding; flexible calls default to zero. Cannot combine with refund{i}."} for i in range(2)},
    **{f"refund{i}": {**_S, "description": f"Fixed immediate token{i} payout; flexible calls default to zero. Cannot combine with funded{i}."} for i in range(2)},
}


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
     "Plan moving a complete position to a new range (testnet router). "
     "Positive liquidity_target with omitted/equal token bounds enforces an exact quote; "
     "max_funding0/1 also size an exact quote. Wider token bounds use actual recovery; "
     "a zero target requires both desired caps and sizes liquidity at execution. "
     "Funding/refunds are fixed, flexible surplus stays collectible on the successor. "
     "Show the plan before executing; amounts are raw base units as strings. "
     "estimated_liquidity is advisory, never a guaranteed minimum. Flexible calls require an upgraded contract.",
     _schema(_REBALANCE_PROPERTIES, _REBALANCE_REQUIRED), _h_plan_rebalance),
    ("rebalance_position",
     "Replace a complete position atomically, keeping its owner and withdrawal address. "
     "Uses plan_rebalance arguments and rebuilds the plan before submitting. "
     "Exact calls can reject on price/fee changes; flexible calls use actual recovery "
     "within token bounds. Funding/refunds stay fixed; surplus stays collectible. "
     "A rejected transaction moves no funds but can cost a fee. Returns the successor, "
     "transaction ID, and submitted plan with amounts as base-unit strings.",
     _schema(_REBALANCE_PROPERTIES, _REBALANCE_REQUIRED), _h_rebalance_position),
]


_REBALANCE_MODES = {
    "exact_amounts": "Enforce exact quoted recovery and deposits. A positive target and equal bounds are required; price or fee changes can reject execution.",
    "to_liquidity": "Require exact successor liquidity using actual recovery. Supply a positive target and both token caps; minimums default to zero and at least one bound must be wider. Insufficient funds reject instead of minting less liquidity.",
    "from_amounts": "Size successor liquidity from actual recovered tokens plus fixed funding minus fixed refunds, capped per token. Supply both desired caps; target must be omitted or zero and minimums default to zero. Estimated liquidity is advisory; no swap occurs.",
}
for _mode, _description in _REBALANCE_MODES.items():
    _properties = dict(_REBALANCE_PROPERTIES)
    _required = list(_REBALANCE_REQUIRED)
    if _mode != "exact_amounts":
        _properties.pop("max_funding0")
        _properties.pop("max_funding1")
        _required.extend(["amount0_desired", "amount1_desired"])
        if _mode == "to_liquidity":
            _required.append("liquidity_target")
    for _prefix, _handler, _effect in (
        ("plan_rebalance", _h_plan_rebalance, "Read chain state without submitting. Show the plan before execution."),
        ("rebalance_position", _h_rebalance_position, "Rebuild the plan and submit atomically, keeping owner and withdrawal. Rejection rolls back funds but can cost a fee."),
    ):
        _TOOLS.append((f"{_prefix}_{_mode}",
                       f"{_description} {_effect} Token amounts use integer base-unit strings. "
                       "Fixed funding/refunds default to zero in flexible modes; surplus stays collectible on the successor. "
                       "Flexible calls require an upgraded contract; rebalance routing is testnet-only.",
                       _schema(dict(_properties), list(_required)), partial(_handler, mode=_mode)))


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
