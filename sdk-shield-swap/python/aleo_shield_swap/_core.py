"""Shared pure logic for the sync and async clients.

Param resolution, deadlines, nonces, dynamic-dispatch import resolution, and
token-record selection — everything both ``client.py`` and
``async_client.py`` need but that does no client-specific I/O itself.
``resolve_swap_params`` is a line-for-line port of the TS SDK's
``utils/params.ts``.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any, Optional

from aleo.codegen.runtime import parse_plaintext

from .errors import InsufficientRecordsError
from .tick_math import MAX_SQRT_PRICE, MIN_SQRT_PRICE, Q64


@dataclass(frozen=True)
class ResolvedSwap:
    """A friendly swap intent resolved into the contract's raw arguments."""

    zero_for_one: bool
    token_out_id: str
    amount_out_min: int
    sqrt_price_limit: int


def resolve_swap_params(
    *,
    pool: Any,
    slot: Any,
    token_in_id: str,
    amount_in: int,
    slippage_bps: int,
    expected_out: Optional[int] = None,
    sqrt_price_limit: Optional[int] = None,
) -> ResolvedSwap:
    """Resolve a swap intent against live pool state.

    Determines direction from the pool's token ordering, validates the
    amount against the contract's no-dust rule, and computes
    ``amount_out_min`` from the slippage tolerance.  Without *expected_out*
    a spot estimate from ``slot.sqrt_price`` is used — it ignores price
    impact and fees, so pass a real quote for anything beyond a tiny trade.
    Pure and local.
    """
    if not 0 <= slippage_bps <= 10_000:
        raise ValueError(f"slippage_bps must be within [0, 10000], got {slippage_bps}")

    token0, token1 = str(pool.token0), str(pool.token1)
    scale0, scale1 = int(pool.scale0), int(pool.scale1)

    zero_for_one = token_in_id == token0
    if not zero_for_one and token_in_id != token1:
        raise ValueError(f"Token {token_in_id} is not in this pool ({token0} / {token1})")
    token_out_id = token1 if zero_for_one else token0

    # The contract normalizes amounts by the token's scale and asserts
    # raw % scale == 0 — reject dust here instead of paying for a revert.
    scale_in = scale0 if zero_for_one else scale1
    if amount_in % scale_in != 0:
        raise ValueError(
            f"amount_in {amount_in} is not a multiple of the token's scale "
            f"{scale_in} — the contract rejects amounts with non-zero dust digits"
        )

    expected = expected_out
    if expected is None:
        # Spot estimate in normalized units: price = (sqrtP/Q64)^2 token1/token0.
        scale_out = scale1 if zero_for_one else scale0
        norm_in = amount_in // scale_in
        sq = int(slot.sqrt_price)
        if zero_for_one:
            norm_out = (norm_in * sq * sq) // (Q64 * Q64)
        else:
            norm_out = (norm_in * Q64 * Q64) // (sq * sq)
        expected = norm_out * scale_out

    amount_out_min = (expected * (10_000 - slippage_bps)) // 10_000

    # Default price bound: the directional extreme — amount_out_min is the
    # real protection; a tight sqrt limit turns into partial fills instead.
    default_limit = MIN_SQRT_PRICE if zero_for_one else MAX_SQRT_PRICE
    limit = sqrt_price_limit if sqrt_price_limit is not None else default_limit
    if not MIN_SQRT_PRICE <= limit <= MAX_SQRT_PRICE:
        raise ValueError(
            f"sqrt_price_limit {limit} outside the contract's accepted range "
            f"[{MIN_SQRT_PRICE}, {MAX_SQRT_PRICE}]"
        )

    return ResolvedSwap(zero_for_one, token_out_id, amount_out_min, limit)


def get_deadline(aleo: Any, offset_blocks: int = 100) -> int:
    """Absolute block-height deadline: current height + *offset_blocks*.

    The contract's ``deadline`` is a height (u32), not a timestamp; the
    finalize asserts the current height is below it.
    """
    return int(aleo.network_client.get_latest_height()) + offset_blocks


def generate_swap_nonce() -> int:
    """Uniform random u64 — uniquifies the swap id in ``swap_outputs``."""
    return secrets.randbits(64)


def generate_field_nonce() -> str:
    """Random field literal for ``mint`` (hashed into the position id).
    248 bits keeps the value below the field modulus."""
    return f"{secrets.randbits(248)}field"


# ── Dynamic-dispatch imports ─────────────────────────────────────────────────

_IMPORTS_CACHE: dict[tuple[str, str], str] = {}


def resolve_imports(
    aleo: Any,
    program_ids: list[str],
    overrides: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """Program sources for dynamic-dispatch dependencies.

    ``shield_swap`` calls token programs through a dynamic interface, so the
    prover cannot discover them statically — every record-spending write
    needs the involved token programs' sources.  Sources are fetched via the
    bound client once and memoized per (network, program); *overrides* win
    without fetching.
    """
    out: dict[str, str] = {}
    # Key by the provider (not just the network name): a devnode and testnet
    # both report "testnet" but serve different program deployments.
    scope = repr(getattr(aleo, "provider", None) or getattr(aleo, "network_name", ""))
    for pid in dict.fromkeys(program_ids):  # de-dup, keep order
        if overrides and pid in overrides:
            out[pid] = overrides[pid]
            continue
        key = (scope, pid)
        if key not in _IMPORTS_CACHE:
            _IMPORTS_CACHE[key] = str(aleo.programs.get(pid).source)
        out[pid] = _IMPORTS_CACHE[key]
    return out


# ── Token record selection ───────────────────────────────────────────────────

def parse_token_record_info(plaintext: str) -> Optional[dict[str, Any]]:
    """Decode a token record's ``amount`` (and ``token_id`` when present).

    Handles both registry-token records (``owner``, ``amount``, ``token_id``,
    …) and ARC-20 wrapper-program records (``owner``, ``amount`` only).
    Returns ``None`` when the plaintext has no ``amount`` — not a token record.
    """
    try:
        decoded = parse_plaintext(plaintext)
    except (ValueError, TypeError):
        return None
    if not isinstance(decoded, dict) or not isinstance(decoded.get("amount"), int):
        return None
    info: dict[str, Any] = {"amount": decoded["amount"]}
    if isinstance(decoded.get("token_id"), str):
        info["token_id"] = decoded["token_id"]
    return info


def select_token_record(
    aleo: Any,
    *,
    program: str,
    min_amount: int,
    token_id: Optional[str] = None,
    account: Any = None,
) -> str:
    """One unspent record plaintext from *program* covering *min_amount*.

    Scans via ``aleo.record_provider.find`` and picks the smallest covering
    record (leaves larger records intact for larger trades).  *token_id*
    filters registry-style records; wrapper-program records carry no
    ``token_id`` and match any.
    """
    provider = aleo.record_provider
    if provider is None:
        raise InsufficientRecordsError(
            "No record provider configured (aleo.record_provider is None) — "
            "pass token_record= explicitly or configure a scanner."
        )
    records = provider.find(account, program=program, unspent=True)
    candidates: list[tuple[int, str]] = []
    for rec in records:
        plaintext = rec.get("record_plaintext") if isinstance(rec, dict) else getattr(rec, "record_plaintext", None)
        if not plaintext:
            continue
        info = parse_token_record_info(plaintext)
        if info is None or info["amount"] < min_amount:
            continue
        if token_id is not None and "token_id" in info and info["token_id"] != token_id:
            continue
        candidates.append((info["amount"], plaintext))
    if not candidates:
        raise InsufficientRecordsError(
            f"No unspent {program} record covers {min_amount} "
            f"(token_id={token_id or 'any'}) — privatize funds or pass token_record=."
        )
    return min(candidates)[1]
