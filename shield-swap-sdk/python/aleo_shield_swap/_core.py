"""Shared pure logic for the sync and async clients.

Param resolution, deadlines, nonces, dynamic-dispatch import resolution, and
token-record selection — everything both ``client.py`` and
``async_client.py`` need but that does no client-specific I/O itself.
``resolve_swap_params`` is a line-for-line port of the TS SDK's
``utils/params.ts``.
"""
from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from typing import Any, Optional

from aleo.codegen.runtime import parse_plaintext

from .errors import InsufficientRecordsError
from .tick_math import MAX_SQRT_RATIO_X128, MIN_SQRT_RATIO_X128, u256_to_int


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

    Amounts are raw native token units end to end — the new AMM does no
    decimal scaling and has no dust rule.  Without *expected_out* a spot
    estimate from the Q128.128 ``slot.sqrt_price`` is used — it ignores
    price impact and fees, so pass a real quote for anything beyond a tiny
    trade.  Pure and local.
    """
    if not 0 <= slippage_bps <= 10_000:
        raise ValueError(f"slippage_bps must be within [0, 10000], got {slippage_bps}")

    token0, token1 = str(pool.token0), str(pool.token1)
    zero_for_one = token_in_id == token0
    if not zero_for_one and token_in_id != token1:
        raise ValueError(f"Token {token_in_id} is not in this pool ({token0} / {token1})")
    token_out_id = token1 if zero_for_one else token0

    expected = expected_out
    if expected is None:
        # Spot estimate: price = (sqrt_price / 2^128)^2 token1-per-token0.
        sq = u256_to_int(slot.sqrt_price)
        if zero_for_one:
            expected = (amount_in * sq * sq) >> 256
        else:
            expected = (amount_in << 256) // (sq * sq)

    amount_out_min = (expected * (10_000 - slippage_bps)) // 10_000

    # Default price bound: the directional extreme — amount_out_min is the
    # real protection; a tight sqrt limit turns into partial fills instead.
    default_limit = MIN_SQRT_RATIO_X128 if zero_for_one else MAX_SQRT_RATIO_X128
    limit = sqrt_price_limit if sqrt_price_limit is not None else default_limit
    if not MIN_SQRT_RATIO_X128 <= limit <= MAX_SQRT_RATIO_X128:
        raise ValueError(
            f"sqrt_price_limit {limit} outside the contract's accepted range "
            f"[{MIN_SQRT_RATIO_X128}, {MAX_SQRT_RATIO_X128}]"
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


# ── Freezelist proofs ────────────────────────────────────────────────────────
#
# mint / claim_swap_output / collect prove signer (and recipient/withdrawal)
# non-inclusion in the AMM freezelist; routed calls additionally carry
# wrapper-freezelist proofs.  While a freezelist is empty, the contract
# accepts two copies of the empty-tree proof below (the credits wrapper
# ignores its proof input entirely).  Building real proofs against a
# populated tree is deferred until a list is non-empty.

EMPTY_MERKLE_PROOF = "{ siblings: [" + ", ".join(["0field"] * 16) + "], leaf_index: 1u32 }"


def default_merkle_proofs() -> str:
    """The ``[MerkleProof; 2]`` literal accepted while the freezelist is empty."""
    return f"[{EMPTY_MERKLE_PROOF}, {EMPTY_MERKLE_PROOF}]"


# ── Shared pure helpers (sync + async clients) ───────────────────────────────

def normalize_mapping_value(raw: Any) -> Optional[str]:
    """A mapping read normalized to plaintext, or ``None`` when absent.

    Node deployments variously return ``None``, ``"null"``, the empty string,
    or a JSON-quoted value — one place decides what "absent" means.
    """
    if raw in (None, "", "null"):
        return None
    text = str(raw).strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    return None if text in ("", "null") else text


def mapping_flag_set(raw: Any) -> bool:
    """A presence/boolean mapping read (``used_blinded_addresses``,
    ``initialized_pools``) as one truth: set when an entry exists and is not
    the literal ``false``.  Every probe of such a mapping — sync, async, and
    the journal-free derivation path — must share this, or the three disagree
    on whether a counter is free."""
    text = normalize_mapping_value(raw)
    return text is not None and text != "false"


def read_mapping_value(aleo: Any, program: str, mapping: str, key: str) -> Optional[str]:
    """One mapping entry via the node's mapping endpoint — no program handle.

    ``aleo.programs.get(program)`` downloads and parses the deployed source
    (1.4 MB for the core) just to hand back a ``Mapping`` whose ``get`` only
    needs ``(program, mapping, key)``; probes that run per swap or per token
    go straight to the network client instead.
    """
    return normalize_mapping_value(
        aleo.network.get_program_mapping_value(program, mapping, key))


_UNSIGNED_LITERAL = re.compile(r"^(\d+)u(?:8|16|32|64|128)$")


def parse_unsigned_literal(raw: Optional[str], program: str, key: str) -> int:
    """An ARC-20 ``balances`` entry (``"5u128"``) as an int; ``None`` reads as 0.

    Raises:
        ValueError: If *raw* is not an unsigned-integer literal — the mapping
            is not ARC-20 shaped, which is a caller error worth surfacing
            rather than a zero balance.
    """
    if raw is None:
        return 0
    match = _UNSIGNED_LITERAL.match(raw.strip())
    if not match:
        raise ValueError(f"{program} balances[{key}] is not an unsigned "
                         f"integer literal: {raw!r}")
    return int(match.group(1))


def pick_covering_record(records: Any, *, min_amount: int,
                         token_id: Optional[str],
                         exclude: Optional[set[str]] = None) -> Optional[str]:
    """Smallest unspent token-record plaintext covering *min_amount*, or None.

    *token_id* filters registry-style records; wrapper-program records carry
    no ``token_id`` and match any.  *exclude* skips records already assigned
    (e.g. to earlier swaps of a concurrent batch).
    """
    candidates: list[tuple[int, str]] = []
    for rec in records:
        plaintext = record_plaintext(rec)
        if not plaintext or (exclude and plaintext in exclude):
            continue
        info = parse_token_record_info(plaintext)
        if info is None or info["amount"] < min_amount:
            continue
        if info["recipient_bound"]:
            continue          # bound wrapper records unwrap only to their bound recipient
        if token_id is not None and "token_id" in info and info["token_id"] != token_id:
            continue
        candidates.append((info["amount"], plaintext))
    return min(candidates)[1] if candidates else None


def record_plaintext(rec: Any) -> Optional[str]:
    """The decrypted plaintext of a provider record, dict- or object-shaped."""
    if isinstance(rec, dict):
        return rec.get("record_plaintext")
    return getattr(rec, "record_plaintext", None)


#: Fields a PositionNFT record carries.  Checked as a set, so a future record
#: type sharing one of them is not mistaken for a position.
POSITION_RECORD_FIELDS = ("token_id", "pool", "tick_lower", "tick_upper",
                          "token0_id", "token1_id", "withdrawal")


def decode_position_record(plaintext: str) -> Optional[dict[str, Any]]:
    """A PositionNFT record's fields, or None if *plaintext* is not one.

    Shared by both clients so the record shape is defined once.  Returns None
    rather than raising for a record of any other type, letting a mixed record
    set be filtered in one pass.
    """
    try:
        decoded = parse_plaintext(plaintext)
    except (ValueError, TypeError):
        return None
    if not isinstance(decoded, dict):
        return None
    return decoded if all(f in decoded for f in POSITION_RECORD_FIELDS) else None


def find_position_plaintext(records: Any, pool_key: str,
                            position_token_id: Optional[str] = None) -> Optional[str]:
    """The unspent PositionNFT plaintext for *pool_key*, or None.

    With *position_token_id* the record must carry that token id — an account
    holding several positions in one pool must not have a write verb pair a
    plan for one position with another position's NFT (a guaranteed revert
    after the proof is paid for).  Without it, the first position in the pool.

    Uses :func:`decode_position_record`, so a record of another type that
    happens to carry a matching ``pool`` field is not returned as a position.
    """
    for rec in records:
        plaintext = record_plaintext(rec)
        if not plaintext:
            continue
        decoded = decode_position_record(plaintext)
        if decoded is None or decoded.get("pool") != pool_key:
            continue
        if position_token_id is not None and str(decoded.get("token_id")) != str(position_token_id):
            continue
        return plaintext
    return None


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


_IMPORT_LINE = re.compile(r"^import\s+(\S+?);\s*$", re.MULTILINE)


def program_imports(source: str) -> list[str]:
    """Program ids named by ``import X.aleo;`` lines in *source*."""
    return _IMPORT_LINE.findall(source)


def register_program_sources(aleo: Any, sources: dict[str, str]) -> None:
    """Add *sources* to the bound client's snarkVM process, dependencies
    first (by declared ``import`` lines, within the provided closure).

    ``Process.load()`` seeds only ``credits.aleo``; every other program a
    write touches — the DEX program itself and the dynamically-dispatched
    token wrapper programs — must be added before ``authorize`` can resolve
    the call.  Idempotent: already-registered programs are skipped.
    """
    process = aleo.process
    if aleo.network_name == "testnet":
        from aleo import testnet as net
    else:
        from aleo import mainnet as net

    added: set[str] = set()

    def add(pid: str) -> None:
        """Register *pid* and its imports, dependencies first.

        Skips ``credits.aleo`` (already seeded), programs handled in this pass,
        anything absent from *sources*, and programs the process already holds —
        so repeated calls are cheap and safe.
        """
        if pid == "credits.aleo" or pid in added or pid not in sources:
            return
        added.add(pid)
        if process.contains_program(net.ProgramID.from_string(pid)):
            return
        for dep in program_imports(sources[pid]):
            add(dep)                     # dependencies before dependents
        process.add_program(net.Program.from_source(sources[pid]))

    for pid in sources:
        add(pid)


def ensure_programs(aleo: Any, program_ids: list[str],
                    overrides: Optional[dict[str, str]] = None) -> None:
    """Fetch the import closure of *program_ids* (sync) and register it.

    Sources come from *overrides* first, else via ``aleo.programs.get``
    (memoized by :func:`resolve_imports`).
    """
    sources: dict[str, str] = {}
    stack = [pid for pid in dict.fromkeys(program_ids) if pid != "credits.aleo"]
    while stack:
        pid = stack.pop()
        if pid in sources or pid == "credits.aleo":
            continue
        sources[pid] = resolve_imports(aleo, [pid], overrides)[pid]
        stack.extend(program_imports(sources[pid]))
    register_program_sources(aleo, sources)


# ── Token record selection ───────────────────────────────────────────────────

def parse_token_record_info(plaintext: str) -> Optional[dict[str, Any]]:
    """Decode a token record's spendable amount (and ``token_id`` when present).

    Handles registry-token records (``owner, amount, token_id, …``), ARC-20
    wrapper/underlying records (``owner, amount``), and native credits
    records (``owner, microcredits``).  Recipient-bound wrapper records are
    flagged — they can only be unwrapped to their bound recipient, never
    spent freely.  Returns ``None`` when neither amount field is present.
    """
    try:
        decoded = parse_plaintext(plaintext)
    except (ValueError, TypeError):
        return None
    if not isinstance(decoded, dict):
        return None
    amount = decoded.get("amount")
    if not isinstance(amount, int):
        amount = decoded.get("microcredits")
    if not isinstance(amount, int):
        return None
    info: dict[str, Any] = {"amount": amount,
                            "recipient_bound": decoded.get("recipient_bound") is True}
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
    chosen = pick_covering_record(records, min_amount=min_amount, token_id=token_id)
    if chosen is None:
        raise InsufficientRecordsError(
            f"No unspent {program} record covers {min_amount} "
            f"(token_id={token_id or 'any'}) — privatize funds or pass token_record=."
        )
    return chosen
