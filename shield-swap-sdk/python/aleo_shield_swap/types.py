"""Semantic layer over the generated wire classes.

The generated ``_generated.py`` classes carry the wire shapes; the classes
here carry meaning: the persistable swap handle, typed method results, and a
``Slot`` view with Q128.128 price math and range helpers.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Optional

from ._generated import HopExecution, Slot, SwapExecutionHeader, SwapExecutionKey
from .tick_math import Q128, round_tick_to_spacing, u256_to_int


@dataclass(frozen=True)
class SwapHandle:
    """The serializable thread between a private swap's two transactions.

    ``swap()`` returns it; ``claim_swap_output()`` consumes it.  Persist it
    (``to_json``) if the process might die before the claim —
    ``blinding_factor`` is the secret that proves ownership at claim time;
    treat it like a key.
    """

    swap_id: Optional[str]
    blinding_factor: Optional[str]
    blinded_address: Optional[str]
    token_in_id: str
    token_out_id: str
    pool_key: str
    amount_in: int
    transaction_id: str
    program: str

    def to_json(self) -> str:
        """Serialize the handle to JSON for storage or hand-off.

        Includes the blinding factor, so the output is secret-bearing — treat it
        like a key. Round-trips through :meth:`from_json`.
        """
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "SwapHandle":
        """Rebuild a handle from :meth:`to_json` output.

        Args:
            s: JSON produced by :meth:`to_json`.

        Returns:
            The restored handle, ready to claim.

        Raises:
            json.JSONDecodeError: If *s* is not valid JSON.
            TypeError: If the JSON omits a field or carries an unknown one — the
                handle is rebuilt strictly, so a partial object fails loudly
                rather than producing an unclaimable handle.
        """
        return cls(**json.loads(s))


@dataclass(frozen=True)
class ClaimResult:
    """Chain-computed amounts collected by a claim (raw atomic u128)."""

    transaction_id: str
    amount_out: int
    amount_remaining: int


@dataclass(frozen=True)
class MintResult:
    """Outcome of a mint: the new position's id and the transaction that made it.

    ``position_token_id`` is ``None`` when the transition published no id, which
    leaves the position discoverable only by a record scan.
    """

    position_token_id: Optional[str]
    transaction_id: str


@dataclass(frozen=True)
class TxResult:
    """Result of a liquidity method; ``position_token_id`` when the method
    re-issues/identifies a position, else ``None``."""

    position_token_id: Optional[str]
    transaction_id: str


class SlotView:
    """A :class:`~aleo_shield_swap._generated.Slot` plus meaning.

    Delegates every field to the wrapped slot and adds the Q128.128
    fixed-point conversions callers would otherwise re-derive subtly wrong.
    """

    def __init__(self, slot: Slot) -> None:
        self._slot = slot

    def __getattr__(self, name: str):
        return getattr(self._slot, name)

    def __repr__(self) -> str:
        return f"SlotView({self._slot!r})"

    @property
    def raw(self) -> Slot:
        """The wrapped slot (escape hatch), with no fixed-point interpretation.

        Prefer :meth:`price` and the other helpers — reading ``sqrt_price`` off
        this directly means re-deriving the Q128.128 conversion yourself.
        """
        return self._slot

    def price(self, decimals0: int, decimals1: int) -> Decimal:
        """Spot price of token1 per 1.0 token0, decimal-adjusted.

        ``sqrt_price`` encodes ``sqrt(token1_raw / token0_raw)`` in Q128.128
        over raw native units; the human price re-applies
        ``10^(decimals0 - decimals1)``.
        """
        sqrt = Decimal(u256_to_int(self._slot.sqrt_price)) / Decimal(Q128)
        return sqrt * sqrt * Decimal(10) ** (decimals0 - decimals1)

    def tick_range(self, width: int) -> tuple[int, int]:
        """A spacing-aligned mint range of ±*width* spacings around the
        current tick."""
        s = self._slot
        return (
            round_tick_to_spacing(s.tick - s.tick_spacing * width, s.tick_spacing),
            round_tick_to_spacing(s.tick + s.tick_spacing * width, s.tick_spacing),
        )


@dataclass
class StageOutcome:
    """One onboarding stage's result: ``action`` is ``"ran"`` or ``"skipped"``."""

    name: str
    action: str
    detail: str = ""


@dataclass
class OnboardReport:
    """What ``onboard()`` did, stage by stage, and whether funds are usable."""

    address: str
    outcomes: list[StageOutcome]
    funded: bool


@dataclass
class PositionView:
    """An open position: journaled, or discovered by scanning records."""

    position_token_id: Optional[str]
    pool_key: str
    source: str                     # "journal" | "scanned"


@dataclass(frozen=True)
class OwnedPositionState:
    """A position's chain-derived state — everything the mappings know.

    Read from ``positions``/``slots``/``ticks`` and the view math, so it moves
    with the pool price rather than being fixed at mint.
    """

    liquidity: int
    #: Token amounts currently backing the range, at the pool's live price.
    amount0: int
    amount1: int
    #: What ``collect`` would pay today: already-accrued ``tokens_owed`` plus
    #: fees earned since the position was last touched.
    collectible0: int
    collectible1: int
    tokens_owed0: int
    tokens_owed1: int


@dataclass(frozen=True)
class OwnedPosition:
    """A position this account owns: its record identity plus live chain state.

    A position spans two sources.  The private ``PositionNFT`` record carries
    identity — pool, range, withdrawal address — and no amounts; the public
    mappings carry amounts and no identity.  This is the join.

    *state* is ``None`` while a fresh mint is still finalizing: the record
    exists but ``positions[token_id]`` is not written yet.  Burned positions
    cannot appear at all, because burn consumes the record.
    """

    position_token_id: str
    pool_key: str
    tick_lower: int
    tick_upper: int
    token0_id: str
    token1_id: str
    withdrawal: str
    #: The spendable record plaintext — pass as ``position_record=`` to write verbs.
    record: str
    state: Optional[OwnedPositionState]


@dataclass(frozen=True)
class HopFill:
    """One pool leg of an executed swap, as the chain recorded it.

    ``fee_paid`` is gross and includes ``protocol_fee``; ``lp_fee`` is the
    difference — what the pool's liquidity providers earned from this leg.
    ``sqrt_price_after`` is the Q128.128 price as an integer.
    """

    pool: str
    zero_for_one: bool
    amount_in: int
    amount_out: int
    fee_paid: int
    protocol_fee: int
    lp_fee: int
    sqrt_price_after: int
    liquidity_after: int
    tick_after: int

    @classmethod
    def from_execution(cls, hop: HopExecution) -> "HopFill":
        return cls(pool=hop.pool, zero_for_one=hop.zero_for_one,
                   amount_in=hop.amount_in, amount_out=hop.amount_out,
                   fee_paid=hop.fee_paid, protocol_fee=hop.protocol_fee,
                   lp_fee=hop.fee_paid - hop.protocol_fee,
                   sqrt_price_after=u256_to_int(hop.sqrt_price_after),
                   liquidity_after=hop.liquidity_after, tick_after=hop.tick_after)


@dataclass(frozen=True)
class SwapExecution:
    """The chain's fill receipt for one swap: when it executed and every hop.

    Written at finalize into ``swap_execution_headers`` /
    ``swap_execution_hops``; a single-pool swap has one hop, a multi-hop swap
    up to three.  Unlike ``swap_outputs`` this is never consumed by a claim, so
    it stays readable as history.
    """

    swap_id: str
    executed_height: int
    hops: list[HopFill]

    @staticmethod
    def hop_key(swap_id: str, hop_index: int) -> str:
        """The ``swap_execution_hops`` mapping key for one leg (a struct literal)."""
        return SwapExecutionKey(swap_id=swap_id, hop_index=hop_index).to_plaintext()

    @classmethod
    def from_plaintexts(cls, swap_id: str, header_text: str,
                        hop_texts: list[Optional[str]]) -> "SwapExecution":
        header = SwapExecutionHeader.from_plaintext(header_text)
        hops: list[HopFill] = []
        for i, text in enumerate(hop_texts):
            if text is None:
                raise ValueError(
                    f"swap {swap_id}: header records {header.hop_count} hops but "
                    f"hop {i} is missing from swap_execution_hops — the node is "
                    "behind or the read raced a finalize; retry.")
            hops.append(HopFill.from_execution(HopExecution.from_plaintext(text)))
        return cls(swap_id=swap_id, executed_height=header.executed_height, hops=hops)


@dataclass
class SessionStatus:
    """Everything an agent needs to re-orient in one call."""

    address: str
    network: str
    authenticated: bool
    has_access: Optional[bool]      # None when not authenticated / unreachable
    balances: dict
    pending_claim_ids: list[str]
    open_positions: list[PositionView]
    counter_cursor: int


@dataclass
class SwapBatchReport:
    """``swap_many()`` outcome — journaled handles plus per-counter failures."""

    handles: list[SwapHandle]
    failures: list[dict]


@dataclass
class CollectReport:
    """``collect_all()`` outcome — what was claimed, what is not ready yet."""

    claimed: list[dict]
    still_pending: list[str]
    fees: list[dict]
