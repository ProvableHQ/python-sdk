"""Semantic layer over the generated wire classes.

The generated ``_generated.py`` classes carry the wire shapes; the classes
here carry meaning: the persistable swap handle, typed verb results, and a
``Slot`` view with Q64 price math and range helpers.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Optional

from ._generated import Slot
from .tick_math import Q64, round_tick_to_spacing


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
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "SwapHandle":
        return cls(**json.loads(s))


@dataclass(frozen=True)
class ClaimResult:
    """Chain-computed amounts collected by a claim (raw atomic u128)."""

    transaction_id: str
    amount_out: int
    amount_remaining: int


@dataclass(frozen=True)
class MintResult:
    position_token_id: Optional[str]
    transaction_id: str


@dataclass(frozen=True)
class TxResult:
    """Result of a liquidity verb; ``position_token_id`` when the verb
    re-issues/identifies a position, else ``None``."""

    position_token_id: Optional[str]
    transaction_id: str


class SlotView:
    """A :class:`~aleo_shield_swap._generated.Slot` plus meaning.

    Delegates every field to the wrapped slot and adds the Q64 fixed-point
    conversions callers would otherwise re-derive subtly wrong.
    """

    def __init__(self, slot: Slot) -> None:
        self._slot = slot

    def __getattr__(self, name: str):
        return getattr(self._slot, name)

    def __repr__(self) -> str:
        return f"SlotView({self._slot!r})"

    @property
    def raw(self) -> Slot:
        return self._slot

    def price(self, decimals0: int, decimals1: int) -> Decimal:
        """Spot price of token1 per 1.0 token0, decimal-adjusted.

        ``sqrt_price`` encodes ``sqrt(token1_norm / token0_norm)`` in Q64
        over the contract's 9-decimal-normalized units; the human price
        re-applies ``10^(min(d0,9) - min(d1,9))``.
        """
        sqrt = Decimal(self._slot.sqrt_price) / Decimal(Q64)
        shift = Decimal(10) ** (min(decimals0, 9) - min(decimals1, 9))
        return sqrt * sqrt * shift

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
