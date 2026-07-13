"""Shield-swap error taxonomy — every package error is a ShieldSwapError.

The base subclasses the facade's :class:`~aleo.AleoError`, so ``except
AleoError`` catches everything from both layers.  Facade errors
(``AleoProvingError``, ``TransactionConfirmationTimeout``, …) pass through
untranslated.
"""
from __future__ import annotations

from aleo import AleoError


class ShieldSwapError(AleoError):
    """Base class for all aleo-shield-swap errors."""


class SwapOutputNotFinalizedError(ShieldSwapError):
    """``swap_outputs[swap_id]`` is empty — the request has not finalized yet
    (retry shortly), or the output was already claimed (a claim consumes the
    entry, so a second claim of the same handle sees the same absence)."""

    def __init__(self, swap_id: str) -> None:
        super().__init__(
            f"swap_outputs[{swap_id}] is empty — the request transaction has "
            "not finalized yet (retry shortly), or this output was already claimed."
        )
        self.swap_id = swap_id


class PoolNotFoundError(ShieldSwapError):
    def __init__(self, pool_key: str) -> None:
        super().__init__(f"Pool {pool_key} does not exist on-chain.")
        self.pool_key = pool_key


class PoolNotInitializedError(ShieldSwapError):
    def __init__(self, pool_key: str) -> None:
        super().__init__(f"Pool {pool_key} exists but is not initialized.")
        self.pool_key = pool_key


class InsufficientRecordsError(ShieldSwapError):
    """The record provider found no unspent record covering the amount."""


class InvalidFeeTierError(ShieldSwapError):
    """The fee tier is not registered on-chain (checked before submission)."""


class DexApiError(ShieldSwapError):
    """A DEX REST API request failed; carries the HTTP status and body."""

    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"DEX API error {status}: {body[:200]}")
        self.status = status
        self.body = body
