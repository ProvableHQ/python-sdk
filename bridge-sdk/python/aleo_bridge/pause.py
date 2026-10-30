"""Operator pause switches, read live on both ends of a route before any funds move.

Every bridge deployment has a kill switch somewhere: the ``pause`` mapping on the Aleo warp and
USDCx programs, ``paused()`` on Circle's xReserve and CCTP transmitters, a ``PausableHook`` /
``PausableIsm`` inside the Ethereum Hyperlane Mailbox's hook and ISM trees, and the enrolled
remote-router table of the Solana warp program (Hyperlane's Sealevel token program has no pause
flag of its own; un-enrolling the router is how an operator stops it). ``quote``, ``execute``,
``resume`` and ``complete`` read every switch governing the route and refuse when one is set.

Fail closed: a switch that cannot be read after the retry budget below is ``unknown`` and blocks
exactly like ``paused``. Nothing here is cached — a pause can flip between quote and execute.

Assumption stated once: the deployed Solana warp programs match Hyperlane's published
``hyperlane-sealevel-token`` source at hyperlane-monorepo ``45c0988`` (the commit the rest of
``_sealevel.py`` derives from); they are BPF-upgradeable, so this cannot be checked on chain.
"""
from __future__ import annotations

import time
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")

IMMEDIATE_RETRIES = 2                       # retried at once, for a blip
BACKOFF_SECONDS = (0.5, 1.0, 2.0, 4.0)      # then exponential backoff; 7 attempts in all


class ReadFailed(Exception):
    """Internal: a switch read exhausted its retries. ``attempts`` and the ``last`` transient error."""

    def __init__(self, attempts: int, last: Exception) -> None:
        self.attempts, self.last = attempts, last
        super().__init__(f"{type(last).__name__}: {last} after {attempts} attempts")


def read_with_retry(fn: Callable[[int], T], *, sleep: Callable[[float], None] = time.sleep) -> T:
    """Call ``fn(attempt)`` until it returns: transient errors (``lifecycle._is_transient_error``) are
    retried ``IMMEDIATE_RETRIES`` times with no delay, then once per ``BACKOFF_SECONDS`` entry; anything
    else propagates at once. Raises :class:`ReadFailed` when the budget is spent."""
    from .lifecycle import _is_transient_error          # lifecycle imports this module; keep the cycle lazy

    delays = [0.0] * IMMEDIATE_RETRIES + list(BACKOFF_SECONDS)
    for attempt in range(len(delays) + 1):
        try:
            return fn(attempt)
        except Exception as exc:                        # noqa: BLE001 — classified right below
            if not _is_transient_error(exc):
                raise
            if attempt == len(delays):
                raise ReadFailed(attempt + 1, exc) from exc
            sleep(delays[attempt])
    raise AssertionError("unreachable")


def rotate(urls: Sequence[str], attempt: int) -> str:
    """The endpoint for *attempt*: walks a list of public RPC URLs so retries land on a different host."""
    return urls[attempt % len(urls)]


__all__ = ["BACKOFF_SECONDS", "IMMEDIATE_RETRIES", "ReadFailed", "read_with_retry", "rotate"]
