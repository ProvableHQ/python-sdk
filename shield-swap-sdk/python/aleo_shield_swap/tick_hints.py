"""Insert-hint selection for position ticks (port of utils/tick-hints.ts).

The contract keeps initialized ticks in a sorted linked list and asserts
``hint.tick < target && hint.next > target`` — the hint must be the target's
predecessor.  This derives the hint from the slot's active-range neighbors,
which covers pools with few initialized ticks around the current price.

Known limitation (inherited from the reference client): when multiple
initialized ticks lie between the slot's neighbors and the target, this does
not walk the list to the true predecessor; a wrong hint reverts on the
contract's assert.  An exact walk is possible with ``derive_tick_key`` —
follow-up.
"""
from __future__ import annotations

from typing import Any

from .tick_math import MIN_TICK


def pick_insert_hint(slot: Any, target_tick: int) -> int:
    """Presumed predecessor of *target_tick*, from the slot's neighbors."""
    if slot is None:
        return MIN_TICK
    if target_tick > slot.tick:
        if slot.next_init_above < target_tick:
            return int(slot.next_init_above)
        return int(slot.next_init_below)
    return int(slot.next_init_below)
