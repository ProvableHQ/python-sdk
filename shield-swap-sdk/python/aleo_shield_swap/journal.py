"""Append-only participant journal — swaps, positions, counters, stages.

One JSONL file per profile.  State (pending claims, open positions, the
counter cursor) is always derived by replaying events, so a crash between
append and action never corrupts anything; the worst case is an event whose
action never happened, which downstream verbs tolerate (a claim of a swap
that never landed just reports not-finalized).

Counter reservation is the concurrency-critical piece: blinded identities
must never collide, so counters are issued once, under an advisory file
lock, and burned (never reused) when their swap fails.
"""
from __future__ import annotations

import fcntl
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .types import SwapHandle

_HANDLE_FIELDS = ("swap_id", "blinding_factor", "blinded_address",
                  "token_in_id", "token_out_id", "pool_key", "amount_in",
                  "transaction_id", "program")


class Journal:
    """Event log at *path* (created on first append)."""

    def __init__(self, path: "Path | str") -> None:
        self.path = Path(path)
        self._lock_path = self.path.with_suffix(".lock")

    def __repr__(self) -> str:
        return f"Journal({str(self.path)!r})"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Advisory lock shared by every writer (and the counter reader)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    # ── Raw events ───────────────────────────────────────────────────────────

    def append(self, type: str, **fields: Any) -> None:
        event = {"type": type, "ts": time.time(), **fields}
        with self._locked():
            with self.path.open("a") as f:
                f.write(json.dumps(event) + "\n")

    def events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line)
                for line in self.path.read_text().splitlines() if line]

    # ── Counters ─────────────────────────────────────────────────────────────

    def reserve_counters(self, n: int) -> list[int]:
        """Issue the next *n* counters, exactly once, under a file lock."""
        if n <= 0:
            return []
        with self._locked():
            start = self.counter_cursor()
            counters = list(range(start, start + n))
            event = {"type": "counters_reserved", "ts": time.time(),
                     "counters": counters}
            with self.path.open("a") as f:      # already under the lock
                f.write(json.dumps(event) + "\n")
            return counters

    def counter_cursor(self) -> int:
        """Next unissued counter (max seen in any event + 1)."""
        top = -1
        for e in self.events():
            if e["type"] == "counters_reserved":
                top = max(top, *e.get("counters") or [-1])
            elif e["type"] in ("swap", "swap_failed"):
                top = max(top, e.get("counter", -1))
        return top + 1

    # ── Typed events ─────────────────────────────────────────────────────────

    def record_swap(self, handle: SwapHandle, counter: int) -> None:
        self.append("swap", counter=counter,
                    **{k: getattr(handle, k) for k in _HANDLE_FIELDS})

    def record_swap_failed(self, counter: int, error: str) -> None:
        self.append("swap_failed", counter=counter, error=error)

    def record_claim(self, swap_id: str, transaction_id: str,
                     amount_out: int) -> None:
        self.append("claim", swap_id=swap_id, transaction_id=transaction_id,
                    amount_out=amount_out)

    def record_position(self, position_token_id: str, pool_key: str,
                        transaction_id: str) -> None:
        self.append("position", position_token_id=position_token_id,
                    pool_key=pool_key, transaction_id=transaction_id)

    def record_position_burned(self, position_token_id: str,
                               transaction_id: str) -> None:
        self.append("position_burned", position_token_id=position_token_id,
                    transaction_id=transaction_id)

    def record_stage(self, name: str, action: str, detail: str = "") -> None:
        self.append("stage", name=name, action=action, detail=detail)

    # ── Derived state ────────────────────────────────────────────────────────

    def pending_claims(self) -> list[SwapHandle]:
        """Claimable swaps: recorded with a swap id and never claimed."""
        events = self.events()
        claimed = {e["swap_id"] for e in events if e["type"] == "claim"}
        out: list[SwapHandle] = []
        for e in events:
            if e["type"] != "swap" or not e.get("swap_id"):
                continue                  # id-less swaps need manual recovery
            if e["swap_id"] in claimed:
                continue
            if not all(k in e for k in _HANDLE_FIELDS):
                continue                  # legacy/malformed event — skip
            out.append(SwapHandle(**{k: e[k] for k in _HANDLE_FIELDS}))
        return out

    def open_positions(self) -> list[dict[str, Any]]:
        """Positions recorded and not burned: {position_token_id, pool_key}."""
        events = self.events()
        burned = {e["position_token_id"] for e in events
                  if e["type"] == "position_burned"}
        return [{"position_token_id": e["position_token_id"],
                 "pool_key": e["pool_key"]}
                for e in events
                if e["type"] == "position" and e["position_token_id"] not in burned]
