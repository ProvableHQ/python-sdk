"""Append-only participant journal — swaps, positions, counters, stages.

One JSONL file per profile.  State (pending claims, open positions, the
counter cursor) is always derived by replaying events, so a crash between
append and action never corrupts anything; the worst case is an event whose
action never happened, which downstream methods tolerate (a claim of a swap
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
    """A participant's durable record of swaps, positions, and counters.

    Backs the JSONL file at *path* (created on first append).  It holds the two
    things the chain cannot give back:

    * every swap's ``blinding_factor`` — the secret
      :meth:`~aleo_shield_swap.client.ShieldSwap.claim_swap_output` proves
      knowledge of.  A swap whose handle is lost cannot be claimed by
      anyone, which is the point of blinding it.
    * which blinding counters this account has spent, so no two swaps derive
      the same blinded address.  :meth:`reserve_counters` issues each one once
      under a file lock, and a failed swap burns its counter rather than
      recycling it.

    Nothing is stored as state — every event is appended and the views are
    replayed from the log, so a crash between append and action leaves an event
    whose action never happened rather than a corrupt file.  Read it through
    :meth:`pending_claims`, :meth:`open_positions`, and :meth:`counter_cursor`
    rather than :meth:`events`.

    ``ShieldSwap.from_profile()`` wires one up per profile; ``swap_many`` and
    ``collect_all`` require it and raise without one.  Treat the file as key
    material — it carries every blinding factor the account has used.

    Args:
        path: Where the log lives, usually ``Profile.journal_path``.  Its
            parent is created on first append, alongside a ``.lock`` sibling
            used to serialize writers.
    """

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
        """Write one event to the log, stamped and under the writer lock.

        Creates the file (and its parent) on first use. Fields are JSON-encoded
        as given, so pass only JSON-safe values. Appending is not the action —
        an event whose action never happened is expected and tolerated on replay.

        Args:
            type: Event type, matched by name when state is derived.
            **fields: Event payload; a ``ts`` wall-clock stamp is added.
        """
        event = {"type": type, "ts": time.time(), **fields}
        with self._locked():
            with self.path.open("a") as f:
                f.write(json.dumps(event) + "\n")

    def events(self) -> list[dict[str, Any]]:
        """Replay the whole log in append order.

        Reads without taking the lock, so a concurrent append may or may not be
        included. An absent file reads as no events rather than raising.

        Returns:
            Every event recorded so far.

        Raises:
            json.JSONDecodeError: If a line is not valid JSON — a truncated write
                from a crash mid-append.
        """
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

    def skip_counters_through(self, counter: int) -> None:
        """Retire every counter up to and including *counter* without issuing it.

        A fresh journal for an account that already swapped starts at 0, but
        those counters are consumed on chain — a swap built on one is rejected
        at finalize.  The client seeds the cursor past the used run with this;
        it never moves the cursor backwards.
        """
        with self._locked():
            if counter < self.counter_cursor():
                return
            event = {"type": "counters_skipped", "ts": time.time(), "through": counter}
            with self.path.open("a") as f:      # already under the lock
                f.write(json.dumps(event) + "\n")

    def counter_cursor(self) -> int:
        """Next unissued counter (max seen in any event + 1)."""
        top = -1
        for e in self.events():
            if e["type"] == "counters_reserved":
                top = max(top, *e.get("counters") or [-1])
            elif e["type"] == "counters_skipped":
                top = max(top, e.get("through", -1))
            elif e["type"] in ("swap", "swap_failed"):
                top = max(top, e.get("counter", -1))
        return top + 1

    # ── Typed events ─────────────────────────────────────────────────────────

    def record_swap(self, handle: SwapHandle, counter: int) -> None:
        """Log a submitted swap so a later session can claim it.

        Persists the handle's blinding factor — the secret needed to claim — so
        the journal file is sensitive. A handle with no ``swap_id`` is still
        recorded, but :meth:`pending_claims` skips it as needing manual recovery.

        Args:
            handle: The swap handle to persist.
            counter: The counter this swap's blinded identity consumed, so it is
                never reissued.
        """
        self.append("swap", counter=counter,
                    **{k: getattr(handle, k) for k in _HANDLE_FIELDS})

    def record_swap_failed(self, counter: int, error: str) -> None:
        """Burn a counter whose swap never landed.

        The counter stays spent — reusing it would risk a blinded-identity
        collision — so this advances the cursor without producing a claimable
        swap.

        Args:
            counter: The counter to retire.
            error: Why the swap failed, kept for diagnosis.
        """
        self.append("swap_failed", counter=counter, error=error)

    def record_claim(self, swap_id: str, transaction_id: str,
                     amount_out: int) -> None:
        """Mark a swap claimed, dropping it from :meth:`pending_claims`.

        Args:
            swap_id: The claimed swap.
            transaction_id: Transaction that settled the claim.
            amount_out: Amount received, in base units.
        """
        self.append("claim", swap_id=swap_id, transaction_id=transaction_id,
                    amount_out=amount_out)

    def record_position(self, position_token_id: str, pool_key: str,
                        transaction_id: str) -> None:
        """Log a minted position so it is found without a record scan.

        Args:
            position_token_id: The position's token id.
            pool_key: Pool the position belongs to.
            transaction_id: Transaction that minted it.
        """
        self.append("position", position_token_id=position_token_id,
                    pool_key=pool_key, transaction_id=transaction_id)

    def record_position_burned(self, position_token_id: str,
                               transaction_id: str) -> None:
        """Mark a position closed, dropping it from :meth:`open_positions`.

        Args:
            position_token_id: The burned position's token id.
            transaction_id: Transaction that burned it.
        """
        self.append("position_burned", position_token_id=position_token_id,
                    transaction_id=transaction_id)

    def record_stage(self, name: str, action: str, detail: str = "") -> None:
        """Log progress through a registration stage, for resumable onboarding.

        Args:
            name: Stage name.
            action: What happened at that stage.
            detail: Optional extra context.
        """
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
