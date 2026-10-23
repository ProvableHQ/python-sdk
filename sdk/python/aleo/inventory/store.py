"""Journal stores for record reservations.

The SDK ships one store, :class:`MemoryRecordInventoryStore`, for a single
process. Anything that satisfies :class:`~aleo.inventory.types.RecordInventoryStore`
can replace it: a database shared by several processes, a key-value service,
or a file. A journal holds record nonces, program and function names, and
proved transactions; it never holds keys or record plaintext.
"""
from __future__ import annotations

import copy
import threading

from .types import RecordInventoryStore, RecordSpend, RecordSpendPatch


def holds_records(entry: RecordSpend) -> bool:
    """Return whether a journal entry still excludes any record from selection.

    Confirmed entries keep excluding their inputs because a scanner can report
    a spent record as unspent for a while. A rejected entry releases its
    inputs but keeps a pinned fee record, since the chain can still charge
    the fee of a rejected execution.
    """
    return entry.status != "cancelled" and (entry.status != "rejected" or entry.fee_record_id is not None)


def held_record_ids(entry: RecordSpend) -> tuple[str, ...]:
    """Return the nonces an entry currently excludes from selection."""
    if not holds_records(entry):
        return ()
    if entry.status == "rejected":
        return (entry.fee_record_id,) if entry.fee_record_id is not None else ()
    return entry.records


class MemoryRecordInventoryStore:
    """Reserve records for writers in one process.

    Entries disappear when the process exits, so a restart cannot resume a
    prepared transaction; supply a durable implementation for a long-running
    service. Each method holds a lock only for the duration of the call. A
    reservation is the entry's stored status, which is why it can outlive a
    crash in a durable backend.
    """

    def __init__(self) -> None:
        self._entries: list[RecordSpend] = []
        self._lock = threading.Lock()

    def list(self, scope: str) -> list[RecordSpend]:
        """Return copies of every entry under *scope*, oldest first.

        Entries are copied so a caller cannot change the journal by mutating
        a result. Saved transactions are shared, not copied: they are written
        once at the prepared checkpoint and read back only for rebroadcast,
        and copying them on every reservation lookup would cost megabytes
        for a busy journal.
        """
        with self._lock:
            return [copy.copy(entry) for entry in self._entries if entry.scope == scope]

    def acquire(self, entry: RecordSpend) -> RecordSpend | None:
        """Reserve all of the entry's records or none of them.

        The supplied ``id`` is ignored; the store assigns one. Returns
        ``None`` when any record is already held under the same scope, so the
        caller can stop before proving.

        Raises:
            ValueError: The entry names the same record twice.
        """
        if len(set(entry.records)) != len(entry.records):
            raise ValueError("Duplicate record inputs")
        with self._lock:
            # Check contention and insert under one lock so another thread
            # cannot reserve an input between the check and the insertion.
            wanted = set(entry.records)
            for old in self._entries:
                if old.scope == entry.scope and wanted.intersection(held_record_ids(old)):
                    return None
            saved = copy.deepcopy(entry)
            saved.id = str(len(self._entries) + 1)
            self._entries.append(saved)
            return copy.copy(saved)

    def update(self, scope: str, entry_id: str, patch: RecordSpendPatch) -> None:
        """Apply a checkpoint to an existing entry.

        Raises:
            KeyError: No entry with that id exists under *scope*.
        """
        with self._lock:
            for entry in self._entries:
                if entry.scope == scope and entry.id == entry_id:
                    if "status" in patch:
                        entry.status = patch["status"]
                    if "transaction_id" in patch:
                        entry.transaction_id = patch["transaction_id"]
                    if "transaction" in patch:
                        # ``None`` removes a transaction that recovery no longer needs.
                        entry.transaction = copy.deepcopy(patch["transaction"])
                    if "fee_microcredits" in patch:
                        entry.fee_microcredits = patch["fee_microcredits"]
                    return
        raise KeyError("Unknown record reservation")


def reserved_record_ids(store: RecordInventoryStore | None, scope: str) -> set[str]:
    """Return every nonce the journal excludes under *scope*; empty without a store."""
    if store is None:
        return set()
    out: set[str] = set()
    for entry in store.list(scope):
        out.update(held_record_ids(entry))
    return out


__all__ = ["MemoryRecordInventoryStore", "RecordInventoryStore", "held_record_ids", "holds_records", "reserved_record_ids"]
