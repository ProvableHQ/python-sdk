"""Journal stores for record reservations.

A journal holds record nonces, program and function names, and proved
transactions awaiting broadcast; it never holds keys or record plaintext.
Three stores ship with the SDK:

- :class:`MemoryRecordInventoryStore` lasts for one process. Use it for a
  script that runs to completion.
- :class:`JsonFileRecordInventoryStore` keeps the journal in one JSON file
  under a file lock. Use it when a process may restart and the file can be
  inspected by hand; it coordinates processes on one machine.
- :class:`SqliteRecordInventoryStore` keeps the journal in a SQLite database
  with the same layout as Veil's store, so the two SDKs can share one
  journal. Use it for a long-running service or several processes on one
  machine.

Anything else that satisfies :class:`~aleo.inventory.types.RecordInventoryStore`
can replace them, such as a database shared across machines. The two
file-backed stores also release reservations left by an SDK process on the
same machine that exited before proving, since such a reservation cannot
have reached the network; reservations written by a wallet client are kept.
"""
from __future__ import annotations

import copy
import json
import os
import socket
import sqlite3
import sys
import threading
from pathlib import Path
from typing import Any

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
                    _apply_patch(entry, patch)
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


# ── Durable stores ────────────────────────────────────────────────────────────


def _apply_patch(entry: RecordSpend, patch: RecordSpendPatch) -> None:
    """Copy every key present in *patch* onto *entry*; absent keys leave the field alone."""
    if "status" in patch:
        entry.status = patch["status"]
    if "records" in patch:
        entry.records = tuple(patch["records"])
    if "fee_record_id" in patch:
        entry.fee_record_id = patch["fee_record_id"]
    if "transaction_id" in patch:
        entry.transaction_id = patch["transaction_id"]
    if "transaction" in patch:
        # ``None`` removes a transaction that recovery no longer needs.
        entry.transaction = copy.deepcopy(patch["transaction"])
    if "fee_microcredits" in patch:
        entry.fee_microcredits = patch["fee_microcredits"]


def _process_alive(pid: int) -> bool:
    """Whether a process id on this machine is still running."""
    if pid <= 0:
        return False
    if sys.platform == "win32":  # pragma: no cover - exercised on Windows only
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # type: ignore[attr-defined]
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _private_file(path: Path) -> None:
    """Create *path* readable by the owner only, keeping an existing file's contents.

    A parent directory this call creates is owner-only as well; an existing
    directory keeps its permissions, since it may be the application's own
    directory or the working directory.
    """
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.parent.chmod(0o700)
        except OSError:  # pragma: no cover - not every filesystem honours modes
            pass
    if not path.exists():
        fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
        os.close(fd)


class _FileLock:
    """An exclusive lock on a sibling file, held for one read-modify-write."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._thread_lock = threading.Lock()

    def __enter__(self) -> None:
        self._thread_lock.acquire()
        # ``__exit__`` never runs when ``__enter__`` raises, so release what
        # was taken before an open or lock failure propagates; the next call
        # then gets a fresh attempt instead of blocking on this thread lock.
        try:
            self._handle = open(self._path, "a+")
            try:
                if sys.platform == "win32":  # pragma: no cover - exercised on Windows only
                    import msvcrt
                    msvcrt.locking(self._handle.fileno(), msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined]
                else:
                    import fcntl
                    fcntl.flock(self._handle, fcntl.LOCK_EX)
            except BaseException:
                self._handle.close()
                raise
        except BaseException:
            self._thread_lock.release()
            raise

    def __exit__(self, *exc: object) -> None:
        try:
            if sys.platform == "win32":  # pragma: no cover
                import msvcrt
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
            else:
                import fcntl
                fcntl.flock(self._handle, fcntl.LOCK_UN)
            self._handle.close()
        finally:
            self._thread_lock.release()


class _DurableStore:
    """Shared acquisition, patching, and stale-reservation recovery for the file-backed stores.

    Each entry is stored with the process id and host that wrote it. On every
    read under *scope*, a ``reserved`` entry written on this host by a process
    that no longer runs is cancelled when its ``account_type`` is ``local``:
    an SDK-held key always writes the prepared checkpoint before broadcast,
    so such an entry never reached the network. An entry a wallet client
    wrote (Veil's ``rpc`` account type) is kept, because the wallet may have
    broadcast before its response was lost.
    """

    def __init__(self) -> None:
        self._pid = os.getpid()
        self._host = socket.gethostname()

    # Subclasses implement these inside one exclusive transaction.
    def _rows(self, scope: str) -> list[tuple[RecordSpend, int, str]]:
        raise NotImplementedError

    def _insert(self, entry: RecordSpend) -> RecordSpend:
        raise NotImplementedError

    def _store(self, entry: RecordSpend) -> None:
        raise NotImplementedError

    def _transaction(self) -> Any:
        raise NotImplementedError

    def _recover(self, scope: str) -> list[RecordSpend]:
        entries: list[RecordSpend] = []
        for entry, pid, host in self._rows(scope):
            if entry.status == "reserved" and entry.account_type == "local" and host == self._host and not _process_alive(pid):
                entry.status = "cancelled"
                self._store(entry)
            entries.append(entry)
        return entries

    def list(self, scope: str) -> list[RecordSpend]:
        """Return every entry under *scope*, oldest first, after releasing reservations of dead local processes."""
        with self._transaction():
            return self._recover(scope)

    def acquire(self, entry: RecordSpend) -> RecordSpend | None:
        """Reserve all of the entry's records or none of them, atomically across processes.

        Raises:
            ValueError: The entry names the same record twice.
        """
        if len(set(entry.records)) != len(entry.records):
            raise ValueError("Duplicate record inputs")
        with self._transaction():
            wanted = set(entry.records)
            for old in self._recover(entry.scope):
                if wanted.intersection(held_record_ids(old)):
                    return None
            return self._insert(copy.deepcopy(entry))

    def update(self, scope: str, entry_id: str, patch: RecordSpendPatch) -> None:
        """Apply a checkpoint to an existing entry.

        Raises:
            KeyError: No entry with that id exists under *scope*.
        """
        with self._transaction():
            for entry, _, _ in self._rows(scope):
                if entry.id == entry_id:
                    _apply_patch(entry, patch)
                    self._store(entry)
                    return
        raise KeyError("Unknown record reservation")


class JsonFileRecordInventoryStore(_DurableStore):
    """A reservation journal in one JSON file, shared by processes on one machine.

    Every call takes an exclusive lock on a sibling ``.lock`` file, reads the
    whole file, and writes it back atomically, so two processes can never
    hold the same record. The file is created readable by its owner only and
    can be inspected with any JSON viewer. It holds nonces, program and
    function names, statuses, and proved transactions awaiting broadcast;
    never keys or record plaintext.

    Args:
        path: The journal file. Its parent directory is created if needed.
    """

    def __init__(self, path: "Path | str") -> None:
        super().__init__()
        self.path = Path(path)
        _private_file(self.path)
        self._lock = _FileLock(self.path.with_suffix(self.path.suffix + ".lock"))
        self._cache: list[dict[str, Any]] | None = None

    def _transaction(self) -> Any:
        store = self

        class _Guard:
            def __enter__(self) -> None:
                store._lock.__enter__()
                try:
                    text = store.path.read_text() or "[]"
                    store._cache = [dict(item) for item in json.loads(text)]
                except BaseException:
                    # A journal that cannot be read must not leave the locks
                    # held; once the file is repaired the next call proceeds.
                    store._cache = None
                    store._lock.__exit__(None, None, None)
                    raise

            def __exit__(self, kind: object, exc: object, tb: object) -> None:
                try:
                    if exc is None and store._cache is not None:
                        # Write a private temporary file, then replace, so the
                        # journal is never partially written and keeps mode 0600.
                        tmp = store.path.with_suffix(store.path.suffix + ".tmp")
                        descriptor = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
                        with os.fdopen(descriptor, "w") as handle:
                            json.dump(store._cache, handle, indent=2)
                            handle.flush()
                            os.fsync(handle.fileno())
                        os.replace(tmp, store.path)
                finally:
                    store._cache = None
                    store._lock.__exit__(kind, exc, tb)
        return _Guard()

    def _rows(self, scope: str) -> list[tuple[RecordSpend, int, str]]:
        assert self._cache is not None
        return [(RecordSpend.from_dict(item["data"]), int(item.get("pid", 0)), str(item.get("host", "")))
                for item in self._cache if item["data"].get("scope") == scope]

    def _insert(self, entry: RecordSpend) -> RecordSpend:
        assert self._cache is not None
        entry.id = str(max((int(item["data"]["id"]) for item in self._cache), default=0) + 1)
        self._cache.append({"data": entry.to_dict(), "pid": self._pid, "host": self._host})
        return copy.deepcopy(entry)

    def _store(self, entry: RecordSpend) -> None:
        assert self._cache is not None
        for item in self._cache:
            if item["data"].get("scope") == entry.scope and str(item["data"].get("id")) == entry.id:
                item["data"] = entry.to_dict()
                return
        raise KeyError("Unknown record reservation")


class SqliteRecordInventoryStore(_DurableStore):
    """A reservation journal in a SQLite database, shared by processes on one machine.

    Uses Python's bundled ``sqlite3``; nothing extra to install. The table
    layout and the JSON inside it match Veil's SQLite store, so a journal
    written by one SDK is read correctly by the other. Each call runs in a
    ``BEGIN IMMEDIATE`` transaction, so contention checks and inserts are
    atomic across processes; writes are flushed before the call returns. The
    database file is created readable by its owner only.

    Args:
        path: The database file. Its parent directory is created if needed.
    """

    def __init__(self, path: "Path | str") -> None:
        super().__init__()
        self.path = Path(path)
        _private_file(self.path)
        self._connection = sqlite3.connect(str(self.path), isolation_level=None, check_same_thread=False, timeout=5.0)
        self._thread_lock = threading.Lock()
        self._connection.executescript(
            "PRAGMA busy_timeout = 5000; PRAGMA journal_mode = WAL; PRAGMA synchronous = FULL;\n"
            "CREATE TABLE IF NOT EXISTS record_spends (\n"
            "  id INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT NOT NULL, data TEXT NOT NULL,\n"
            "  pid INTEGER NOT NULL, host TEXT NOT NULL\n"
            ");\n"
            "CREATE INDEX IF NOT EXISTS record_spends_scope ON record_spends(scope);"
        )

    def close(self) -> None:
        """Release the database handle; the journal stays on disk."""
        self._connection.close()

    def __enter__(self) -> "SqliteRecordInventoryStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _transaction(self) -> Any:
        store = self

        class _Guard:
            def __enter__(self) -> None:
                store._thread_lock.acquire()
                try:
                    store._connection.execute("BEGIN IMMEDIATE")
                except BaseException:
                    store._thread_lock.release()
                    raise

            def __exit__(self, kind: object, exc: object, tb: object) -> None:
                try:
                    store._connection.execute("COMMIT" if exc is None else "ROLLBACK")
                finally:
                    store._thread_lock.release()
        return _Guard()

    def _rows(self, scope: str) -> list[tuple[RecordSpend, int, str]]:
        rows = self._connection.execute(
            "SELECT id, data, pid, host FROM record_spends WHERE scope = ? ORDER BY id", (scope,),
        ).fetchall()
        out: list[tuple[RecordSpend, int, str]] = []
        for row_id, data, pid, host in rows:
            entry = RecordSpend.from_dict(json.loads(data))
            if entry.scope != scope or entry.id != str(row_id):
                raise ValueError("Invalid inventory journal entry")
            out.append((entry, int(pid), str(host)))
        return out

    def _insert(self, entry: RecordSpend) -> RecordSpend:
        cursor = self._connection.execute(
            "INSERT INTO record_spends(scope, data, pid, host) VALUES (?, ?, ?, ?)",
            (entry.scope, json.dumps(entry.to_dict()), self._pid, self._host),
        )
        entry.id = str(cursor.lastrowid)
        self._connection.execute("UPDATE record_spends SET data = ? WHERE id = ?", (json.dumps(entry.to_dict()), int(entry.id)))
        return copy.deepcopy(entry)

    def _store(self, entry: RecordSpend) -> None:
        changed = self._connection.execute(
            "UPDATE record_spends SET data = ? WHERE scope = ? AND id = ?", (json.dumps(entry.to_dict()), entry.scope, int(entry.id)),
        ).rowcount
        if changed != 1:
            raise KeyError("Unknown record reservation")


__all__ = [
    "JsonFileRecordInventoryStore", "MemoryRecordInventoryStore", "RecordInventoryStore", "SqliteRecordInventoryStore",
    "held_record_ids", "holds_records", "reserved_record_ids",
]
