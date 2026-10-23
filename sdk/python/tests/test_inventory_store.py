"""Journal stores: one contract, three implementations (memory, JSON file, SQLite).

_source: ProvableHQ/veil packages/core/src/inventory/store.ts,
packages/shield-swap-cli/src/storage.ts, and the store cases in
test/inventory/privateFees.test.ts and shield-swap-cli/test/inventory-storage.test.ts,
PR #172 @ 2e324d5.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable

import pytest

from aleo.inventory import (
    JsonFileRecordInventoryStore, MemoryRecordInventoryStore, RecordSpend, SqliteRecordInventoryStore, held_record_ids,
    holds_records,
)
from aleo.inventory.store import reserved_record_ids

SCOPE = '["testnet","aleo1owner"]'
Factory = Callable[[], Any]


def entry(*records: str, status: str = "reserved", fee: str | None = None, scope: str = SCOPE) -> RecordSpend:
    return RecordSpend(id="", scope=scope, records=records, program="credits.aleo", function="join",
                       status=status, created_at=1, fee_record_id=fee)  # type: ignore[arg-type]


@pytest.fixture(params=["memory", "json", "sqlite"])
def make_store(request: pytest.FixtureRequest, tmp_path: Path) -> Factory:
    """A factory returning a store over the same backing state each call (a reopen for the durable ones)."""
    kind = request.param
    memory = MemoryRecordInventoryStore()
    if kind == "memory":
        return lambda: memory
    if kind == "json":
        return lambda: JsonFileRecordInventoryStore(tmp_path / "inventory.json")
    return lambda: SqliteRecordInventoryStore(tmp_path / "inventory.sqlite")


def test_acquire_is_all_or_nothing_and_assigns_ids(make_store: Factory) -> None:
    store = make_store()
    first = store.acquire(entry("1group", "2group"))
    assert first is not None and first.id == "1"
    assert store.acquire(entry("2group", "3group")) is None          # overlaps a held input
    second = store.acquire(entry("3group"))
    assert second is not None and second.id == "2"
    with pytest.raises(ValueError, match="Duplicate"):
        store.acquire(entry("4group", "4group"))
    assert [e.records for e in store.list(SCOPE)] == [("1group", "2group"), ("3group",)]
    assert store.list('["mainnet","aleo1owner"]') == []


def test_list_and_acquire_return_copies(make_store: Factory) -> None:
    store = make_store()
    saved = store.acquire(entry("1group"))
    assert saved is not None
    saved.status = "cancelled"                                        # mutating the copy changes nothing
    assert store.list(SCOPE)[0].status == "reserved"


def test_update_patches_fields_and_clears_transactions(make_store: Factory) -> None:
    store = make_store()
    saved = store.acquire(entry("1group"))
    assert saved is not None
    store.update(SCOPE, saved.id, {"status": "prepared", "transaction_id": "at1x", "transaction": {"id": "at1x"}, "fee_microcredits": "100"})
    stored = make_store().list(SCOPE)[0]
    assert (stored.status, stored.transaction_id, stored.transaction, stored.fee_microcredits) == ("prepared", "at1x", {"id": "at1x"}, "100")
    store.update(SCOPE, saved.id, {"status": "confirmed", "transaction": None})
    assert make_store().list(SCOPE)[0].transaction is None
    # Folding a fee record into the entry changes both fields in one update.
    store.update(SCOPE, saved.id, {"records": ("1group", "7group"), "fee_record_id": "7group"})
    folded = make_store().list(SCOPE)[0]
    assert (folded.records, folded.fee_record_id) == (("1group", "7group"), "7group")
    with pytest.raises(KeyError):
        store.update(SCOPE, "99", {"status": "cancelled"})


def test_rejected_entries_release_inputs_but_keep_the_charged_fee_record(make_store: Factory) -> None:
    store = make_store()
    saved = store.acquire(entry("1group", "2group", fee="2group"))
    assert saved is not None
    store.update(SCOPE, saved.id, {"status": "rejected"})
    assert reserved_record_ids(store, SCOPE) == {"2group"}
    assert store.acquire(entry("2group")) is None
    assert store.acquire(entry("1group")) is not None
    assert reserved_record_ids(None, SCOPE) == set()


def test_concurrent_acquisitions_never_share_an_input(make_store: Factory) -> None:
    barrier = threading.Barrier(8)
    results: list[RecordSpend | None] = []
    lock = threading.Lock()

    def worker() -> None:
        store = make_store()                                          # separate handles, one backing journal
        barrier.wait()
        result = store.acquire(entry("1group"))
        with lock:
            results.append(result)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(result is not None for result in results) == 1
    assert len(make_store().list(SCOPE)) == 1


def test_held_records_follow_status_and_fee_records() -> None:
    assert holds_records(entry("1group", status="confirmed")) is True
    assert holds_records(entry("1group", status="cancelled")) is False
    assert holds_records(entry("1group", status="rejected")) is False
    rejected_with_fee = entry("1group", "2group", status="rejected", fee="2group")
    assert holds_records(rejected_with_fee) is True
    assert held_record_ids(rejected_with_fee) == ("2group",)
    assert held_record_ids(entry("1group", "2group", status="submitted", fee="2group")) == ("1group", "2group")


def test_entries_round_trip_with_veil_field_names() -> None:
    saved = RecordSpend(id="1", scope=SCOPE, records=("1group",), program="credits.aleo", function="split",
                        status="prepared", created_at=5, asset_program=None, fee_record_id=None,
                        transaction_id="at1x", transaction={"id": "at1x"}, fee_microcredits="7")
    data = saved.to_dict()
    assert data["createdAt"] == 5 and data["transactionId"] == "at1x" and "assetProgram" not in data
    assert RecordSpend.from_dict(data) == saved


# ── Durable stores ────────────────────────────────────────────────────────────

@pytest.fixture(params=["json", "sqlite"])
def durable(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[Factory, Path]:
    if request.param == "json":
        path = tmp_path / "journal" / "inventory.json"
        return (lambda: JsonFileRecordInventoryStore(path)), path
    path = tmp_path / "journal" / "inventory.sqlite"
    return (lambda: SqliteRecordInventoryStore(path)), path


def test_durable_stores_create_private_files_and_survive_reopen(durable: tuple[Factory, Path]) -> None:
    make, path = durable
    store = make()
    assert path.exists()
    if sys.platform != "win32":
        assert (path.stat().st_mode & 0o077) == 0 and (path.parent.stat().st_mode & 0o077) == 0
    saved = store.acquire(entry("1group"))
    assert saved is not None
    store.update(SCOPE, saved.id, {"status": "submitted", "transaction_id": "at1x", "transaction": {"id": "at1x"}})
    if sys.platform != "win32":
        assert (path.stat().st_mode & 0o077) == 0, "writes keep the journal owner-only"
    reopened = make().list(SCOPE)
    assert len(reopened) == 1 and reopened[0].status == "submitted" and reopened[0].transaction == {"id": "at1x"}
    assert make().acquire(entry("1group")) is None                    # still held after reopening


def test_durable_stores_release_reservations_of_dead_local_processes(durable: tuple[Factory, Path]) -> None:
    make, _ = durable
    writer = make()
    writer._pid = dead_pid()  # pyright: ignore[reportPrivateUsage]
    held = writer.acquire(entry("1group"))
    assert held is not None
    live = make()
    live._pid = os.getpid()  # pyright: ignore[reportPrivateUsage]
    prepared = live.acquire(entry("2group"))
    assert prepared is not None
    live.update(SCOPE, prepared.id, {"status": "prepared", "transaction_id": "at1y", "transaction": {"id": "at1y"}})
    wallet = make()
    wallet._pid = dead_pid()  # pyright: ignore[reportPrivateUsage]
    rpc = entry("3group")
    rpc.account_type = "rpc"                                              # a Veil wallet client's entry
    assert wallet.acquire(rpc) is not None
    journal = make().list(SCOPE)
    # The dead SDK writer's reserved entry never reached the network; a prepared
    # entry from a dead writer is kept, anything from a live one is kept, and a
    # wallet's reserved entry is kept because the wallet may have broadcast.
    assert [(e.records, e.status) for e in journal] == [(("1group",), "cancelled"), (("2group",), "prepared"), (("3group",), "reserved")]
    assert make().acquire(entry("1group")) is not None
    assert make().acquire(entry("3group")) is None


def dead_pid() -> int:
    """The pid of a process that has already exited."""
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def test_sqlite_layout_matches_veil(tmp_path: Path) -> None:
    path = tmp_path / "inventory.sqlite"
    store = SqliteRecordInventoryStore(path)
    saved = store.acquire(entry("1group", "2group"))
    assert saved is not None
    store.update(SCOPE, saved.id, {"status": "prepared", "transaction_id": "at1x", "fee_microcredits": "100"})
    store.close()
    rows = sqlite3.connect(str(path)).execute("SELECT id, scope, data, pid, host FROM record_spends").fetchall()
    assert len(rows) == 1
    row_id, scope, data, pid, host = rows[0]
    assert (row_id, scope, pid, bool(host)) == (1, SCOPE, os.getpid(), True)
    payload = json.loads(data)
    assert payload["id"] == "1" and payload["records"] == ["1group", "2group"]
    assert payload["status"] == "prepared" and payload["transactionId"] == "at1x" and payload["feeMicrocredits"] == "100"
    assert payload["createdAt"] == 1 and payload["accountType"] == "local"
    # A Veil-written row is read back as-is.
    with sqlite3.connect(str(path)) as raw:
        veil = {"id": "2", "scope": SCOPE, "records": ["9group"], "program": "credits.aleo", "function": "split",
                "status": "submitted", "createdAt": 7, "accountType": "local", "transactionId": "at1veil"}
        raw.execute("INSERT INTO record_spends(scope, data, pid, host) VALUES (?, ?, ?, ?)", (SCOPE, json.dumps(veil), 1, "elsewhere"))
    with SqliteRecordInventoryStore(path) as again:
        entries = again.list(SCOPE)
    assert entries[1].records == ("9group",) and entries[1].transaction_id == "at1veil"


def test_json_file_is_plain_json_with_writer_metadata(tmp_path: Path) -> None:
    path = tmp_path / "inventory.json"
    store = JsonFileRecordInventoryStore(path)
    saved = store.acquire(entry("1group"))
    assert saved is not None
    items = json.loads(path.read_text())
    assert items[0]["data"]["records"] == ["1group"] and items[0]["pid"] == os.getpid() and items[0]["host"]
