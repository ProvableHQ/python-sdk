"""In-memory journal store: atomic acquisition, held records, patches.

_source: ProvableHQ/veil packages/core/src/inventory/store.ts and
test/inventory/privateFees.test.ts (rejected fee records), PR #172 @ 2e324d5.
"""
from __future__ import annotations

import threading

import pytest

from aleo.inventory import MemoryRecordInventoryStore, RecordSpend, held_record_ids, holds_records
from aleo.inventory.store import reserved_record_ids

SCOPE = '["testnet","aleo1owner"]'


def entry(*records: str, status: str = "reserved", fee: str | None = None, scope: str = SCOPE) -> RecordSpend:
    return RecordSpend(id="", scope=scope, records=records, program="credits.aleo", function="join",
                       status=status, created_at=1, fee_record_id=fee)  # type: ignore[arg-type]


def test_acquire_is_all_or_nothing_and_assigns_ids() -> None:
    store = MemoryRecordInventoryStore()
    first = store.acquire(entry("1group", "2group"))
    assert first is not None and first.id == "1"
    assert store.acquire(entry("2group", "3group")) is None          # overlaps a held input
    assert store.acquire(entry("3group")) is not None
    with pytest.raises(ValueError, match="Duplicate"):
        store.acquire(entry("4group", "4group"))
    assert [e.records for e in store.list(SCOPE)] == [("1group", "2group"), ("3group",)]
    assert store.list('["mainnet","aleo1owner"]') == []


def test_list_and_acquire_return_copies() -> None:
    store = MemoryRecordInventoryStore()
    saved = store.acquire(entry("1group"))
    assert saved is not None
    saved.status = "cancelled"                                        # mutating the copy changes nothing
    assert store.list(SCOPE)[0].status == "reserved"


def test_update_patches_fields_and_clears_transactions() -> None:
    store = MemoryRecordInventoryStore()
    saved = store.acquire(entry("1group"))
    assert saved is not None
    store.update(SCOPE, saved.id, {"status": "prepared", "transaction_id": "at1x", "transaction": {"id": "at1x"}, "fee_microcredits": "100"})
    stored = store.list(SCOPE)[0]
    assert (stored.status, stored.transaction_id, stored.transaction, stored.fee_microcredits) == ("prepared", "at1x", {"id": "at1x"}, "100")
    store.update(SCOPE, saved.id, {"status": "confirmed", "transaction": None})
    assert store.list(SCOPE)[0].transaction is None
    with pytest.raises(KeyError):
        store.update(SCOPE, "99", {"status": "cancelled"})


def test_held_records_follow_status_and_fee_records() -> None:
    assert holds_records(entry("1group", status="confirmed")) is True
    assert holds_records(entry("1group", status="cancelled")) is False
    assert holds_records(entry("1group", status="rejected")) is False
    rejected_with_fee = entry("1group", "2group", status="rejected", fee="2group")
    assert holds_records(rejected_with_fee) is True
    assert held_record_ids(rejected_with_fee) == ("2group",)
    assert held_record_ids(entry("1group", "2group", status="submitted", fee="2group")) == ("1group", "2group")


def test_rejected_entries_release_inputs_but_keep_the_charged_fee_record() -> None:
    store = MemoryRecordInventoryStore()
    saved = store.acquire(entry("1group", "2group", fee="2group"))
    assert saved is not None
    store.update(SCOPE, saved.id, {"status": "rejected"})
    assert reserved_record_ids(store, SCOPE) == {"2group"}
    assert store.acquire(entry("2group")) is None
    assert store.acquire(entry("1group")) is not None
    assert reserved_record_ids(None, SCOPE) == set()


def test_concurrent_acquisitions_never_share_an_input() -> None:
    store = MemoryRecordInventoryStore()
    barrier = threading.Barrier(8)
    results: list[RecordSpend | None] = []
    lock = threading.Lock()

    def worker() -> None:
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


def test_entries_round_trip_with_veil_field_names() -> None:
    saved = RecordSpend(id="1", scope=SCOPE, records=("1group",), program="credits.aleo", function="split",
                        status="prepared", created_at=5, asset_program=None, fee_record_id=None,
                        transaction_id="at1x", transaction={"id": "at1x"}, fee_microcredits="7")
    data = saved.to_dict()
    assert data["createdAt"] == 5 and data["transactionId"] == "at1x" and "assetProgram" not in data
    assert RecordSpend.from_dict(data) == saved
