"""Offline checks of the concurrent-swaps example: batch orchestration and the file store.

_source: ProvableHQ/veil examples/shield-swap/concurrent-swaps.test.ts, PR #172 @ 2e324d5.
Network, proving, and the DEX are replaced; the example's own code runs.
"""
from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from aleo.inventory import RecordSpend
from aleo_shield_swap.types import ClaimResult, SwapHandle

EXAMPLE = Path(__file__).parents[1] / "examples" / "concurrent-swaps" / "concurrent_swaps.py"


def load_example() -> ModuleType:
    spec = importlib.util.spec_from_file_location("concurrent_swaps_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module          # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


class _Call:
    def __init__(self, result: Any) -> None:
        self._result = result

    def delegate(self, **kw: Any) -> Any:
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result


class FakeDex:
    """Swaps succeed or fail per index; claims follow a second table."""

    def __init__(self, swap_errors: dict[int, BaseException] | None = None, claim_errors: dict[int, BaseException] | None = None) -> None:
        self.swap_errors = swap_errors or {}
        self.claim_errors = claim_errors or {}
        self.swaps = 0
        self.claimed: list[str] = []
        self._lock = threading.Lock()

    def swap(self, quote: Any) -> _Call:
        with self._lock:
            self.swaps += 1
            index = self.swaps
        if index in self.swap_errors:
            return _Call(self.swap_errors[index])
        return _Call(SwapHandle(swap_id=f"{index}field", blinding_factor="1field", blinded_address="aleo1x",
                                token_in_id="1field", token_out_id="2field", pool_key="5field", amount_in=10,
                                transaction_id=f"at1swap{index}", program="shield_swap.aleo"))

    def claim_swap_output(self, handle: SwapHandle, *, timeout: float = 0.0) -> _Call:
        index = int(handle.transaction_id.removeprefix("at1swap"))
        self.claimed.append(handle.transaction_id)
        if index in self.claim_errors:
            return _Call(self.claim_errors[index])
        return _Call(ClaimResult(transaction_id=f"at1claim{index}", amount_out=5, amount_remaining=0))


def test_batch_reports_every_trade_and_keeps_handles_of_failed_claims() -> None:
    example = load_example()
    dex = FakeDex(swap_errors={2: RuntimeError("no record")}, claim_errors={3: TimeoutError("output not finalized")})
    outcomes = example.submit_concurrent_swaps(dex, quote=object(), count=4, claim_timeout=1.0)
    # The pool runs trades concurrently, so which batch slot hits which fake
    # failure is not fixed; the shape of the report is.
    assert [o.index for o in outcomes] == [1, 2, 3, 4]
    swap_failed = [o for o in outcomes if o.handle is None]
    claim_failed = [o for o in outcomes if o.handle is not None and o.claim is None]
    claimed = [o for o in outcomes if o.claim is not None]
    assert len(swap_failed) == 1 and isinstance(swap_failed[0].error, RuntimeError)
    assert len(claim_failed) == 1 and isinstance(claim_failed[0].error, TimeoutError)
    assert claim_failed[0].handle is not None and claim_failed[0].handle.transaction_id == "at1swap3"
    assert len(claimed) == 2 and all(o.claim is not None and o.claim.amount_out == 5 for o in claimed)
    assert sorted(dex.claimed) == ["at1swap1", "at1swap3", "at1swap4"]  # the failed claim was attempted, not skipped


def test_json_file_store_reserves_atomically_and_survives_reopen(tmp_path: Path) -> None:
    example = load_example()
    path = tmp_path / "inventory.json"
    store = example.JsonFileRecordInventoryStore(path)
    scope = '["aleo:testnet","aleo1owner"]'

    def entry(*records: str) -> RecordSpend:
        return RecordSpend(id="", scope=scope, records=records, program="credits.aleo", function="join",
                           status="reserved", created_at=1)

    first = store.acquire(entry("1group", "2group"))
    assert first is not None and first.id == "1"
    assert store.acquire(entry("2group", "3group")) is None
    store.update(scope, first.id, {"status": "prepared", "transaction_id": "at1x", "transaction": {"id": "at1x"}, "fee_microcredits": "100"})
    reopened = example.JsonFileRecordInventoryStore(path)
    saved = reopened.list(scope)[0]
    assert (saved.status, saved.transaction_id, saved.transaction, saved.fee_microcredits) == ("prepared", "at1x", {"id": "at1x"}, "100")
    reopened.update(scope, first.id, {"status": "confirmed", "transaction": None})
    assert reopened.list(scope)[0].transaction is None
    with pytest.raises(KeyError):
        reopened.update(scope, "99", {"status": "cancelled"})

    results: list[RecordSpend | None] = []
    barrier = threading.Barrier(6)

    def race() -> None:
        barrier.wait()
        results.append(example.JsonFileRecordInventoryStore(path).acquire(entry("9group")))

    threads = [threading.Thread(target=race) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(result is not None for result in results) == 1
