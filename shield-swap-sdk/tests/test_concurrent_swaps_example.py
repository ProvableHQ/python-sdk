"""Offline checks of the concurrent-swaps example: batch orchestration.

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
