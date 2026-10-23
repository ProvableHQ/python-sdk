"""Write tier: record inventory maintenance, recovery, and concurrent swaps on the REAL testnet.

_source: ProvableHQ/veil packages/shield-swap/test/integration/
{recordInventoryHarness.ts, recordInventory.e2e.test.ts,
recordInventoryStrategy.e2e.test.ts, concurrentInventory.e2e.test.ts},
PR #172 @ 2e324d5.

Keeps ``SHIELD_SWAP_INVENTORY_RECORDS`` (default 6) equal records of USDCx and
ETH in the e2e account, trades USDCx to ETH and back (one explicit round trip,
then a concurrent batch in each direction with automatic record selection),
and verifies record identity, exact token accounting, durable reservation
outcomes, and recovery after reopening the journal. Spends testnet funds and
takes tens of minutes. Opt in with ``-m "live and slow"`` and
``ALEO_E2E_PRIVATE_KEY``; the journal lives under
``.shield-swap/inventory-e2e/testnet/<address>/`` and survives the run.
"""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

import pytest

from aleo import Aleo, HTTPProvider, TransactionNotFound
from aleo.facade.call import BoundCall
from aleo.inventory import (
    InventoryRecord, InventoryTarget, RecordAsset, RecordInventorySnapshot, RecordManagementConfig, RecordSpend,
    SqliteRecordInventoryStore,
)
from aleo_shield_swap import ShieldSwap
from aleo_shield_swap._core import record_nonce
from aleo_shield_swap.journal import Journal
from aleo_shield_swap.types import ClaimResult, SwapHandle

from .conftest import ENDPOINT, PRIVATE_KEY, write_tier

pytestmark = [pytest.mark.live, pytest.mark.slow]

WRITE_TIMEOUT = 600.0        # proving, confirmation, and scanner indexing for one write
REBALANCE_TIMEOUT = 2400.0   # a bounded sequence of dependent joins and splits
_failed = False


@pytest.fixture(autouse=True)
def _abort_after_first_failure(request: pytest.FixtureRequest) -> Any:
    """Later steps depend on earlier ones; after a failure, skip instead of spending more."""
    if _failed:
        pytest.skip("an earlier inventory step failed; retain the journal before retrying")
    yield
    report = getattr(request.node, "rep_call", None)
    if report is not None and report.failed:
        globals()["_failed"] = True


def poll(label: str, read: Callable[[], Any], ready: Callable[[Any], bool], timeout: float = WRITE_TIMEOUT) -> Any:
    """Wait for a chain or scanner predicate without resubmitting anything."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = read()
        if ready(value):
            return value
        time.sleep(3.0)
    raise TimeoutError(f"Timed out waiting for {label}; retain the inventory and swap journals before retrying")


def base_units(amount: str, decimals: int) -> int:
    return int(Decimal(amount).scaleb(decimals))


@dataclass
class Token:
    symbol: str
    address: str
    decimals: int
    program: str
    minimum: int
    target: InventoryTarget

    @property
    def asset(self) -> RecordAsset:
        return RecordAsset(self.program)


class SwitchableStore:
    """Delegates to an inner store that a test may replace for one batch."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def list(self, scope: str) -> list[RecordSpend]:
        return self.inner.list(scope)

    def acquire(self, entry: RecordSpend) -> RecordSpend | None:
        return self.inner.acquire(entry)

    def update(self, scope: str, entry_id: str, patch: Any) -> None:
        self.inner.update(scope, entry_id, patch)


class Harness:
    """The e2e account with a durable reservation journal, a swap journal, and inventory helpers."""

    def __init__(self) -> None:
        self.count = int(os.environ.get("SHIELD_SWAP_INVENTORY_RECORDS", "6"))
        if not 3 <= self.count <= 16:
            raise ValueError("SHIELD_SWAP_INVENTORY_RECORDS must be from 3 through 16")
        probe = Aleo(HTTPProvider(ENDPOINT, network="testnet"))
        self.account = probe.account.from_private_key(PRIVATE_KEY)
        self.address = str(self.account.address)
        self.directory = Path(os.environ.get("SHIELD_SWAP_INVENTORY_E2E_DIR")
                              or Path(__file__).parents[2] / ".shield-swap" / "inventory-e2e" / "testnet" / self.address)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.scope = f'["aleo:testnet","{self.address}"]'
        self.connect()
        unresolved = [e for e in self.aleo.inventory.reconcile_record_inventory()
                      if e.status not in ("confirmed", "rejected", "cancelled")]
        if unresolved:
            raise RuntimeError(f"Resolve {len(unresolved)} pending inventory operations in {self.directory} before another run")
        registry = {t.symbol: t for t in self.dex.api.get_tokens()}
        self.tokens: list[Token] = []
        for symbol, minimum in (("USDCx", os.environ.get("SHIELD_SWAP_INVENTORY_MIN_USDCX", "0.1")),
                                ("ETH", os.environ.get("SHIELD_SWAP_INVENTORY_MIN_ETH", "0.000001"))):
            doc = registry[symbol]
            if not doc.underlying_program:
                raise RuntimeError(f"{symbol} lacks underlying program metadata")
            lot = base_units(minimum, doc.decimals)
            self.tokens.append(Token(symbol, doc.address, doc.decimals, doc.underlying_program, lot,
                                     InventoryTarget(records=self.count, min_record_amount=lot, distribution="balanced", tolerance_bps=0)))
        for token in self.tokens:
            inventory = self.scan(token)
            if inventory.balance < token.minimum * (self.count + 1):
                raise RuntimeError(f"Insufficient {token.symbol} for {self.count} records plus a trading margin")
            print(f"inventory preflight: {token.symbol}, {len(inventory.available)} records, {inventory.balance} base units")

    def connect(self) -> None:
        """Bind a fresh client to the same account, journal file, and swap journal."""
        previous = getattr(self, "file_store", None)
        if previous is not None:
            previous.close()
        self.file_store = SqliteRecordInventoryStore(self.directory / "inventory.sqlite")
        # The client holds this wrapper, so a test can interpose on acquisition
        # without rebuilding the client's frozen management settings.
        self.store = SwitchableStore(self.file_store)
        self.aleo = Aleo(HTTPProvider(ENDPOINT, network="testnet"),
                         record_management=RecordManagementConfig(enabled=True, store=self.store, chain_id="aleo:testnet",
                                                                  wait_for_sync=True, sync_timeout=WRITE_TIMEOUT))
        self.account = self.aleo.account.from_private_key(PRIVATE_KEY)
        self.aleo.default_account = self.account
        registration = self.aleo.records.register(self.account)
        assert registration.get("ok"), registration
        self.dex = ShieldSwap(self.aleo)
        self.dex.journal = Journal(self.directory / "swaps.jsonl")
        self.dex.api.authenticate(self.address, lambda message: str(self.aleo.account.sign(message.encode(), self.account)))

    def restart(self) -> None:
        """Reopen the journal file and rebuild the client, as a new process would."""
        self.connect()

    # ── Inventory helpers ────────────────────────────────────────────────────

    def scan(self, token: Token) -> RecordInventorySnapshot:
        return self.aleo.inventory.get_record_inventory(token.asset)

    def assert_balanced(self, token: Token) -> RecordInventorySnapshot:
        inventory = self.scan(token)
        assert len(inventory.available) == self.count, f"{token.symbol} record count"
        assert inventory.reserved == (), f"{token.symbol} pending inputs"
        assert all(record.amount >= token.minimum for record in inventory.available), f"{token.symbol} minimum"
        amounts = sorted(record.amount for record in inventory.available)
        # Zero tolerance permits only integer rounding: equal lots differ by at most one base unit.
        assert amounts[-1] - amounts[0] <= 1, f"{token.symbol} balanced amounts {amounts}"
        assert len({record.id for record in inventory.available}) == self.count
        return inventory

    def rebalance(self, token: Token) -> Any:
        before = self.scan(token)
        result = self.dex.rebalance_inventory(token.address, token.target, max_transactions=24,
                                              timeout=WRITE_TIMEOUT, poll_interval=3.0)
        assert result.status == "complete", result.reason
        after = self.assert_balanced(token)
        assert after.balance == before.balance, f"{token.symbol} rebalance conserves value"
        print(f"inventory balanced: {self.count} {token.symbol} records, {result.completed_steps} transactions")
        return result

    def confirm(self, tx_id: str) -> dict[str, Any]:
        def read() -> Any:
            try:
                return self.aleo.network.get_confirmed_transaction(tx_id)
            except TransactionNotFound:
                return None
        confirmed = poll(f"confirmation of {tx_id}", read, lambda value: value is not None)
        assert confirmed["status"] == "accepted", f"Transaction {tx_id} was not accepted"
        self.aleo.inventory.reconcile_record_inventory()
        return confirmed["transaction"]

    def output_records(self, token: Token, tx_id: str, expected: int) -> list[InventoryRecord]:
        transaction = self.confirm(tx_id)
        # Only outputs of the accepted transaction may satisfy this step; a
        # concurrent deposit or a stale scanner page cannot stand in for one.
        commitments = {str(output["id"]) for transition in transaction["execution"]["transitions"]
                       for output in transition.get("outputs", [])
                       if output.get("type") in ("record", "record_with_dynamic_id")}
        inventory = poll(f"{token.symbol} outputs of {tx_id}", lambda: self.scan(token),
                         lambda inv: len([r for r in inv.available if r.commitment in commitments]) == expected)
        return [record for record in inventory.available if record.commitment in commitments]

    def journal_entry(self, tx_id: str) -> RecordSpend:
        entry = next((e for e in self.store.list(self.scope) if e.transaction_id == tx_id), None)
        assert entry is not None, f"journal entry for {tx_id}"
        return entry

    def pool_key(self, a: Token, b: Token) -> str:
        """The deepest pool for the pair, by on-chain liquidity."""
        best, deepest = None, -1
        for pool in self.dex.api.get_pools():
            if {pool.token0, pool.token1} == {a.address, b.address}:
                liquidity = self.dex.get_slot(pool.key).liquidity
                if liquidity > deepest:
                    best, deepest = pool.key, liquidity
        assert best is not None and deepest > 0, "No USDCx/ETH pool with on-chain liquidity"
        return best


@pytest.fixture(scope="module")
def h() -> Harness:
    if not PRIVATE_KEY:
        pytest.skip("ALEO_E2E_PRIVATE_KEY not set")
    return Harness()


@pytest.fixture(scope="module")
def usdcx(h: Harness) -> Token:
    return h.tokens[0]


@pytest.fixture(scope="module")
def eth(h: Harness) -> Token:
    return h.tokens[1]


@pytest.fixture(scope="module")
def pool(h: Harness, usdcx: Token, eth: Token) -> str:
    return h.pool_key(usdcx, eth)


@pytest.fixture(scope="module")
def trade_amount(usdcx: Token) -> int:
    amount = base_units(os.environ.get("SHIELD_SWAP_INVENTORY_SWAP_USDCX", "0.1"), usdcx.decimals)
    assert amount > 0
    return amount


TRANSACTIONS: list[str] = []
CONSUMED: set[str] = set()


def swap_and_claim(h: Harness, pool: str, source: Token, target: Token, amount: int, record: InventoryRecord) -> tuple[int, InventoryRecord]:
    """One explicit-record swap and its claim, checked against independently predicted balances."""
    before_in, before_out = h.scan(source), h.scan(target)
    assert record.amount >= amount, f"{source.symbol} covering input"
    assert record.id not in CONSUMED, "a previous swap input cannot be reused"
    handle = h.dex.swap(pool_key=pool, token_in_id=source.address, amount_in=amount, slippage_bps=100,
                        token_record=record.plaintext, token_in_program=source.program).delegate(wait=True, wait_timeout=WRITE_TIMEOUT)
    CONSUMED.add(record.id)
    TRANSACTIONS.append(handle.transaction_id)
    assert handle.swap_id, "confirmed swap has a claim id"
    claim = h.dex.claim_swap_output(handle, timeout=WRITE_TIMEOUT).delegate(wait=True, wait_timeout=WRITE_TIMEOUT)
    TRANSACTIONS.append(claim.transaction_id)
    outputs = h.output_records(target, claim.transaction_id, 1)
    assert claim.amount_out > 0 and claim.amount_remaining == 0, "small test trades fill completely"
    assert outputs[0].amount == claim.amount_out
    after_in, after_out = h.scan(source), h.scan(target)
    assert after_in.balance == before_in.balance - amount, f"{source.symbol} debit"
    assert after_out.balance == before_out.balance + claim.amount_out, f"{target.symbol} credit"
    assert not any(r.id == record.id for r in after_in.available), "spent nonce stays excluded"
    spend = h.journal_entry(handle.transaction_id)
    assert spend.status == "confirmed" and record.id in spend.records
    assert handle.swap_id not in {p.swap_id for p in h.dex.journal.pending_claims()}
    return claim.amount_out, outputs[0]


@write_tier
def test_prepares_balanced_records_above_the_minimum(h: Harness, usdcx: Token, eth: Token, trade_amount: int) -> None:
    for token in h.tokens:
        h.rebalance(token)
    inventory = h.assert_balanced(usdcx)
    # Each source record stays above its minimum after a forward swap, so the
    # count change from each claim is the only reason maintenance is needed.
    assert all(record.amount - trade_amount >= usdcx.minimum for record in inventory.available)


@write_tier
def test_round_trip_then_restores_balanced_records(h: Harness, usdcx: Token, eth: Token, pool: str, trade_amount: int) -> None:
    before_usdcx, before_eth = h.assert_balanced(usdcx), h.assert_balanced(eth)
    amount_out, eth_record = swap_and_claim(h, pool, usdcx, eth, trade_amount, before_usdcx.available[0])
    assert len(h.scan(eth).available) == h.count + 1
    # Spend the exact newly claimed ETH record on the return leg, keeping the
    # existing ETH lots and avoiding a second ambiguous selection.
    returned, _ = swap_and_claim(h, pool, eth, usdcx, amount_out, eth_record)
    assert len(h.scan(usdcx).available) == h.count + 1
    repair = h.dex.plan_inventory(usdcx.address, usdcx.target, max_transactions=24)
    assert repair.steps[0].kind == "join" and len(repair.steps[0].inputs) == min(h.count + 1, 16)
    assert repair.steps[0].join_program is not None and repair.steps[0].join_program.program.startswith("aj_test_usdcx_stablecoin_")
    assert h.rebalance(usdcx).completed_steps == len(repair.steps)
    assert h.rebalance(eth).completed_steps == 0
    assert h.assert_balanced(usdcx).balance == before_usdcx.balance - trade_amount + returned
    assert h.assert_balanced(eth).balance == before_eth.balance
    print(f"round trip: {trade_amount} USDCx -> {amount_out} ETH -> {returned} USDCx base units")


class Barrier:
    """Holds the first *racers* acquisitions until all have arrived, so every swap selects the same record first."""

    def __init__(self, store: Any, racers: int) -> None:
        self._store, self._racers = store, racers
        self.attempts: list[tuple[str, ...]] = []
        self._lock = threading.Lock()
        self._gate = threading.Event()
        self._deadline = threading.Timer(60.0, self._gate.set)   # a failed preparation must not strand the others
        self._deadline.start()

    def list(self, scope: str) -> list[RecordSpend]:
        return self._store.list(scope)

    def update(self, scope: str, entry_id: str, patch: Any) -> None:
        self._store.update(scope, entry_id, patch)

    def acquire(self, entry: RecordSpend) -> RecordSpend | None:
        with self._lock:
            self.attempts.append(tuple(entry.records))
            first = len(self.attempts) <= self._racers
            if len(self.attempts) == self._racers:
                self._gate.set()
        if first and entry.records:
            self._gate.wait()
        return self._store.acquire(entry)

    def release(self) -> None:
        self._deadline.cancel()
        self._gate.set()


def batch(h: Harness, pool: str, source: Token, target: Token, amounts: list[int], monkeypatch: pytest.MonkeyPatch) -> tuple[list[tuple[SwapHandle, ClaimResult, InventoryRecord]], list[str]]:
    """Submit one swap per amount at the same time with automatic record selection, then claim each."""
    before_in, before_out = h.scan(source), h.scan(target)
    barrier = Barrier(h.file_store, len(amounts))
    h.store.inner = barrier
    proved: list[str] = []
    active, peak = [0], [0]
    counter = threading.Lock()
    original_delegate = BoundCall.delegate

    def counting_delegate(self: BoundCall, account: Any = None, *, broadcast: bool = True, **kw: Any) -> Any:
        # Count only the prover round trips the hook makes for the swaps, and
        # record nonces rather than plaintext.
        if broadcast or self.function_name not in ("swap", "swap_from_wrapped"):
            return original_delegate(self, account, broadcast=broadcast, **kw)
        nonce = next((record_nonce(a) for a in self.args if "_nonce" in a), None)
        assert nonce, "swap reached proving without a record input"
        with counter:
            proved.append(nonce)
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        try:
            return original_delegate(self, account, broadcast=broadcast, **kw)
        finally:
            with counter:
                active[0] -= 1

    monkeypatch.setattr(BoundCall, "delegate", counting_delegate)
    outcomes: list[Any] = []
    try:
        def one(amount: int) -> Any:
            try:
                handle = h.dex.swap(pool_key=pool, token_in_id=source.address, amount_in=amount, slippage_bps=100,
                                    token_in_program=source.program).delegate(wait=True, wait_timeout=WRITE_TIMEOUT)
                TRANSACTIONS.append(handle.transaction_id)
                print(f"{source.symbol} -> {target.symbol} swap confirmed: {handle.transaction_id}")
                return handle
            except BaseException as exc:  # noqa: BLE001 - settled below
                return exc
        with ThreadPoolExecutor(max_workers=len(amounts)) as workers:
            outcomes = list(workers.map(one, amounts))
    finally:
        barrier.release()
        h.store.inner = h.file_store
        monkeypatch.undo()
    handles = [o for o in outcomes if isinstance(o, SwapHandle)]

    def claim(handle: SwapHandle) -> Any:
        try:
            result = h.dex.claim_swap_output(handle, timeout=WRITE_TIMEOUT).delegate(wait=True, wait_timeout=WRITE_TIMEOUT)
            TRANSACTIONS.append(result.transaction_id)
            outputs = h.output_records(target, result.transaction_id, 1)
            assert result.amount_out > 0 and result.amount_remaining == 0
            assert outputs[0].amount == result.amount_out
            print(f"{source.symbol} -> {target.symbol} claimed {result.amount_out} base units: {result.transaction_id}")
            return handle, result, outputs[0]
        except BaseException as exc:  # noqa: BLE001 - settled below
            return exc

    # Settle every claim before asserting, even if a sibling swap or claim failed.
    with ThreadPoolExecutor(max_workers=max(1, len(handles))) as workers:
        claims = list(workers.map(claim, handles))
    failures = [f"operation {i + 1}: {o}" for i, o in enumerate([*outcomes, *claims]) if isinstance(o, BaseException)]
    assert failures == [], "all submitted swaps and successful outputs must settle"
    settled = [c for c in claims if not isinstance(c, BaseException)]
    assert len(handles) == len(amounts)
    assert len({handle.transaction_id for handle in handles}) == len(amounts)
    assert len({handle.blinded_address for handle in handles}) == len(amounts)
    assert peak[0] > 1, "proving executions overlap"
    assert len(proved) == len(amounts) and len(set(proved)) == len(amounts), "every prover consumes a distinct input"
    first = [a for a in barrier.attempts if a][: len(amounts)]
    assert len(first) == len(amounts)
    if source.symbol == "USDCx":
        assert len({ids[0] for ids in first}) == 1, "all first selections collide"
        assert len([a for a in barrier.attempts if a]) > len(amounts), "the losing selections retry"
    for nonce in proved:
        assert nonce not in CONSUMED, "inputs from earlier batches are not reused"
        CONSUMED.add(nonce)
    after_in, after_out = h.scan(source), h.scan(target)
    assert after_in.balance == before_in.balance - sum(amounts)
    assert after_out.balance == before_out.balance + sum(c.amount_out for _, c, _ in settled)
    assert not any(r.id in proved for r in after_in.available), "spent inputs stay excluded"
    for handle in handles:
        entry = h.journal_entry(handle.transaction_id)
        assert entry.status == "confirmed" and len(entry.records) == 1 and entry.records[0] in proved
    assert not {p.swap_id for p in h.dex.journal.pending_claims()} & {hd.swap_id for hd in handles}
    print(f"{source.symbol} batch verified: {len(handles)} swaps, {len(settled)} claims, {peak[0]} overlapping executions, "
          f"{len([a for a in barrier.attempts if a]) - len(handles)} reservation conflicts")
    return settled, proved


@write_tier
def test_swaps_concurrently_in_both_directions_and_restores_the_inventories(
    h: Harness, usdcx: Token, eth: Token, pool: str, trade_amount: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    before_usdcx, before_eth = h.assert_balanced(usdcx), h.assert_balanced(eth)
    forward, _ = batch(h, pool, usdcx, eth, [trade_amount] * h.count, monkeypatch)
    # Return the amounts received; automatic selection matches each to a
    # covering record instead of passing the plaintexts explicitly.
    backward, proved = batch(h, pool, eth, usdcx, [c.amount_out for _, c, _ in forward], monkeypatch)
    assert set(proved) == {record.id for _, _, record in forward}
    for token in h.tokens:
        h.rebalance(token)
    assert h.assert_balanced(usdcx).balance == before_usdcx.balance - trade_amount * h.count + sum(c.amount_out for _, c, _ in backward)
    assert h.assert_balanced(eth).balance == before_eth.balance


@write_tier
def test_retains_confirmed_checkpoints_after_reopening_the_journal(h: Harness) -> None:
    h.restart()
    journal = h.aleo.inventory.reconcile_record_inventory()
    assert len(TRANSACTIONS) == 4 + h.count * 4
    for tx_id in TRANSACTIONS:
        entry = next((e for e in journal if e.transaction_id == tx_id), None)
        assert entry is not None and entry.status == "confirmed", f"confirmed checkpoint for {tx_id}"
        assert entry.transaction is None
    used = [nonce for e in journal if e.status == "confirmed" for nonce in e.records]
    assert len(set(used)) == len(used), "no record was consumed by two confirmed transactions"
    for token in h.tokens:
        h.assert_balanced(token)
    print(json.dumps({"transactions": len(TRANSACTIONS), "journal": len(journal)}))
