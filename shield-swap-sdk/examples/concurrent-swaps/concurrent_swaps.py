"""Prepare six balanced USDCx records and swap them for ETH at the same time.

Preview (reads, plans, and quotes; submits nothing):

    python shield-swap-sdk/examples/concurrent-swaps/concurrent_swaps.py

Execute (reshapes the account's available USDCx records, submits six swaps,
claims each output):

    python shield-swap-sdk/examples/concurrent-swaps/concurrent_swaps.py --execute

Requires ``SHIELD_SWAP_PRIVATE_KEY`` naming a funded testnet account with DEX
access. Execution spends 0.1 USDCx per swap by default and pays network fees
through the delegated prover's fee master. See README.md for the recovery
steps after an interruption.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from aleo import Aleo, HTTPProvider, testnet
from aleo.inventory import (
    InventoryPlan, InventoryTarget, RecordManagementConfig, RecordSpend, RecordSpendPatch, held_record_ids,
)
from aleo_shield_swap import ShieldSwap
from aleo_shield_swap.journal import Journal
from aleo_shield_swap.types import ClaimResult, SwapHandle, SwapQuote

NETWORK = "testnet"
ENDPOINT = "https://edge.provable.com/api"
UNRESOLVED = ("reserved", "prepared", "submitted")


class JsonFileRecordInventoryStore:
    """A reservation journal in one JSON file, shared by processes on one machine.

    The SDK ships only an in-memory store. This class shows what a durable
    backend needs: every mutation finishes before the method returns, and
    ``acquire`` checks and inserts under one file lock so two processes can
    never hold the same record. A database with a uniqueness constraint on
    held nonces serves the same contract across machines.

    The file holds record nonces, program and function names, and proved
    transactions awaiting broadcast; never keys or record plaintext.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path = path.with_suffix(path.suffix + ".lock")
        self._thread_lock = threading.Lock()

    def _read(self) -> list[RecordSpend]:
        if not self.path.exists():
            return []
        return [RecordSpend.from_dict(item) for item in json.loads(self.path.read_text() or "[]")]

    def _write(self, entries: list[RecordSpend]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps([entry.to_dict() for entry in entries], indent=2))
        os.replace(tmp, self.path)

    def _locked(self) -> Any:
        store = self

        class _Guard:
            def __enter__(self) -> None:
                store._thread_lock.acquire()
                self.handle = open(store._lock_path, "a+")
                fcntl.flock(self.handle, fcntl.LOCK_EX)

            def __exit__(self, *exc: object) -> None:
                fcntl.flock(self.handle, fcntl.LOCK_UN)
                self.handle.close()
                store._thread_lock.release()
        return _Guard()

    def list(self, scope: str) -> list[RecordSpend]:
        with self._locked():
            return [entry for entry in self._read() if entry.scope == scope]

    def acquire(self, entry: RecordSpend) -> RecordSpend | None:
        if len(set(entry.records)) != len(entry.records):
            raise ValueError("Duplicate record inputs")
        with self._locked():
            entries = self._read()
            wanted = set(entry.records)
            for old in entries:
                if old.scope == entry.scope and wanted.intersection(held_record_ids(old)):
                    return None
            saved = RecordSpend(**{**entry.__dict__, "id": str(len(entries) + 1)})
            entries.append(saved)
            self._write(entries)
            return saved

    def update(self, scope: str, entry_id: str, patch: RecordSpendPatch) -> None:
        with self._locked():
            entries = self._read()
            for entry in entries:
                if entry.scope == scope and entry.id == entry_id:
                    if "status" in patch:
                        entry.status = patch["status"]
                    if "transaction_id" in patch:
                        entry.transaction_id = patch["transaction_id"]
                    if "transaction" in patch:
                        entry.transaction = patch["transaction"]
                    if "fee_microcredits" in patch:
                        entry.fee_microcredits = patch["fee_microcredits"]
                    self._write(entries)
                    return
        raise KeyError("Unknown record reservation")


@dataclass(frozen=True)
class SwapOutcome:
    """One trade's result, kept separate so a failed sibling never hides a successful claim.

    Attributes:
        index: Position in the batch, from 1.
        handle: The swap handle once submission succeeded; it is what a later
            claim needs, so keep it when ``claim`` is ``None``.
        claim: The confirmed claim with the output and any refund.
        error: Why the swap or the claim failed.
    """

    index: int
    handle: SwapHandle | None = None
    claim: ClaimResult | None = None
    error: BaseException | None = None


def submit_concurrent_swaps(dex: ShieldSwap, quote: SwapQuote, count: int, *, claim_timeout: float = 600.0) -> list[SwapOutcome]:
    """Submit *count* swaps from one quote at the same time, then claim each successful output.

    Each swap selects its own input record; the facade's record management
    reserves it before proving, and a swap whose record another swap took
    first moves to the next eligible record. A swap and its claim are two
    transactions, so each claim waits for the swap output to appear before
    submitting. Nothing is retried after an uncertain submission.
    """
    def one(index: int) -> SwapOutcome:
        try:
            handle = dex.swap(quote).delegate(wait=False)
        except BaseException as exc:  # noqa: BLE001 - reported per trade, never raised mid-batch
            return SwapOutcome(index, error=exc)
        try:
            claim = dex.claim_swap_output(handle, timeout=claim_timeout).delegate(wait=True)
            return SwapOutcome(index, handle=handle, claim=claim)
        except BaseException as exc:  # noqa: BLE001 - the handle is kept for a later claim
            return SwapOutcome(index, handle=handle, error=exc)

    with ThreadPoolExecutor(max_workers=count) as pool:
        return list(pool.map(one, range(1, count + 1)))


def build_client(private_key: str, directory: Path) -> tuple[Aleo, ShieldSwap]:
    """Bind an account, a durable reservation journal, and a swap journal to one client."""
    store = JsonFileRecordInventoryStore(directory / "inventory.json")
    aleo = Aleo(HTTPProvider(ENDPOINT, network=NETWORK),
                record_management=RecordManagementConfig(enabled=True, store=store, chain_id=f"aleo:{NETWORK}"))
    key = testnet.PrivateKey.from_string(private_key)
    account = aleo.account.from_private_key(key)
    aleo.default_account = account
    registration = aleo.records.register(account)      # shares the view key with the scanner
    if not registration.get("ok"):
        raise RuntimeError(f"Record scanner registration failed: {registration.get('error')}")
    dex = ShieldSwap(aleo)
    dex.journal = Journal(directory / "swaps.jsonl")   # blinded identities and claim handles
    dex.api.authenticate(str(account.address), lambda message: str(key.sign(message.encode())))
    return aleo, dex


def refuse_unresolved_work(aleo: Aleo, dex: ShieldSwap, directory: Path) -> None:
    """Stop before planning when an earlier run left reservations or claims unresolved."""
    journal = aleo.inventory.reconcile_record_inventory()
    pending = [entry for entry in journal if entry.status in UNRESOLVED]
    if pending:
        raise SystemExit(f"Reconcile {len(pending)} unresolved operations in {directory / 'inventory.json'} before another batch")
    if dex.journal is not None and dex.journal.pending_claims():
        raise SystemExit(f"Claim the pending swaps in {directory / 'swaps.jsonl'} (dex.collect_all()) before another batch")


def describe_plan(plan: InventoryPlan) -> None:
    print(f"{len(plan.steps)} maintenance operations; credits deduction {plan.deduction} microcredits")
    for step in plan.steps:
        helper = step.join_program.program if step.join_program else "native"
        print(f"  {step.kind:5} inputs={len(step.inputs):2} outputs={len(step.outputs)} via {helper}")


def concurrent_swaps(*, execute: bool, token_in: str = "USDCx", token_out: str = "ETH",
                     amount_in: str = "0.1", count: int = 6, directory: Path | None = None) -> list[SwapOutcome]:
    """Preview or run inventory maintenance followed by a concurrent swap batch."""
    private_key = os.environ.get("SHIELD_SWAP_PRIVATE_KEY")
    if not private_key:
        raise SystemExit("Set SHIELD_SWAP_PRIVATE_KEY to a funded testnet account with DEX access")
    if not 2 <= count <= 16:
        raise SystemExit("count must be from 2 through 16")
    address = str(testnet.PrivateKey.from_string(private_key).address)
    directory = directory or Path(".shield-swap") / "concurrent-swaps" / NETWORK / address
    directory.mkdir(parents=True, exist_ok=True)

    aleo, dex = build_client(private_key, directory)
    refuse_unresolved_work(aleo, dex, directory)

    token = dex.api.get_token(token_in)
    minimum = int(Decimal(amount_in).scaleb(token.decimals))
    # One record per trade lets every swap start at once. Balanced lots with
    # zero tolerance give each trade the same spendable amount.
    plan = dex.plan_inventory(token_in, InventoryTarget(records=count, min_record_amount=minimum,
                                                         distribution="balanced", tolerance_bps=0),
                              max_transactions=24)
    print(f"{count} concurrent {token_in} -> {token_out} swaps of {amount_in} {token_in}; recovery data in {directory}")
    describe_plan(plan)
    preview = dex.quote(token_in=token_in, token_out=token_out, amount_in=amount_in, slippage_bps=50)
    print(f"Per swap: estimated {preview.estimated_amount_out} {token_out}; minimum {preview.minimum_amount_out} {token_out}")
    if not execute:
        print("Preview only. Pass --execute to prepare the records, submit the swaps, and claim the outputs.")
        return []

    # A marker survives a crash so a later run does not repeat an unfinished batch.
    marker = directory / "batch.pending"
    if marker.exists():
        raise SystemExit(f"Resolve the previous batch before removing {marker}")
    marker.write_text("")
    result = aleo.inventory.rebalance_record_inventory(plan, timeout=600.0, poll_interval=3.0)
    if result.status != "complete":
        raise SystemExit(f"Maintenance interrupted: {result.reason}; keep {directory}")

    # Quote after maintenance so joins and splits do not consume its lifetime.
    # A quote fixes the output floor; it reserves no pool liquidity.
    quote = dex.quote(token_in=token_in, token_out=token_out, amount_in=amount_in, slippage_bps=50)
    outcomes = submit_concurrent_swaps(dex, quote, count)
    for outcome in outcomes:
        if outcome.claim is not None and outcome.handle is not None:
            print(f"{outcome.index}: {outcome.handle.transaction_id} -> {outcome.claim.transaction_id}; "
                  f"received {outcome.claim.amount_out} base units; refunded {outcome.claim.amount_remaining}")
        elif outcome.handle is not None:
            print(f"{outcome.index}: swap {outcome.handle.transaction_id} submitted; claim failed: {outcome.error}", file=sys.stderr)
        else:
            print(f"{outcome.index}: swap failed: {outcome.error}", file=sys.stderr)
    failures = [outcome for outcome in outcomes if outcome.claim is None]
    if failures:
        raise SystemExit(f"{len(failures)} trades need recovery in {directory}; do not repeat the batch")
    marker.unlink()
    return outcomes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare records and run concurrent Shield Swap trades on testnet.")
    parser.add_argument("--execute", action="store_true", help="prepare records, submit swaps, and claim outputs")
    parser.add_argument("--count", type=int, default=6, help="trades to run at once and records to prepare (2-16)")
    parser.add_argument("--amount", default="0.1", help="input amount per trade in token units")
    args = parser.parse_args()
    concurrent_swaps(execute=args.execute, count=args.count, amount_in=args.amount)
