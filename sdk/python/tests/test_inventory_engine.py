"""Inventory actions against a fake facade: scans, plans, execution, recovery, helpers, private fees.

_source: ProvableHQ/veil packages/core/test/inventory/{inventory,autojoinHelpers,
configuration,privateFees}.test.ts, PR #172 @ 2e324d5. Selection, planning,
reservation, and lifecycle handling run for real; only chain, scanner, and
prover I/O are replaced. Wallet ("rpc") cases are not ported.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any, Callable

import pytest

from aleo.facade.errors import ProgramNotFound, TransactionNotFound
from aleo.inventory import (
    AsyncRecordInventory, InventoryError, InventoryPlan, InventoryTarget, MemoryRecordInventoryStore, RecordAsset,
    RecordInventory, RecordManagementConfig, RecordReservationConflictError, RecordSpend, TokenJoinRouter,
    build_inventory_plan,
)
from aleo.inventory._engine import BuildTransaction, PreparedTransaction

FIXTURES = Path(__file__).parent / "fixtures"
OWNER = "aleo1owner"
SCOPE = '["testnet","aleo1owner"]'
CREDITS = RecordAsset("credits.aleo", standard="credits")
ARC20 = RecordAsset("test_arc20_eth.aleo", standard="arc20")
ARC22 = RecordAsset("test_usdcx_stablecoin.aleo", standard="arc22")
ROUTER = TokenJoinRouter("test_aj_arc20_2_15.aleo")
NONCE = re.compile(r"_nonce:\s*(\d+group)")


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


def field_of(asset: RecordAsset) -> tuple[str, str]:
    return ("microcredits", "u64") if asset.program == "credits.aleo" else ("amount", "u128")


def make_record(amount: int, nonce: int, asset: RecordAsset = CREDITS, extra: str = "") -> dict[str, Any]:
    field, width = field_of(asset)
    return {
        "program_name": asset.program, "record_name": "credits" if asset.program == "credits.aleo" else "Token",
        "tag": f"{nonce}field", "commitment": f"{nonce}field", "spent": False,
        "record_plaintext": f"{{ owner: {OWNER}.private, {field}: {amount}{width}.private, {extra}_nonce: {nonce}group.public }}",
    }


def static_helper(program: str, asset: str, record: str, low: int, high: int) -> str:
    out = f"import {asset};\nprogram {program};\n"
    for n in range(low, high + 1):
        out += f"function join_{n}:\n" + "".join(f"    input r{j} as {asset}/{record}.record;\n" for j in range(n))
        out += f"    output r{n} as {asset}/{record}.record;\n\n"
    return out


class Account:
    def __init__(self, address: str = OWNER) -> None:
        self.address = address


class _Bound:
    def __init__(self, world: "World", program: str, function: str, args: tuple[str, ...]) -> None:
        self.world, self.program, self.function, self.args = world, program, function, args

    def delegate(self, account: Any = None, *, broadcast: bool = True, pay_own_fee: bool = False,
                 fee_record: Any = None, priority_fee: int = 0) -> dict[str, Any]:
        assert broadcast is False
        return {"transaction": self.world.build(self.program, self.function, self.args, fee_record, priority_fee)}

    def build_transaction(self, account: Any = None, *, priority_fee: int = 0, fee_record: Any = None,
                          private_fee: bool = False, base_fee: int | None = None) -> Any:
        tx = self.world.build(self.program, self.function, self.args, fee_record if (private_fee or fee_record) else None, priority_fee)

        class _Raw:
            def to_json(self) -> str:
                return json.dumps(tx)

        class _Result:
            id = tx["id"]
            raw = _Raw()
        return _Result()


class _Functions:
    def __init__(self, world: "World", program: str) -> None:
        self.world, self.program = world, program

    def __getitem__(self, name: str) -> Callable[..., _Bound]:
        return lambda *args: _Bound(self.world, self.program, name, tuple(str(a) for a in args))


class _Program:
    def __init__(self, world: "World", program: str) -> None:
        self.source = world.sources[program]
        self.functions = _Functions(world, program)


class _Programs:
    def __init__(self, world: "World") -> None:
        self.world = world

    def get(self, program_id: str) -> _Program:
        self.world.lookups.append(program_id)
        if program_id not in self.world.sources:
            if self.world.lookup_error is not None:
                raise self.world.lookup_error
            raise ProgramNotFound(program_id)
        return _Program(self.world, program_id)


class _Provider:
    def __init__(self, world: "World") -> None:
        self.world = world

    def find(self, account: Any = None, *, program: str | None = None, unspent: bool = True, **_: Any) -> list[dict[str, Any]]:
        self.world.scans += 1
        # A real scanner returns fresh dicts on every scan; copying keeps a
        # later spend from mutating records a caller already holds.
        found = [dict(r) for r in self.world.records if r["program_name"] == program and not (unspent and r["spent"])]
        return self.world.scan_filter(found) if self.world.scan_filter else found


class _Network:
    def __init__(self, world: "World") -> None:
        self.world = world

    def submit_transaction(self, raw: Any) -> str:
        # The facade accepts a built result, a transaction, or a string; the
        # node client stringifies whatever it is given.
        return self.world.submit(json.loads(str(raw)))

    def get_confirmed_transaction(self, tx_id: str) -> dict[str, Any]:
        if self.world.on_confirmed is not None:
            return self.world.on_confirmed(tx_id)
        if tx_id not in self.world.transactions:
            raise TransactionNotFound(tx_id)
        return {"status": self.world.confirm_status, "transaction": self.world.transactions[tx_id]}


class _Records:
    def status(self) -> dict[str, Any]:
        return {"ok": True, "data": {"synced": True}}


class FakeAleo:
    network_name = "testnet"
    process = None

    def __init__(self, world: "World") -> None:
        self.default_account: Any = Account()
        self.programs = _Programs(world)
        self.record_provider = _Provider(world)
        self.network = _Network(world)
        self.records = _Records()


class World:
    """Scanner, node, and prover state behind one fake facade."""

    def __init__(self, amounts: list[int], asset: RecordAsset = CREDITS, token_join: TokenJoinRouter | bool | None = None, *,
                 enabled: bool = True, helpers: bool = False, store: MemoryRecordInventoryStore | None = None,
                 max_fee: int | None = None, network: str = "testnet", proving: str = "delegated") -> None:
        self.asset = asset
        self.records = [make_record(amount, index + 1, asset) for index, amount in enumerate(amounts)]
        # The node always serves credits.aleo; the engine never reads its source.
        self.sources: dict[str, str] = {"credits.aleo": "program credits.aleo;"}
        if asset.program != "credits.aleo":
            self.sources[asset.program] = fixture(asset.program)
        if isinstance(token_join, TokenJoinRouter):
            self.sources[token_join.program] = fixture("main_aj_arc20_2_15.aleo").replace("main_aj_arc20_2_15.aleo", token_join.program)
        if helpers:
            record = "credits" if asset.program == "credits.aleo" else "Token"
            prefix = "autojoin_credits" if asset.program == "credits.aleo" else f"aj_{asset.program[:-5]}"
            for low, high in ((2, 10), (11, 14), (15, 16)):
                helper = f"{prefix}_{low}_{high}.aleo"
                self.sources[helper] = static_helper(helper, asset.program, record, low, high)
            dynamic = f"{'main' if network == 'mainnet' else 'test'}_aj_arc20_2_15.aleo"
            self.sources[dynamic] = fixture("main_aj_arc20_2_15.aleo").replace("main_aj_arc20_2_15.aleo", dynamic)
        self.next_nonce = 2000
        self.sequence = 0
        self.transactions: dict[str, dict[str, Any]] = {}
        self.pending: dict[str, tuple[list[str], list[dict[str, Any]]]] = {}
        self.builds: list[dict[str, Any]] = []
        self.submits: list[dict[str, Any]] = []
        self.lookups: list[str] = []
        self.scans = 0
        self.submit_mode = "ok"
        self.confirm_status = "accepted"
        self.on_confirmed: Callable[[str], dict[str, Any]] | None = None
        self.scan_filter: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None
        self.lookup_error: Exception | None = None
        self.fail_proving = False
        self.wrong_fee_tag = False
        self.aleo = FakeAleo(self)
        self.aleo.network_name = network  # type: ignore[misc]
        config = RecordManagementConfig(enabled=enabled, store=store, token_join=token_join,  # type: ignore[arg-type]
                                        max_fee_microcredits=max_fee, chain_id=network if network in ("mainnet", "testnet") else "local:test")
        self.inventory = RecordInventory(self.aleo, config, proving=proving)  # type: ignore[arg-type]
        self.store = self.inventory.store
        self.scope = f'["{config.chain_id}","{OWNER}"]'

    # ── prover and node behaviour ────────────────────────────────────────────

    def _amount(self, plaintext: str) -> int:
        field, width = field_of(self.asset)
        match = re.search(rf"{field}:\s*(\d+){width}", plaintext)
        assert match, plaintext
        return int(match.group(1))

    def _by_nonce(self, nonce: str) -> dict[str, Any]:
        return next(r for r in self.records if NONCE.search(r["record_plaintext"]).group(1) == nonce)  # type: ignore[union-attr]

    def build(self, program: str, function: str, inputs: tuple[str, ...], fee_record: Any, priority_fee: int) -> dict[str, Any]:
        if self.fail_proving:
            raise RuntimeError("Proving failed")
        plaintexts = [item for item in inputs if item.lstrip().startswith("{")]
        nonces = [NONCE.search(item).group(1) for item in plaintexts]  # type: ignore[union-attr]
        deduction = 10_000 if self.asset.program == "credits.aleo" else 0
        if function.startswith("join"):
            amounts = [sum(self._amount(item) for item in plaintexts)]
        elif function == "split":
            first = int(re.sub(r"u(64|128)$", "", inputs[1]))
            amounts = [first, self._amount(plaintexts[0]) - first - deduction]
        else:
            amounts = []                                   # a transfer: inputs spent, no inventory outputs
        outputs: list[dict[str, Any]] = []
        for amount in amounts:
            outputs.append(make_record(amount, self.next_nonce, self.asset))
            self.next_nonce += 1
        self.sequence += 1
        tx_id = f"at1tx{self.sequence}"
        if function.startswith("join_"):
            dynamic = inputs[0].startswith("'")
            if dynamic:
                transitions = [
                    {"program": self.asset.program, "function": "join", "outputs": [{"type": "record_with_dynamic_id", "id": "intermediatefield", "dynamic_id": "intermediatedynamicfield"}]},
                    {"program": self.asset.program, "function": "join", "outputs": [{"type": "record_with_dynamic_id", "id": outputs[0]["commitment"], "dynamic_id": f"dynamic{self.sequence}field"}]},
                    {"program": program, "function": function, "outputs": [{"type": "record_dynamic", "id": f"dynamic{self.sequence}field"}]},
                ]
            else:
                transitions = [
                    {"program": self.asset.program, "function": "join", "outputs": [{"type": "record", "id": outputs[0]["commitment"]}]},
                    {"program": program, "function": function, "outputs": [{"type": "external_record", "id": "externalfield"}]},
                ]
        else:
            transitions = [{"program": program, "function": function,
                            "outputs": [{"type": "record", "id": record["commitment"]} for record in outputs]}]
        if callable(fee_record):
            chosen = fee_record(100 + priority_fee)
            fee_nonce = NONCE.search(chosen).group(1)  # type: ignore[union-attr]
            nonces.append(fee_nonce)
            tag = "999field" if self.wrong_fee_tag else self._by_nonce(fee_nonce)["tag"]
            fee = {"transition": {"program": "credits.aleo", "function": "fee_private", "inputs": [
                {"type": "record", "tag": tag}, {"type": "public", "value": "100u64"}, {"type": "public", "value": f"{priority_fee}u64"}]}}
        else:
            fee = {"transition": {"program": "credits.aleo", "function": "fee_public", "inputs": [
                {"type": "public", "value": "100u64"}, {"type": "public", "value": f"{priority_fee}u64"}]}}
        transaction = {"id": tx_id, "type": "execute", "execution": {"transitions": transitions}, "fee": fee}
        self.pending[tx_id] = (nonces, outputs)
        self.builds.append({"program": program, "function": function, "inputs": list(inputs), "tx": transaction})
        return transaction

    def submit(self, transaction: dict[str, Any]) -> str:
        if self.submit_mode == "offline":
            raise ConnectionError("offline before acceptance")
        if self.submit_mode == "rejected":
            from aleo import AleoNetworkError
            raise AleoNetworkError("POST /transaction/broadcast returned 400: stale state root", status=400)
        tx_id = str(transaction["id"])
        if tx_id in self.pending:
            nonces, outputs = self.pending.pop(tx_id)
            for record in self.records:
                if NONCE.search(record["record_plaintext"]).group(1) in nonces:  # type: ignore[union-attr]
                    record["spent"] = True
            self.records.extend(outputs)
            self.transactions[tx_id] = transaction
        self.submits.append(transaction)
        if self.submit_mode == "lost":
            raise ConnectionError("response lost")
        return tx_id

    # ── helpers for assertions ───────────────────────────────────────────────

    def journal(self) -> list[RecordSpend]:
        assert self.store is not None
        return self.store.list(self.scope)

    def hold(self, *nonces: str, fee: str | None = None, status: str = "reserved", account_type: str = "local") -> RecordSpend | None:
        assert self.store is not None
        return self.store.acquire(RecordSpend(id="", scope=self.scope, records=nonces, program="credits.aleo", function="join",
                                              status=status, created_at=1, fee_record_id=fee, account_type=account_type))  # type: ignore[arg-type]

    def available_amounts(self, asset: RecordAsset | None = None) -> list[int]:
        return sorted(r.amount for r in self.inventory.get_record_inventory(asset or self.asset).available)


FAST = {"timeout": 2.0, "poll_interval": 0.001}


# ── Scanner-backed inventory ──────────────────────────────────────────────────

def test_deduplicates_identities_across_scanner_results() -> None:
    w = World([100] * 1001)
    w.records.append(dict(w.records[0]))                      # the scanner repeats a record
    inventory = w.inventory.get_record_inventory(CREDITS)
    assert len(inventory.available) == 1001 and inventory.balance == 100_100


@pytest.mark.parametrize("asset", [ARC20, ARC22])
def test_filters_compliance_and_recipient_bound_records(asset: RecordAsset) -> None:
    w = World([10], asset)
    w.records.append(make_record(20, 2, asset, "recipient_bound: true.private, "))
    w.records.append({**make_record(30, 3, asset, f"sender: aleo1sender.private, recipient: {OWNER}.private, "), "record_name": "ComplianceRecord"})
    assert w.inventory.get_record_inventory(asset).balance == 10


def test_refuses_records_without_plaintext_rather_than_claiming_an_empty_inventory() -> None:
    w = World([10])
    w.records = [{"program_name": "credits.aleo", "record_name": "credits", "tag": "1field", "spent": False}]
    with pytest.raises(InventoryError, match="plaintext"):
        w.inventory.get_record_inventory(CREDITS)


# ── Execution and recovery ────────────────────────────────────────────────────

@pytest.mark.parametrize("asset", [CREDITS, ARC20, ARC22])
def test_executes_a_dependent_split_plan_and_reaches_an_idempotent_target(asset: RecordAsset) -> None:
    w = World([1_030_000], asset)
    target = InventoryTarget(records=4, distribution="balanced")
    plan = w.inventory.plan_record_inventory(asset, target)
    result = w.inventory.rebalance_record_inventory(plan, **FAST)
    assert (result.status, result.completed_steps) == ("complete", 3)
    assert w.inventory.plan_record_inventory(asset, target).steps == ()
    assert all(entry.status == "confirmed" for entry in w.journal())
    assert all(entry.transaction is None for entry in w.journal())


def test_autojoins_only_enough_records_and_retains_the_rest() -> None:
    w = World([40, 30, 10], ARC20)
    result = w.inventory.auto_join(ARC20, min_amount=60, **FAST)
    assert result.completed_steps == 1
    assert w.available_amounts() == [10, 70]
    assert w.inventory.auto_join(ARC20, min_amount=60, **FAST).completed_steps == 0


def test_atomically_rejects_competing_spending_before_the_second_proof() -> None:
    w = World([1_000_000])
    record = w.inventory.get_record_inventory(CREDITS).available[0]
    outcomes: list[Any] = []
    barrier = threading.Barrier(2)

    def attempt(amount: int) -> None:
        barrier.wait()
        try:
            outcomes.append(w.inventory.split_record(CREDITS, record, amount))
        except RecordReservationConflictError as exc:
            outcomes.append(exc)

    threads = [threading.Thread(target=attempt, args=(amount,)) for amount in (100_000, 200_000)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(isinstance(item, str) for item in outcomes) == 1
    assert sum(isinstance(item, RecordReservationConflictError) for item in outcomes) == 1
    assert len(w.builds) == 1


def test_enforces_the_fee_ceiling_before_broadcast_and_releases_the_build() -> None:
    w = World([1_000_000])
    plan = w.inventory.plan_record_inventory(CREDITS, InventoryTarget(records=2))
    result = w.inventory.rebalance_record_inventory(plan, max_fee_microcredits=99, **FAST)
    assert result.status == "interrupted" and "max_fee_microcredits" in str(result.reason)
    assert w.submits == []
    assert w.journal()[0].status == "cancelled"


def test_retains_a_prepared_transaction_when_the_broadcast_response_is_lost_then_reconciles() -> None:
    w = World([1_000_000])
    w.submit_mode = "lost"
    plan = w.inventory.plan_record_inventory(CREDITS, InventoryTarget(records=2))
    result = w.inventory.rebalance_record_inventory(plan, **FAST)
    assert (result.status, result.transaction_ids) == ("interrupted", ("at1tx1",))
    assert (w.journal()[0].status, w.journal()[0].transaction_id) == ("prepared", "at1tx1")
    w.inventory.reconcile_record_inventory()
    assert w.journal()[0].status == "confirmed"


def test_rebroadcasts_the_exact_saved_transaction_without_rebuilding_its_proof() -> None:
    w = World([1_000_000])
    w.submit_mode = "offline"
    plan = w.inventory.plan_record_inventory(CREDITS, InventoryTarget(records=2))
    w.inventory.rebalance_record_inventory(plan, **FAST)
    saved = w.journal()[0]
    w.inventory.reconcile_record_inventory()                      # a missing transaction keeps the reservation
    assert w.journal()[0].status == "prepared"
    w.submit_mode = "ok"
    w.inventory.reconcile_record_inventory(rebroadcast=True)
    assert w.submits[-1] == saved.transaction and len(w.builds) == 1
    w.inventory.reconcile_record_inventory()
    assert (w.journal()[0].status, w.journal()[0].transaction) == ("confirmed", None)


def test_releases_inputs_only_after_a_confirmed_rejection() -> None:
    w = World([1_000_000])
    w.confirm_status = "rejected"
    plan = w.inventory.plan_record_inventory(CREDITS, InventoryTarget(records=2))
    assert w.inventory.rebalance_record_inventory(plan, **FAST).status == "interrupted"
    assert w.journal()[0].status == "rejected"
    w.records = [r for r in w.records if r["commitment"] == "1field"]     # the chain kept the input unspent
    w.records[0]["spent"] = False
    assert len(w.inventory.get_record_inventory(CREDITS).available) == 1


def test_does_not_confuse_a_new_deposit_with_the_expected_outputs() -> None:
    w = World([1_000_000])
    plan = w.inventory.plan_record_inventory(CREDITS, InventoryTarget(records=2))
    w.scan_filter = lambda found: found if any(r["commitment"] == "1field" and not r["spent"] for r in found) else [make_record(1_000_000, 9999)]
    result = w.inventory.rebalance_record_inventory(plan, timeout=0.05, poll_interval=0.001)
    assert (result.status, result.completed_steps) == ("interrupted", 0)
    assert "Scanner" in str(result.reason) and len(w.builds) == 1


def test_retains_uncertain_inputs_on_a_confirmation_outage() -> None:
    w = World([1_000_000])
    plan = w.inventory.plan_record_inventory(CREDITS, InventoryTarget(records=2))

    def outage(tx_id: str) -> dict[str, Any]:
        raise ConnectionError("network offline")

    w.on_confirmed = outage
    assert w.inventory.rebalance_record_inventory(plan, **FAST).status == "interrupted"
    with pytest.raises(ConnectionError, match="network offline"):
        w.inventory.reconcile_record_inventory()
    assert w.journal()[0].status == "submitted"


def test_rejects_stale_inputs_modified_plans_and_cancellation() -> None:
    w = World([1_000_000])
    plan = w.inventory.plan_record_inventory(CREDITS, InventoryTarget(records=2))
    altered = InventoryPlan.from_dict({**plan.to_dict(), "steps": [{**s, "amount": "1"} for s in plan.to_dict()["steps"]]})
    altered = InventoryPlan(**{**altered.__dict__, "asset": plan.asset})
    with pytest.raises(InventoryError, match="modified"):
        w.inventory.rebalance_record_inventory(altered, **FAST)
    cancel = threading.Event()
    cancel.set()
    assert w.inventory.rebalance_record_inventory(plan, cancel=cancel, **FAST).status == "interrupted"
    w.records = []
    assert "changed" in str(w.inventory.rebalance_record_inventory(plan, **FAST).reason)
    assert w.builds == []


def test_plan_bound_to_another_account_or_chain_is_refused() -> None:
    w = World([1_000_000])
    plan = w.inventory.plan_record_inventory(CREDITS, InventoryTarget(records=2))
    foreign = InventoryPlan(**{**plan.__dict__, "scope": '["mainnet","aleo1owner"]'})
    with pytest.raises(InventoryError, match="another account or chain"):
        w.inventory.rebalance_record_inventory(foreign, **FAST)


def test_default_account_change_mid_run_interrupts_and_keeps_reservations() -> None:
    w = World([1_000_000])
    plan = w.inventory.plan_record_inventory(CREDITS, InventoryTarget(records=2))

    def swap_account(tx_id: str) -> dict[str, Any]:
        w.aleo.default_account = Account("aleo1other")
        return {"status": "accepted", "transaction": w.transactions[tx_id]}

    w.on_confirmed = swap_account
    result = w.inventory.rebalance_record_inventory(plan, **FAST)
    assert result.status == "interrupted" and "changed" in str(result.reason)
    assert w.journal()[0].status == "confirmed"


# ── Configured token batch joins ──────────────────────────────────────────────

def test_forwards_the_public_identifier_and_dynamic_import_for_routed_joins() -> None:
    w = World([100, 100, 100], ARC22, ROUTER)
    records = w.inventory.get_record_inventory(ARC22).available
    w.inventory.join_records(ARC22, records)
    build = w.builds[0]
    assert (build["program"], build["function"]) == (ROUTER.program, "join_3")
    assert build["inputs"] == ["'test_usdcx_stablecoin'", *(r.plaintext for r in records)]
    entry = w.journal()[0]
    assert (entry.records, entry.asset_program, entry.status, entry.program) == (("1group", "2group", "3group"), ARC22.program, "submitted", ROUTER.program)


@pytest.mark.parametrize("count", [2, 15, 16, 30])
def test_consolidates_records_in_bounded_router_transactions(count: int) -> None:
    w = World([100] * count, ARC20, ROUTER)
    plan = w.inventory.plan_record_inventory(ARC20, InventoryTarget(records=1), max_transactions=3)
    assert plan.token_join == ROUTER
    assert len(plan.steps) == -(-(count - 1) // 14)
    result = w.inventory.rebalance_record_inventory(plan, **FAST)
    assert (result.status, result.completed_steps) == ("complete", len(plan.steps))
    first = min(count, 15)
    assert (w.builds[0]["program"], w.builds[0]["function"]) == (ROUTER.program, f"join_{first}")
    assert w.inventory.get_record_inventory(ARC20).balance == count * 100
    entry = w.journal()[0]
    assert (entry.program, entry.asset_program, entry.function, entry.status) == (ROUTER.program, ARC20.program, f"join_{first}", "confirmed")
    assert entry.records == plan.steps[0].inputs


@pytest.mark.parametrize("asset", [ARC20, ARC22])
def test_preserves_target_lots_then_executes_batch_joins_followed_by_native_splits(asset: RecordAsset) -> None:
    w = World([100] * 16, asset, ROUTER)
    preserve = w.inventory.plan_record_inventory(asset, InventoryTarget(records=4))
    assert len(preserve.steps) == 1 and len(preserve.steps[0].inputs) == 13
    plan = w.inventory.plan_record_inventory(asset, InventoryTarget(records=4, distribution="balanced"))
    assert [step.kind for step in plan.steps] == ["join", "join", "split", "split", "split"]
    assert w.inventory.rebalance_record_inventory(plan, **FAST).completed_steps == 5
    assert w.available_amounts(asset) == [400, 400, 400, 400]
    assert w.inventory.plan_record_inventory(asset, plan.target).steps == ()


def test_retains_sufficient_record_selection_and_native_credits_joins() -> None:
    w = World([40, 30, 20, 10], ARC20, ROUTER)
    assert w.inventory.auto_join(ARC20, min_amount=80, **FAST).completed_steps == 1
    assert w.builds[0]["function"] == "join_3"
    assert w.available_amounts() == [10, 90]
    native = World([100, 100, 100], CREDITS, ROUTER)
    plan = native.inventory.plan_record_inventory(CREDITS, InventoryTarget(records=1))
    assert plan.token_join is None and [len(step.inputs) for step in plan.steps] == [2, 2]
    assert native.inventory.rebalance_record_inventory(plan, **FAST).status == "complete"
    assert all(b["program"] == "credits.aleo" and b["function"] == "join" for b in native.builds)


def test_keeps_batch_reservations_atomic_and_enforces_the_proved_fee_limit() -> None:
    w = World([100, 100, 100], ARC20, ROUTER)
    records = w.inventory.get_record_inventory(ARC20).available
    w.inventory.join_records(ARC20, records)
    with pytest.raises(RecordReservationConflictError) as conflict:
        w.inventory.join_records(ARC20, records[:2])
    assert conflict.value.record_ids == ("1group", "2group")
    assert len(w.builds) == 1
    limited = World([100, 100, 100], ARC20, ROUTER)
    plan = limited.inventory.plan_record_inventory(ARC20, InventoryTarget(records=1))
    result = limited.inventory.rebalance_record_inventory(plan, max_fee_microcredits=99, **FAST)
    assert result.status == "interrupted" and "max_fee_microcredits" in str(result.reason)
    assert limited.submits == []
    assert (limited.journal()[0].status, limited.journal()[0].records) == ("cancelled", ("1group", "2group", "3group"))


def test_rebroadcasts_an_interrupted_batch_proof_and_replans_dependent_inventory() -> None:
    w = World([100] * 16, ARC22, ROUTER)
    w.submit_mode = "offline"
    plan = w.inventory.plan_record_inventory(ARC22, InventoryTarget(records=1))
    assert w.inventory.rebalance_record_inventory(plan, **FAST).completed_steps == 0
    saved = w.journal()[0]
    assert len(saved.records) == 15
    w.submit_mode = "ok"
    w.inventory.reconcile_record_inventory(rebroadcast=True)
    assert w.submits[-1] == saved.transaction
    w.inventory.reconcile_record_inventory()
    assert w.inventory.auto_join(ARC22, **FAST).completed_steps == 1
    assert len(w.builds) == 2
    assert w.inventory.get_record_inventory(ARC22).balance == 1600


def test_rejects_incompatible_router_code_duplicates_overflow_and_excessive_sizes_before_proving() -> None:
    w = World([100] * 16, ARC20, ROUTER)
    records = w.inventory.get_record_inventory(ARC20).available
    with pytest.raises(InventoryError, match="2 to 15"):
        w.inventory.join_records(ARC20, records)
    with pytest.raises(InventoryError, match="distinct"):
        w.inventory.join_records(ARC20, [records[0], records[0]])
    huge = World([(1 << 128) - 1, 1], ARC20, ROUTER)
    with pytest.raises(InventoryError, match="overflows"):
        huge.inventory.join_records(ARC20, huge.inventory.get_record_inventory(ARC20).available)
    w.sources[ROUTER.program] = w.sources[ROUTER.program].replace("identifier.public", "identifier.private")
    with pytest.raises(InventoryError, match="incompatible"):
        w.inventory.join_records(ARC20, records[:2])
    assert w.builds == [] and huge.builds == []


def test_rejects_a_mismatched_dynamic_output_instead_of_selecting_intermediate_records() -> None:
    w = World([100, 100, 100], ARC20, ROUTER)

    def tampered(tx_id: str) -> dict[str, Any]:
        tx = json.loads(json.dumps(w.transactions[tx_id]))
        tx["execution"]["transitions"][-2]["outputs"][0]["type"] = "record"
        return {"status": "accepted", "transaction": tx}

    w.on_confirmed = tampered
    plan = w.inventory.plan_record_inventory(ARC20, InventoryTarget(records=1))
    result = w.inventory.rebalance_record_inventory(plan, **FAST)
    assert result.status == "interrupted" and "underlying token commitment" in str(result.reason)
    assert w.journal()[0].status == "confirmed" and len(w.builds) == 1


def test_honors_custom_batch_ceilings_and_executes_plans_built_without_a_router() -> None:
    w = World([100] * 4, ARC20, TokenJoinRouter(ROUTER.program, max_records=3))
    plan = w.inventory.plan_record_inventory(ARC20, InventoryTarget(records=1))
    assert [len(step.inputs) for step in plan.steps] == [3, 2]
    native = build_inventory_plan(scope=w.scope, asset=plan.asset, inputs=plan.inputs, target=plan.target)
    assert w.inventory.rebalance_record_inventory(native, **FAST).completed_steps == 3
    assert all(b["program"] == ARC20.program for b in w.builds)


# ── Automatic helper selection ────────────────────────────────────────────────

@pytest.mark.parametrize("count,helper", [(3, "autojoin_credits_2_10.aleo"), (11, "autojoin_credits_11_14.aleo"),
                                          (15, "autojoin_credits_15_16.aleo"), (16, "autojoin_credits_15_16.aleo")])
def test_uses_one_helper_transaction_for_credits_records(count: int, helper: str) -> None:
    w = World([100] * count, helpers=True)
    plan = w.inventory.plan_record_inventory(RecordAsset("credits.aleo"), InventoryTarget(records=1), max_transactions=1)
    assert len(plan.steps) == 1
    assert w.inventory.rebalance_record_inventory(plan, **FAST).completed_steps == 1
    assert (w.builds[0]["program"], w.builds[0]["function"], len(w.builds[0]["inputs"])) == (helper, f"join_{count}", count)
    assert w.inventory.get_record_inventory(CREDITS).balance == count * 100
    assert (w.journal()[0].asset_program, w.journal()[0].status) == ("credits.aleo", "confirmed")


def test_chains_a_sixteen_record_helper_output_into_a_native_join() -> None:
    w = World([100] * 17, helpers=True)
    assert w.inventory.auto_join(RecordAsset("credits.aleo"), max_transactions=2, **FAST).completed_steps == 2
    assert [(b["program"], b["function"]) for b in w.builds] == [("autojoin_credits_15_16.aleo", "join_16"), ("credits.aleo", "join")]
    assert any("_nonce: 2000group" in item for item in w.builds[1]["inputs"])


def test_prefers_a_dedicated_stablecoin_helper_over_a_generic_arc_helper() -> None:
    w = World([10, 20, 30], ARC22, helpers=True)
    asset = RecordAsset("test_usdcx_stablecoin.aleo")
    w.inventory.join_records(asset, w.inventory.get_record_inventory(asset).available)
    assert (w.builds[0]["program"], w.builds[0]["function"]) == ("aj_test_usdcx_stablecoin_2_10.aleo", "join_3")
    assert all(item.lstrip().startswith("{") for item in w.builds[0]["inputs"])


@pytest.mark.parametrize("network", ["mainnet", "testnet"])
def test_uses_the_dynamic_arc_helper_with_a_program_identifier(network: str) -> None:
    w = World([10, 20, 30], ARC20, helpers=True, network=network)
    asset = RecordAsset("test_arc20_eth.aleo")
    assert w.inventory.auto_join(asset, max_transactions=1, **FAST).completed_steps == 1
    build = w.builds[0]
    assert (build["program"], build["function"]) == (f"{'main' if network == 'mainnet' else 'test'}_aj_arc20_2_15.aleo", "join_3")
    assert build["inputs"][0] == "'test_arc20_eth'" and len(build["inputs"]) == 4


@pytest.mark.parametrize("options", [{"helpers": False}, {"helpers": True, "token_join": False}, {"helpers": True, "network": "local"}])
def test_falls_back_to_native_joins_for_unavailable_disabled_or_unknown_network_helpers(options: dict[str, Any]) -> None:
    w = World([100, 100, 100], **options)
    assert w.inventory.auto_join(RecordAsset("credits.aleo"), **FAST).completed_steps == 2
    assert [b["program"] for b in w.builds] == ["credits.aleo", "credits.aleo"]


def test_rejects_a_mismatched_external_record_abi_before_choosing_a_helper() -> None:
    w = World([100, 100, 100], helpers=True)
    w.sources["autojoin_credits_2_10.aleo"] = static_helper("autojoin_credits_2_10.aleo", "other.aleo", "credits", 2, 10)
    plan = w.inventory.plan_record_inventory(RecordAsset("credits.aleo"), InventoryTarget(records=1))
    assert [len(step.inputs) for step in plan.steps] == [2, 2]


def test_surfaces_helper_lookup_outages_without_submitting_native_transactions() -> None:
    from aleo import AleoNetworkError
    w = World([100, 100, 100], helpers=True)
    w.sources = {}
    w.lookup_error = AleoNetworkError("Gateway unavailable", status=503)
    with pytest.raises(AleoNetworkError, match="Gateway unavailable"):
        w.inventory.auto_join(RecordAsset("credits.aleo"), **FAST)
    assert w.builds == []


def test_keeps_the_helper_chosen_during_planning_and_rejects_a_changed_abi_before_submission() -> None:
    w = World([100, 100, 100], helpers=True)
    plan = w.inventory.plan_record_inventory(RecordAsset("credits.aleo"), InventoryTarget(records=1))
    w.sources["autojoin_credits_2_10.aleo"] = static_helper("autojoin_credits_2_10.aleo", "other.aleo", "credits", 2, 10)
    result = w.inventory.rebalance_record_inventory(plan, **FAST)
    assert (result.status, result.completed_steps) == ("interrupted", 0) and "incompatible" in str(result.reason)
    assert w.builds == []


def test_uses_helpers_for_an_explicit_join_without_enabling_coordination() -> None:
    w = World([100, 100, 100], helpers=True, enabled=False)
    asset = RecordAsset("credits.aleo")
    w.inventory.join_records(asset, w.inventory.get_record_inventory(asset).available)
    assert w.builds[0]["function"] == "join_3"
    assert w.inventory.store is None and w.inventory.reserved_record_ids() == set()


# ── Configuration ─────────────────────────────────────────────────────────────

def test_disabled_management_allows_reads_plans_and_single_writes_but_not_execution() -> None:
    w = World([100_000, 100_000], enabled=False)
    inventory = w.inventory.get_record_inventory(CREDITS)
    assert len(inventory.available) == 2 and inventory.reserved == ()
    plan = w.inventory.plan_record_inventory(CREDITS, InventoryTarget(records=1))
    assert plan.scope == SCOPE and len(plan.steps) == 1
    assert w.inventory.join_records(CREDITS, inventory.available) == "at1tx1"
    assert w.inventory.split_record(CREDITS, w.inventory.get_record_inventory(CREDITS).available[0], 10_000) == "at1tx2"
    with pytest.raises(InventoryError, match="record management"):
        w.inventory.rebalance_record_inventory(plan, **FAST)
    with pytest.raises(InventoryError, match="record management"):
        w.inventory.reconcile_record_inventory()


def test_enabled_configuration_defaults_to_a_private_in_memory_store_and_validates_settings() -> None:
    w = World([1])
    assert isinstance(w.inventory.store, MemoryRecordInventoryStore)
    with pytest.raises(InventoryError, match="negative"):
        RecordInventory(w.aleo, RecordManagementConfig(enabled=True, max_fee_microcredits=-1))
    with pytest.raises(InventoryError, match="2 to 15"):
        RecordInventory(w.aleo, RecordManagementConfig(token_join=TokenJoinRouter(ROUTER.program, 16)))
    with pytest.raises(ValueError, match="proving"):
        RecordInventory(w.aleo, proving="remote")  # type: ignore[arg-type]


def test_two_clients_sharing_a_store_see_each_others_reservations() -> None:
    store = MemoryRecordInventoryStore()
    a = World([100, 100], store=store)
    b = World([100, 100], store=store)
    b.records = a.records
    with a.inventory.reserve("credits.aleo", "transfer_private", ["1group", "2group"]):
        snapshot = b.inventory.get_record_inventory(CREDITS)
        assert snapshot.available == () and [r.id for r in snapshot.reserved] == ["1group", "2group"]
        assert b.inventory.reserved_record_ids() == {"1group", "2group"}
        with pytest.raises(RecordReservationConflictError):
            b.inventory.join_records(CREDITS, snapshot.reserved)
    assert b.inventory.reserved_record_ids() == set()        # released: nothing was prepared


def test_reserve_context_manager_journals_a_self_managed_write() -> None:
    w = World([1_000_000])
    with w.inventory.reserve("credits.aleo", "transfer_private", ["1group"]) as checkpoint:
        tx = w.build("credits.aleo", "transfer_private", (w.records[0]["record_plaintext"],), None, 0)
        checkpoint.prepared(tx["id"], tx)
        checkpoint.submitted(w.submit(tx))
    assert (w.journal()[0].status, w.journal()[0].fee_microcredits) == ("submitted", "100")
    with pytest.raises(RecordReservationConflictError):
        with w.inventory.reserve("credits.aleo", "join", ["1group"]):
            pass
    with pytest.raises(RuntimeError):
        with w.inventory.reserve("credits.aleo", "join", ["2000group"]):
            raise RuntimeError("proving failed")
    assert w.journal()[-1].status == "cancelled"


# ── Managed private fees ──────────────────────────────────────────────────────

def private_fee_world(amounts: list[int] | None = None, max_fee: int | None = 150) -> World:
    return World(amounts if amounts is not None else [500, 110, 200], max_fee=max_fee)


def managed(w: World, *, fee_record: str | None = None, priority: int = 0, inputs: list[str] | None = None) -> str:
    spend = inputs if inputs is not None else [w.records[0]["record_plaintext"], "aleo1recipient", "1u64"]

    def builder(request: BuildTransaction) -> PreparedTransaction:
        tx = w.build(request.program, request.function, request.inputs, request.fee_record, priority)
        return PreparedTransaction(tx["id"], tx)

    return w.inventory.managed_write("credits.aleo", "transfer_private", spend, private_fee=True, fee_record=fee_record, builder=builder)


def spends(w: World) -> list[RecordSpend]:
    """Journal entries that are spends, without the short-lived claims that win a fee record."""
    return [entry for entry in w.journal() if entry.status != "cancelled" or entry.transaction_id is not None]


def test_selects_the_fee_record_and_journals_it_in_the_same_entry_as_the_inputs() -> None:
    w = private_fee_world()
    managed(w, priority=20)
    (entry,) = spends(w)
    # One entry holds inputs and fee, as Veil writes it, so a crash between
    # checkpoints can never keep the transaction while releasing its fee record.
    assert (entry.records, entry.fee_record_id, entry.fee_microcredits, entry.status) == (("1group", "3group"), "3group", "120", "submitted")
    assert "microcredits:" not in json.dumps([entry.to_dict() for entry in w.journal()])
    claims = [entry for entry in w.journal() if entry not in spends(w)]
    assert [(claim.records, claim.status) for claim in claims] == [(("3group",), "cancelled")]


def test_uses_a_supplied_fee_record_when_eligible_and_replaces_a_reserved_or_input_one() -> None:
    w = private_fee_world()
    managed(w, fee_record=w.records[2]["record_plaintext"])
    assert spends(w)[-1].fee_record_id == "3group"
    w = private_fee_world()
    w.hold("3group", status="prepared")
    managed(w, fee_record=w.records[2]["record_plaintext"])
    assert spends(w)[-1].fee_record_id == "2group"
    w = private_fee_world()
    managed(w, fee_record=w.records[0]["record_plaintext"])
    assert spends(w)[-1].fee_record_id == "2group"


def test_chooses_disjoint_fee_records_when_independent_writes_race() -> None:
    w = private_fee_world([100, 100])
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def run() -> None:
        barrier.wait()
        try:
            managed(w, inputs=["aleo1recipient", "1u64"])
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert {entry.fee_record_id for entry in spends(w)} == {"1group", "2group"}


def test_excludes_reserved_records_and_fails_before_proving_when_nothing_covers_the_fee() -> None:
    w = private_fee_world([500, 100, 99])
    w.hold("2group", status="prepared")
    with pytest.raises(InventoryError, match="private fee") as error:
        managed(w)
    assert not isinstance(error.value, RecordReservationConflictError)
    assert w.submits == [] and len(w.journal()) == 2 and w.journal()[1].status == "cancelled"


def test_reports_input_contention_separately_from_private_fee_exhaustion() -> None:
    w = private_fee_world()
    w.hold("1group", status="prepared")
    with pytest.raises(RecordReservationConflictError) as conflict:
        managed(w)
    assert conflict.value.record_ids == ("1group",)
    assert w.builds == [] and len(w.journal()) == 1


def test_releases_both_inputs_when_proving_fails_before_a_prepared_transaction_exists() -> None:
    w = private_fee_world()
    w.fail_proving = True
    with pytest.raises(RuntimeError, match="Proving failed"):
        managed(w)
    assert [entry.status for entry in w.journal()] == ["cancelled"]
    assert w.inventory.reserved_record_ids() == set()


def test_rejects_a_prover_that_spends_a_different_fee_record_before_broadcast() -> None:
    w = private_fee_world()
    w.wrong_fee_tag = True
    with pytest.raises(InventoryError, match="reserved private fee record"):
        managed(w)
    assert w.submits == [] and {entry.status for entry in w.journal()} == {"cancelled"}


def test_retains_the_spent_private_fee_after_a_rejected_execution_while_releasing_its_other_inputs() -> None:
    w = private_fee_world()
    managed(w)
    w.confirm_status = "rejected"
    w.inventory.reconcile_record_inventory()
    assert w.inventory.reserved_record_ids() == {"2group"}
    assert w.hold("2group") is None and w.hold("1group") is not None


def test_preserves_both_reservations_when_the_broadcast_outcome_is_unknown() -> None:
    w = private_fee_world()
    w.submit_mode = "offline"
    with pytest.raises(ConnectionError, match="offline"):
        managed(w)
    assert [(entry.status, entry.records) for entry in spends(w)] == [("prepared", ("1group", "2group"))]
    assert w.inventory.reserved_record_ids() == {"1group", "2group"}


def test_the_fee_record_is_never_held_apart_from_the_inputs_once_the_transaction_is_saved() -> None:
    """Every checkpoint after fee selection is one store update (the reviewer's crash probe)."""
    w = private_fee_world()
    updates: list[tuple[str, dict[str, Any]]] = []
    store = w.store
    assert store is not None
    original = store.update

    def record_update(scope: str, entry_id: str, patch: Any) -> None:
        updates.append((entry_id, dict(patch)))
        original(scope, entry_id, patch)

    store.update = record_update  # type: ignore[method-assign]
    managed(w)
    (entry,) = spends(w)
    # From the prepared checkpoint on, only the spend's own entry is touched,
    # and it already carries the fee record when the transaction is saved.
    after_fee = [(entry_id, patch) for entry_id, patch in updates if "transaction" in patch or patch.get("status") == "submitted"]
    assert after_fee and all(entry_id == entry.id for entry_id, _ in after_fee)
    assert entry.records == ("1group", "2group") and entry.fee_record_id == "2group"


def test_a_nested_conflict_after_acquisition_is_not_reported_as_retryable() -> None:
    w = World([1_000_000, 500])
    record = w.inventory.get_record_inventory(CREDITS).available[0]

    def builder(request: BuildTransaction) -> PreparedTransaction:
        # A caller callback runs another managed write on the same input.
        w.inventory.split_record(CREDITS, record, 100)
        raise AssertionError("unreachable")

    with pytest.raises(InventoryError, match="after acquiring inputs") as error:
        w.inventory.managed_write("credits.aleo", "split", [record.plaintext, "100u64"], builder=builder)
    assert isinstance(error.value.__cause__, RecordReservationConflictError)


# ── Async twin ────────────────────────────────────────────────────────────────

class AsyncFakeAleo:
    """Awaitable view over the same world state."""

    def __init__(self, fake: FakeAleo) -> None:
        self._fake = fake
        self.network_name = fake.network_name
        self.process = None
        outer = self

        class _AsyncBound:
            def __init__(self, bound: _Bound) -> None:
                self._bound = bound

            async def delegate(self, account: Any = None, **kw: Any) -> dict[str, Any]:
                return self._bound.delegate(account, **kw)

            async def build_transaction(self, account: Any = None, **kw: Any) -> Any:
                return self._bound.build_transaction(account, **kw)

        class _AsyncFunctions:
            def __init__(self, functions: _Functions) -> None:
                self._functions = functions

            def __getitem__(self, name: str) -> Callable[..., _AsyncBound]:
                return lambda *args: _AsyncBound(self._functions[name](*args))

        class _AsyncProgram:
            def __init__(self, program: _Program) -> None:
                self.source = program.source
                self.functions = _AsyncFunctions(program.functions)

        class _P:
            async def get(self, pid: str) -> _AsyncProgram:
                return _AsyncProgram(outer._fake.programs.get(pid))

        class _Prov:
            async def find(self, account: Any = None, **kw: Any) -> list[dict[str, Any]]:
                return outer._fake.record_provider.find(account, **kw)

        class _N:
            async def submit_transaction(self, raw: Any) -> str:
                return outer._fake.network.submit_transaction(raw)

            async def get_confirmed_transaction(self, tx_id: str) -> dict[str, Any]:
                return outer._fake.network.get_confirmed_transaction(tx_id)

        class _R:
            async def status(self) -> dict[str, Any]:
                return outer._fake.records.status()

        self.programs, self.record_provider, self.network, self.records = _P(), _Prov(), _N(), _R()

    @property
    def default_account(self) -> Any:
        return self._fake.default_account


@pytest.mark.asyncio
async def test_async_client_runs_the_same_engine() -> None:
    w = World([1_030_000], ARC20, ROUTER)
    inventory = AsyncRecordInventory(AsyncFakeAleo(w.aleo), RecordManagementConfig(enabled=True, store=w.store, token_join=ROUTER))
    plan = await inventory.plan_record_inventory(ARC20, InventoryTarget(records=4, distribution="balanced"))
    result = await inventory.rebalance_record_inventory(plan, **FAST)
    assert (result.status, result.completed_steps) == ("complete", 3)
    assert (await inventory.plan_record_inventory(ARC20, plan.target)).steps == ()
    journal = await inventory.reconcile_record_inventory()
    assert all(entry.status == "confirmed" for entry in journal)
    w.records = []
    assert (await inventory.get_record_inventory(ARC20)).available == ()


# ── Review follow-ups ─────────────────────────────────────────────────────────

def test_a_node_rejected_broadcast_releases_the_inputs_while_a_lost_response_keeps_them() -> None:
    w = World([1_000_000])
    w.submit_mode = "rejected"
    plan = w.inventory.plan_record_inventory(CREDITS, InventoryTarget(records=2))
    result = w.inventory.rebalance_record_inventory(plan, **FAST)
    assert result.status == "interrupted" and "400" in str(result.reason)
    assert w.journal()[0].status == "cancelled" and w.inventory.reserved_record_ids() == set()
    w.submit_mode = "lost"
    assert w.inventory.rebalance_record_inventory(plan, **FAST).status == "interrupted"
    assert w.journal()[-1].status == "prepared"


def test_reconcile_releases_reserved_entries_only_on_request() -> None:
    w = World([1_000_000])
    w.hold("1group")                                        # a crashed writer that never proved
    assert w.inventory.reserved_record_ids() == {"1group"}
    w.inventory.reconcile_record_inventory()
    assert w.journal()[0].status == "reserved"
    w.inventory.reconcile_record_inventory(release_reserved=True)
    assert w.journal()[0].status == "cancelled" and w.inventory.reserved_record_ids() == set()


def test_reconcile_keeps_a_wallet_clients_reservation_even_on_request() -> None:
    w = World([1_000_000])
    w.hold("1group", account_type="rpc")                    # a Veil wallet entry in a shared journal
    w.inventory.reconcile_record_inventory(release_reserved=True)
    # The wallet may have broadcast before its response was lost; only the
    # SDK's own reserved entries provably never reached the network.
    assert w.journal()[0].status == "reserved" and w.inventory.reserved_record_ids() == {"1group"}


def test_a_refusal_after_an_uncertain_broadcast_attempt_keeps_the_reservation() -> None:
    from aleo import AleoNetworkError
    w = World([1_000_000])
    record = w.inventory.get_record_inventory(CREDITS).available[0]
    with w.inventory.reserve("credits.aleo", "split", [record.id]) as checkpoint:
        checkpoint.prepared("at1x", w.build("credits.aleo", "split", (record.plaintext, "1u64"), None, 0))
        checkpoint.broadcast_failed(AleoNetworkError("POST returned 503", status=503))
        assert checkpoint.is_uncertain and not checkpoint.is_closed
        checkpoint.broadcast_failed(AleoNetworkError("POST returned 400: already exists", status=400))
        checkpoint.discard()                                # cleanup after a timeout must not release it
    assert w.journal()[0].status == "prepared" and w.inventory.reserved_record_ids() == {record.id}
    with w.inventory.reserve("credits.aleo", "split", ["2000group"]) as fresh:
        fresh.broadcast_failed(AleoNetworkError("POST returned 400", status=400))
    assert w.journal()[-1].status == "cancelled"            # a first-attempt refusal still releases


def test_a_checkpoint_in_flight_ignores_discard_and_a_discarded_one_refuses_to_broadcast() -> None:
    w = World([1_000_000, 500])
    first, second = w.inventory.get_record_inventory(CREDITS).available
    with w.inventory.reserve("credits.aleo", "split", [first.id]) as checkpoint:
        checkpoint.prepared("at1x", w.build("credits.aleo", "split", (first.plaintext, "1u64"), None, 0))
        checkpoint.begin_broadcast()                        # the request is on the wire
        checkpoint.discard()
        assert checkpoint.is_broadcasting and not checkpoint.is_closed
        checkpoint.submitted("at1x")
    assert w.journal()[0].status == "submitted" and first.id in w.inventory.reserved_record_ids()
    with w.inventory.reserve("credits.aleo", "split", [second.id]) as dropped:
        pass                                                # left the block: released
    with pytest.raises(InventoryError, match="discarded"):
        dropped.begin_broadcast()


def test_rebalance_stops_before_proving_once_the_fee_budget_is_spent() -> None:
    w = World([100] * 3)
    plan = w.inventory.plan_record_inventory(CREDITS, InventoryTarget(records=1))
    assert len(plan.steps) == 2
    result = w.inventory.rebalance_record_inventory(plan, max_fee_microcredits=100, **FAST)
    assert (result.status, result.completed_steps) == ("interrupted", 1)
    assert "budget exhausted" in str(result.reason) and len(w.builds) == 1
