"""Record inventory through the DEX client: reserved exclusion, swap reselection, inventory verbs.

_source: ProvableHQ/veil packages/shield-swap/test/actions/swap/concurrentSwaps.test.ts
and test/actions/inventory.test.ts, PR #172 @ 2e324d5. Selection, input
assembly, acquisition, and lifecycle handling run for real; the stub replaces
chain, scanner, and prover I/O and performs the facade's managed-write hook.
"""
from __future__ import annotations

import json
import threading
from typing import Any, Callable

import pytest

from aleo.inventory import (
    InventoryTarget, MemoryRecordInventoryStore, PreparedTransaction, RecordInventory, RecordManagementConfig,
    RecordReservationConflictError, RecordSpend,
)
from aleo_shield_swap import ShieldSwap
from aleo_shield_swap._api_models import TokenDoc
from aleo_shield_swap._core import record_nonce, select_token_record, select_token_records
from aleo_shield_swap.errors import ShieldSwapError

from .conftest import POOL_TEXT, SIGNER, SLOT_TEXT, StubAccount, StubAleo

TOKEN_PROGRAM = "token_a.aleo"
SCOPE = f'["testnet","{SIGNER}"]'
# A token whose declarations satisfy the inventory's join/split interface check.
UNDERLYING_SOURCE = """program test_token_a.aleo;
record Token:
    owner as address.private;
    amount as u128.private;
function transfer_public:
    input r0 as address.public;
    input r1 as u128.public;
function transfer_private:
    input r0 as Token.record;
    input r1 as address.private;
    input r2 as u128.private;
    output r3 as Token.record;
    output r4 as Token.record;
function join:
    input r0 as Token.record;
    input r1 as Token.record;
    output r2 as Token.record;
function split:
    input r0 as Token.record;
    input r1 as u128.private;
    output r2 as Token.record;
    output r3 as Token.record;
"""


def plaintext(nonce: int, amount: int = 100) -> str:
    return f"{{ owner: {SIGNER}.private, amount: {amount}u128.private, _nonce: {nonce}group.public }}"


def records(count: int) -> list[dict[str, Any]]:
    return [{"program_name": TOKEN_PROGRAM, "record_name": "Token", "spent": False, "tag": f"{n}field",
             "commitment": f"{n}field", "record_plaintext": plaintext(n)} for n in range(1, count + 1)]


class BarrierStore:
    """Delays the first *racers* acquisitions until all have arrived, so every writer picks the same record first."""

    def __init__(self, racers: int) -> None:
        self.backing = MemoryRecordInventoryStore()
        self.attempts: list[tuple[str, ...]] = []
        self.on_acquire: Callable[[], Any] | None = None
        self._barrier = threading.Barrier(racers) if racers > 1 else None
        self._lock = threading.Lock()

    def list(self, scope: str) -> list[RecordSpend]:
        return self.backing.list(scope)

    def acquire(self, entry: RecordSpend) -> RecordSpend | None:
        with self._lock:
            self.attempts.append(tuple(entry.records))
            first = len(self.attempts) <= (self._barrier.parties if self._barrier else 1)
        if first and self._barrier is not None:
            self._barrier.wait()
        if self.on_acquire is not None:
            self.on_acquire()
        return self.backing.acquire(entry)

    def update(self, scope: str, entry_id: str, patch: Any) -> None:
        self.backing.update(scope, entry_id, patch)


class _ManagedBound:
    """A prepared call whose terminal verbs run the facade's managed-write contract."""

    def __init__(self, world: Any, program: str, function: str, args: tuple[Any, ...]) -> None:
        self._world, self.program_id, self.function_name = world, program, function
        self.args = [str(a) for a in args]
        world.last_call = (function, list(self.args))
        world.last_program = program

    def simulate(self, account: Any = None) -> str:
        return "simulated"

    def _prove(self) -> dict[str, Any]:
        self._world.proved.append({"program": self.program_id, "function": self.function_name, "inputs": list(self.args)})
        if self._world.prove_error is not None:
            raise self._world.prove_error
        tx_id = f"at1tx{len(self._world.proved)}"
        return {"id": tx_id, "type": "execute", "execution": {"transitions": [
            {"program": self.program_id, "function": self.function_name, "outputs": [{"value": "77field"}]}]},
            "fee": {"transition": {"program": "credits.aleo", "function": "fee_public",
                                   "inputs": [{"type": "public", "value": "100u64"}, {"type": "public", "value": "0u64"}]}}}

    def delegate(self, account: Any = None, *, broadcast: bool = True, **kw: Any) -> dict[str, Any]:
        if not broadcast or not self._world.record_management.enabled:
            return {"transaction": self._prove()}
        payloads: list[dict[str, Any]] = []

        def build(request: Any) -> PreparedTransaction:
            tx = self._prove()
            payloads.append({"transaction": tx})
            return PreparedTransaction(tx["id"], tx)

        self._world.inventory.managed_write(self.program_id, self.function_name, self.args, account=account, builder=build)
        return payloads[0]

    def build_transaction(self, account: Any = None, **kw: Any) -> Any:
        # The facade contract: on a managed client the build reserves inputs
        # and saves the proved transaction; the submit records the broadcast.
        checkpoint = None
        if self._world.record_management.enabled:
            checkpoint = self._world.inventory.begin_spend(self.program_id, self.function_name, self.args, account=account)
        try:
            tx = self._prove()
            if checkpoint is not None:
                checkpoint.prepared(tx["id"], tx)
        except BaseException:
            if checkpoint is not None:
                checkpoint.discard()
            raise
        outer = self

        class _Result:
            id = tx["id"]

            class raw:
                @staticmethod
                def to_json() -> str:
                    return json.dumps(tx)

            def decoded(self) -> list[dict[str, Any]]:
                return [{"program": outer.program_id, "function": outer.function_name, "outputs": [{"value": "77field"}]}]
        result = _Result()
        result.checkpoint = checkpoint  # type: ignore[attr-defined]
        return result


class _ManagedFunctions:
    def __init__(self, world: Any, program: str) -> None:
        self._world, self._program = world, program

    def __getattr__(self, name: str) -> Callable[..., _ManagedBound]:
        return lambda *args: _ManagedBound(self._world, self._program, name, args)

    __getitem__ = __getattr__


class _ManagedPrograms:
    def __init__(self, world: Any, inner: Any) -> None:
        self._world, self._inner = world, inner

    def get(self, pid: str) -> Any:
        program = self._inner.get(pid)
        if pid in self._world.sources:
            program.source = self._world.sources[pid]
        program.functions = _ManagedFunctions(self._world, pid)
        return program


class ManagedStubAleo(StubAleo):
    """The conftest stub plus the facade's record management surface."""

    def __init__(self, record_list: list[dict[str, Any]], *, store: Any = None, enabled: bool = True) -> None:
        super().__init__(mappings={"pools": {"5field": POOL_TEXT}, "slots": {"5field": SLOT_TEXT},
                                   "swap_outputs": {}, "used_blinded_addresses": {}, "from_wrapper_token_id": {}},
                         records=record_list)
        self.store = store if store is not None else BarrierStore(1)
        self.record_management = RecordManagementConfig(enabled=enabled, store=self.store)
        self.inventory = RecordInventory(self, self.record_management)
        self.programs = _ManagedPrograms(self, self.programs)
        self.sources: dict[str, str] = {}
        self.proved: list[dict[str, Any]] = []
        self.prove_error: Exception | None = None
        self.submitted_json: list[dict[str, Any]] = []
        original = self.network.submit_transaction

        def submit(raw: Any) -> str:
            checkpoint = getattr(raw, "checkpoint", None)
            if isinstance(raw, str) or checkpoint is not None:
                tx = json.loads(raw if isinstance(raw, str) else raw.raw.to_json())
                self.submitted_json.append(tx)
                if checkpoint is not None:
                    checkpoint.submitted(tx["id"])
                return tx["id"]
            return original(raw)

        self.network.submit_transaction = submit  # type: ignore[method-assign]


def dex_for(world: ManagedStubAleo) -> ShieldSwap:
    dex = ShieldSwap(world)
    dex._token_program = lambda tid: TOKEN_PROGRAM  # type: ignore[method-assign]
    return dex


def swap_call(dex: ShieldSwap, index: int = 1, **kw: Any) -> Any:
    return dex.swap(pool_key="5field", token_in_id="1field", amount_in=10, expected_out=10, nonce=index,
                    token_in_program=TOKEN_PROGRAM, track=False, **kw)


def proved_nonces(world: Any) -> list[str]:
    return [record_nonce(next(i for i in call["inputs"] if "_nonce" in i)) or "" for call in world.proved]


def hold(store: BarrierStore, *nonces: str) -> None:
    assert store.backing.acquire(RecordSpend(id="", scope=SCOPE, records=nonces, program=TOKEN_PROGRAM, function="join",
                                             status="reserved", created_at=1)) is not None


# ── Selection skips reserved records ──────────────────────────────────────────

def test_selection_skips_records_the_journal_holds() -> None:
    world = ManagedStubAleo(records(2))
    hold(world.store, "1group")
    assert record_nonce(select_token_record(world, program=TOKEN_PROGRAM, min_amount=10)) == "2group"
    assert [record_nonce(p) for p in select_token_records(world, program=TOKEN_PROGRAM, min_amount=10)] == ["2group"]
    dex = dex_for(world)
    assert dex.has_swap_balance("1field", 10) is True
    hold(world.store, "2group")
    assert dex.has_swap_balance("1field", 10) is False


# ── Concurrent swaps reselect ─────────────────────────────────────────────────

def test_reselects_six_distinct_inputs_when_all_first_acquisitions_race() -> None:
    world = ManagedStubAleo(records(6), store=BarrierStore(6))
    dex = dex_for(world)
    results: list[Any] = []
    lock = threading.Lock()

    def run(index: int) -> None:
        try:
            handle = swap_call(dex, index).delegate(wait=False)
        except BaseException as exc:  # noqa: BLE001
            handle = exc
        with lock:
            results.append(handle)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(1, 7)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert all(not isinstance(r, BaseException) for r in results), results
    assert world.store.attempts[:6] == [("1group",)] * 6
    assert sorted(proved_nonces(world)) == [f"{n}group" for n in range(1, 7)]
    assert sorted(entry.records[0] for entry in world.store.list(SCOPE)) == [f"{n}group" for n in range(1, 7)]
    assert all(entry.status == "submitted" for entry in world.store.list(SCOPE))
    assert all(call["inputs"][0].startswith("{") and "10u128" in call["inputs"] for call in world.proved)


def test_stops_after_its_finite_candidates_are_reserved_even_when_the_scanner_stays_stale() -> None:
    world = ManagedStubAleo(records(2))
    dex = dex_for(world)
    call = swap_call(dex)
    world.store.on_acquire = lambda: (hold(world.store, "1group") if not any("1group" in a for a in world.store.attempts[:-1]) else None,
                                      hold(world.store, "2group") if len(world.store.attempts) == 1 else None)
    with pytest.raises(RecordReservationConflictError):
        call.delegate(wait=False)
    assert world.store.attempts == [("1group",), ("2group",)]
    assert world.proved == []


def test_keeps_explicitly_supplied_records_fixed_on_contention() -> None:
    world = ManagedStubAleo(records(2))
    dex = dex_for(world)
    hold(world.store, "1group")
    with pytest.raises(RecordReservationConflictError, match="reserved or spent"):
        swap_call(dex, token_record=plaintext(1)).delegate(wait=False)
    assert world.store.attempts == [("1group",)]
    assert world.proved == []


def test_does_not_retry_execution_errors() -> None:
    world = ManagedStubAleo(records(2))
    dex = dex_for(world)
    world.prove_error = RuntimeError("Execution failed")
    with pytest.raises(RuntimeError, match="Execution failed"):
        swap_call(dex).delegate(wait=False)
    assert proved_nonces(world) == ["1group"]
    assert world.store.attempts == [("1group",)]
    assert world.store.list(SCOPE)[0].status == "cancelled"


def test_refuses_to_retry_after_the_account_changes() -> None:
    world = ManagedStubAleo(records(2))
    dex = dex_for(world)
    call = swap_call(dex)

    def swap_account() -> None:
        hold(world.store, "1group")
        other = StubAccount()
        other.address = "aleo1other"
        world.default_account = other
        world.store.on_acquire = None

    world.store.on_acquire = swap_account
    with pytest.raises(ShieldSwapError, match="changed"):
        call.delegate(wait=False)
    assert world.proved == []


def test_preserves_accepted_terms_when_the_first_record_is_taken() -> None:
    world = ManagedStubAleo(records(3))
    dex = dex_for(world)
    call = swap_call(dex, 7)
    world.store.on_acquire = lambda: (hold(world.store, "1group"), setattr(world.store, "on_acquire", None))
    handle = call.delegate(wait=False)
    assert handle.amount_in == 10
    assert proved_nonces(world) == ["2group"]
    assert world.store.attempts == [("1group",), ("2group",)]


def test_retains_legacy_uncoordinated_selection_when_management_is_disabled() -> None:
    world = ManagedStubAleo(records(2), enabled=False)
    dex = dex_for(world)
    swap_call(dex, 1).delegate(wait=False)
    swap_call(dex, 2).delegate(wait=False)
    assert proved_nonces(world) == ["1group", "1group"]
    assert world.store.attempts == []


def test_managed_transact_reserves_and_harvests_outputs() -> None:
    world = ManagedStubAleo(records(2))
    dex = dex_for(world)
    handle = swap_call(dex).transact()
    assert handle.swap_id == "77field" and handle.transaction_id == "at1tx1"
    assert [tx["id"] for tx in world.submitted_json] == ["at1tx1"]
    assert (world.store.list(SCOPE)[0].status, world.store.list(SCOPE)[0].records) == ("submitted", ("1group",))


# ── Inventory verbs ───────────────────────────────────────────────────────────

class _Api:
    def __init__(self, tokens: list[TokenDoc]) -> None:
        self._tokens = tokens

    def get_tokens(self) -> list[TokenDoc]:
        return self._tokens


def test_plan_inventory_targets_the_underlying_program() -> None:
    underlying = "test_token_a.aleo"
    owned = [{"program_name": underlying, "record_name": "Token", "spent": False, "tag": "1field", "commitment": "1field",
              "record_plaintext": f"{{ owner: {SIGNER}.private, amount: 1000u128.private, _nonce: 1group.public }}"}]
    world = ManagedStubAleo(owned)
    world.sources[underlying] = UNDERLYING_SOURCE
    dex = dex_for(world)
    dex.api = _Api([TokenDoc(address="1field", decimals=6, id="t", name="Asset", symbol="ASSET",  # type: ignore[assignment]
                             amm_token_program="wrapper.aleo", underlying_program=underlying)])
    plan = dex.plan_inventory("asset", InventoryTarget(records=4, distribution="balanced"))
    assert plan.asset.program == underlying and len(plan.steps) == 3
    assert dex.plan_inventory("1field", InventoryTarget(records=1)).steps == ()
    with pytest.raises(ShieldSwapError, match="Unknown token"):
        dex.plan_inventory("nope", InventoryTarget(records=1))
    dex.api = _Api([TokenDoc(address="2field", decimals=6, id="t", name="Wrapped", symbol="WRAP")])  # type: ignore[assignment]
    with pytest.raises(ShieldSwapError, match="no underlying"):
        dex.plan_inventory("WRAP", InventoryTarget(records=1))


# ── Async parity ──────────────────────────────────────────────────────────────

from aleo.inventory import AsyncRecordInventory  # noqa: E402
from aleo_shield_swap import AsyncShieldSwap  # noqa: E402

from .test_async_client import AsyncStubAleo  # noqa: E402


class _AsyncManagedBound(_ManagedBound):
    async def delegate(self, account: Any = None, *, broadcast: bool = True, **kw: Any) -> dict[str, Any]:  # type: ignore[override]
        if not broadcast or not self._world.record_management.enabled:
            return {"transaction": self._prove()}
        payloads: list[dict[str, Any]] = []

        def build(request: Any) -> PreparedTransaction:
            tx = self._prove()
            payloads.append({"transaction": tx})
            return PreparedTransaction(tx["id"], tx)

        await self._world.inventory.managed_write(self.program_id, self.function_name, self.args, account=account, builder=build)
        return payloads[0]


class _AsyncManagedFunctions(_ManagedFunctions):
    def __getattr__(self, name: str) -> Callable[..., _AsyncManagedBound]:
        return lambda *args: _AsyncManagedBound(self._world, self._program, name, args)  # type: ignore[arg-type]

    __getitem__ = __getattr__


class _AsyncManagedPrograms:
    def __init__(self, world: Any, inner: Any) -> None:
        self._world, self._inner = world, inner

    async def get(self, pid: str) -> Any:
        program = await self._inner.get(pid)
        program.functions = _AsyncManagedFunctions(self._world, pid)
        return program


class AsyncManagedStubAleo(AsyncStubAleo):
    def __init__(self, record_list: list[dict[str, Any]]) -> None:
        super().__init__(mappings={"pools": {"5field": POOL_TEXT}, "slots": {"5field": SLOT_TEXT},
                                   "swap_outputs": {}, "used_blinded_addresses": {}, "from_wrapper_token_id": {}},
                         records=record_list)
        self.store = BarrierStore(1)
        self.record_management = RecordManagementConfig(enabled=True, store=self.store)
        self.inventory = AsyncRecordInventory(self, self.record_management)
        self.programs = _AsyncManagedPrograms(self, self.programs)
        self.proved: list[dict[str, Any]] = []
        self.prove_error: Exception | None = None


async def test_async_swap_reselects_after_a_conflict_and_keeps_explicit_records_fixed() -> None:
    world = AsyncManagedStubAleo(records(3))
    dex = AsyncShieldSwap(world)
    dex._token_program = _async_token_program  # type: ignore[method-assign]
    call = await dex.swap(pool_key="5field", token_in_id="1field", amount_in=10, expected_out=10, nonce=1,
                          token_in_program=TOKEN_PROGRAM)
    world.store.on_acquire = lambda: (hold(world.store, "1group"), setattr(world.store, "on_acquire", None))
    handle = await call.delegate()
    assert handle.amount_in == 10 and proved_nonces(world) == ["2group"]
    assert world.store.attempts == [("1group",), ("2group",)]
    explicit = await dex.swap(pool_key="5field", token_in_id="1field", amount_in=10, expected_out=10, nonce=2,
                              token_in_program=TOKEN_PROGRAM, token_record=plaintext(1))
    with pytest.raises(RecordReservationConflictError):
        await explicit.delegate()
    assert proved_nonces(world) == ["2group"]


async def _async_token_program(tid: str) -> str:
    return TOKEN_PROGRAM
