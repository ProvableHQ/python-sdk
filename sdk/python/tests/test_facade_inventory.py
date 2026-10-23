"""The facade's managed write hook and ``aleo.inventory`` wiring.

_source: ProvableHQ/veil packages/core/test/inventory/configuration.test.ts and
privateFees.test.ts, PR #172 @ 2e324d5, adapted to the facade's verbs.
Proving and broadcast are replaced; reservation, checkpointing, and fee
selection run for real.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from aleo import Aleo, AsyncAleo, HTTPProvider, RecordReservationConflictError
from aleo.facade.async_client import AsyncBoundCall
from aleo.facade.call import BoundCall
from aleo.facade.programs import Program
from aleo.inventory import InventoryError, MemoryRecordInventoryStore, RecordAsset, RecordManagementConfig
from aleo.mainnet import PrivateKey, Program as RawProgram

BASE = "https://api.provable.com/v2"
CREDITS_SOURCE = RawProgram.credits().source


def record(owner: str, amount: int, nonce: int) -> dict[str, Any]:
    return {"program_name": "credits.aleo", "record_name": "credits", "tag": f"{nonce}field", "commitment": f"{nonce}field",
            "spent": False, "record_plaintext": f"{{ owner: {owner}.private, microcredits: {amount}u64.private, _nonce: {nonce}group.public }}"}


class Provider:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def find(self, account: Any = None, *, program: str | None = None, unspent: bool = True, **_: Any) -> list[dict[str, Any]]:
        return [dict(r) for r in self.records if r["program_name"] == program]


class AsyncProvider(Provider):
    async def find(self, account: Any = None, *, program: str | None = None, unspent: bool = True, **_: Any) -> list[dict[str, Any]]:  # type: ignore[override]
        return super().find(account, program=program, unspent=unspent)


def fake_transaction(tx_id: str, fee_tag: str | None = None, fee: int = 100) -> dict[str, Any]:
    inputs: list[dict[str, Any]] = [{"type": "public", "value": f"{fee}u64"}, {"type": "public", "value": "0u64"}]
    function = "fee_public"
    if fee_tag is not None:
        inputs.insert(0, {"type": "record", "tag": fee_tag})
        function = "fee_private"
    return {"id": tx_id, "type": "execute", "execution": {"transitions": []},
            "fee": {"transition": {"program": "credits.aleo", "function": function, "inputs": inputs}}}


class FakeResult:
    def __init__(self, tx: dict[str, Any]) -> None:
        self.id = tx["id"]
        self.raw = self

    def to_json(self) -> str:
        return json.dumps(self._tx)

    @classmethod
    def of(cls, tx: dict[str, Any]) -> "FakeResult":
        result = cls(tx)
        result._tx = tx
        return result


def setup(monkeypatch: pytest.MonkeyPatch, *, enabled: bool = True, max_fee: int | None = None, fee: int = 100,
          records: list[int] | None = None) -> tuple[Aleo, Any, Program, dict[str, Any]]:
    """A mainnet client whose proving and broadcast are faked; its inventory is real."""
    config = RecordManagementConfig(enabled=enabled, store=MemoryRecordInventoryStore(), max_fee_microcredits=max_fee)
    aleo = Aleo(HTTPProvider(BASE), record_management=config)
    account = aleo.account.from_private_key(PrivateKey.random())
    aleo.default_account = account
    owner = str(account.address)
    aleo.record_provider = Provider([record(owner, amount, index + 1) for index, amount in enumerate(records or [1_000_000, 110, 200])])
    seen: dict[str, Any] = {"builds": [], "submits": [], "nonce_tag": {}}
    sequence = [0]

    def build_transaction(self: BoundCall, acct: Any = None, *, priority_fee: int = 0, fee_record: Any = None,
                          private_fee: bool = False, base_fee: int | None = None) -> FakeResult:
        sequence[0] += 1
        tag = None
        if private_fee or fee_record is not None:
            chosen = fee_record(fee + priority_fee) if callable(fee_record) else str(fee_record)
            tag = next(r["tag"] for r in aleo.record_provider.records if r["record_plaintext"] == chosen)
        tx = fake_transaction(f"at1tx{sequence[0]}", tag, fee + priority_fee)
        seen["builds"].append({"args": list(self.args), "private_fee": private_fee, "fee_record": fee_record, "tx": tx})
        return FakeResult.of(tx)

    monkeypatch.setattr(BoundCall, "build_transaction", build_transaction)

    def submit_transaction(raw: Any) -> str:
        # The managed path submits JSON text; the plain path submits the result object.
        tx = json.loads(raw if isinstance(raw, str) else raw.to_json())
        seen["submits"].append(tx)
        return str(tx["id"])

    monkeypatch.setattr(aleo.network, "submit_transaction", submit_transaction)
    # The synthetic records cannot be authorized by snarkVM, so a prover-side
    # delegate (broadcast=False) is answered directly; the broadcasting call
    # still runs the real hook.
    original_delegate = BoundCall.delegate

    def delegate(self: BoundCall, acct: Any = None, *, broadcast: bool = True, pay_own_fee: bool = False,
                 fee_record: Any = None, priority_fee: int = 0) -> Any:
        if broadcast:
            return original_delegate(self, acct, broadcast=True, pay_own_fee=pay_own_fee, fee_record=fee_record, priority_fee=priority_fee)
        sequence[0] += 1
        tx = fake_transaction(f"at1dps{sequence[0]}", None, fee)
        seen["builds"].append({"args": list(self.args), "delegated": True, "fee_record": fee_record, "tx": tx})
        return {"transaction": tx, "broadcast_result": None}

    monkeypatch.setattr(BoundCall, "delegate", delegate)
    credits = Program(aleo, RawProgram.from_source(CREDITS_SOURCE))
    return aleo, account, credits, seen


def transfer(credits: Program, aleo: Aleo, index: int = 0) -> BoundCall:
    plaintext = aleo.record_provider.records[index]["record_plaintext"]
    call = credits.functions.transfer_private(plaintext, str(aleo.default_account.address), 1)
    assert isinstance(call, BoundCall)
    return call


def test_client_exposes_a_lazy_inventory_and_its_configuration() -> None:
    aleo = Aleo(HTTPProvider(BASE))
    assert aleo.record_management is None
    assert aleo.inventory is aleo.inventory and aleo.inventory.enabled is False
    config = RecordManagementConfig(enabled=True)
    managed = Aleo(HTTPProvider(BASE, network="testnet"), record_management=config)
    assert managed.record_management is config and managed.inventory.enabled is True
    assert managed.inventory.store is not None and managed.inventory.network == "testnet"


def test_managed_transact_reserves_inputs_saves_the_transaction_and_journals_submission(monkeypatch: pytest.MonkeyPatch) -> None:
    aleo, _, credits, seen = setup(monkeypatch)
    tx_id = transfer(credits, aleo).transact()
    assert tx_id == "at1tx1" and [s["id"] for s in seen["submits"]] == ["at1tx1"]
    entry = aleo.inventory.store.list(aleo.inventory.scope())[0]
    assert (entry.records, entry.status, entry.transaction_id, entry.fee_microcredits) == (("1group",), "submitted", "at1tx1", "100")
    assert entry.transaction == seen["builds"][0]["tx"]
    assert aleo.inventory.reserved_record_ids() == {"1group"}
    assert aleo.inventory.get_record_inventory(RecordAsset("credits.aleo")).available[0].id == "2group"


def test_managed_transact_refuses_a_reserved_input_before_proving(monkeypatch: pytest.MonkeyPatch) -> None:
    aleo, _, credits, seen = setup(monkeypatch)
    transfer(credits, aleo).transact()
    with pytest.raises(RecordReservationConflictError) as conflict:
        transfer(credits, aleo).transact()
    assert conflict.value.record_ids == ("1group",) and len(seen["builds"]) == 1


def test_unmanaged_client_writes_are_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    aleo, _, credits, seen = setup(monkeypatch, enabled=False)
    assert transfer(credits, aleo).transact() == "at1tx1"
    assert transfer(credits, aleo).transact() == "at1tx2"            # the same record spends twice: nothing coordinates
    assert aleo.inventory.store is None and len(seen["submits"]) == 2


def test_fee_ceiling_is_enforced_before_broadcast(monkeypatch: pytest.MonkeyPatch) -> None:
    aleo, _, credits, seen = setup(monkeypatch, max_fee=50)
    with pytest.raises(InventoryError, match="max_fee_microcredits"):
        transfer(credits, aleo).transact()
    assert seen["submits"] == []
    assert aleo.inventory.store.list(aleo.inventory.scope())[0].status == "cancelled"
    assert aleo.inventory.reserved_record_ids() == set()


def test_private_fee_record_is_chosen_after_proving_and_reserved_beside_the_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    aleo, _, credits, seen = setup(monkeypatch)
    transfer(credits, aleo).transact(private_fee=True, priority_fee=20)
    build = seen["builds"][0]
    assert build["private_fee"] is True and callable(build["fee_record"])
    inputs_entry, fee_entry = aleo.inventory.store.list(aleo.inventory.scope())
    assert (inputs_entry.records, inputs_entry.fee_record_id, inputs_entry.fee_microcredits) == (("1group",), None, "120")
    assert (fee_entry.records, fee_entry.fee_record_id, fee_entry.status) == (("3group",), "3group", "submitted")
    assert build["tx"]["fee"]["transition"]["inputs"][0]["tag"] == "3field"


def test_supplied_fee_record_is_preferred_when_eligible(monkeypatch: pytest.MonkeyPatch) -> None:
    aleo, _, credits, _ = setup(monkeypatch, records=[1_000_000, 500, 200])
    transfer(credits, aleo).transact(fee_record=aleo.record_provider.records[1]["record_plaintext"])
    assert aleo.inventory.store.list(aleo.inventory.scope())[1].fee_record_id == "2group"


def test_managed_delegate_proves_without_broadcast_and_submits_from_the_client(monkeypatch: pytest.MonkeyPatch) -> None:
    aleo, account, credits, seen = setup(monkeypatch)
    payload = transfer(credits, aleo).delegate(account)
    assert payload["transaction"]["id"] == "at1dps1" and seen["builds"][0]["delegated"] is True
    assert [s["id"] for s in seen["submits"]] == ["at1dps1"]
    entry = aleo.inventory.store.list(aleo.inventory.scope())[0]
    assert (entry.status, entry.transaction_id, entry.records) == ("submitted", "at1dps1", ("1group",))
    with pytest.raises(RecordReservationConflictError):
        transfer(credits, aleo).delegate(account)


def test_delegate_without_broadcast_stays_unmanaged(monkeypatch: pytest.MonkeyPatch) -> None:
    aleo, account, credits, seen = setup(monkeypatch)
    transfer(credits, aleo).delegate(account, broadcast=False)
    assert aleo.inventory.store.list(aleo.inventory.scope()) == [] and seen["submits"] == []


def test_lost_broadcast_response_keeps_the_prepared_transaction(monkeypatch: pytest.MonkeyPatch) -> None:
    aleo, _, credits, _ = setup(monkeypatch)

    def lost(raw: str) -> str:
        raise ConnectionError("response lost")

    monkeypatch.setattr(aleo.network, "submit_transaction", lost)
    with pytest.raises(ConnectionError):
        transfer(credits, aleo).transact()
    entry = aleo.inventory.store.list(aleo.inventory.scope())[0]
    assert entry.status == "prepared" and entry.transaction is not None and entry.transaction_id == "at1tx1"


def test_records_owned_by_another_account_cannot_be_reserved(monkeypatch: pytest.MonkeyPatch) -> None:
    aleo, _, credits, seen = setup(monkeypatch)
    foreign = record("aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3ljyzc", 5, 9)["record_plaintext"]
    with pytest.raises(InventoryError, match="another account"):
        credits.functions.transfer_private(foreign, str(aleo.default_account.address), 1).transact()  # type: ignore[union-attr]
    assert seen["builds"] == []


@pytest.mark.asyncio
async def test_async_managed_transact_runs_the_same_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    config = RecordManagementConfig(enabled=True, store=MemoryRecordInventoryStore())
    aleo = AsyncAleo(HTTPProvider(BASE), record_management=config)
    account = aleo.account.from_private_key(PrivateKey.random())
    aleo.default_account = account
    aleo.record_provider = AsyncProvider([record(str(account.address), 1_000_000, 1)])
    submits: list[dict[str, Any]] = []

    async def build_transaction(self: AsyncBoundCall, acct: Any = None, **kw: Any) -> FakeResult:
        return FakeResult.of(fake_transaction("at1async"))

    async def submit_transaction(raw: str) -> str:
        submits.append(json.loads(raw))
        return "at1async"

    monkeypatch.setattr(AsyncBoundCall, "build_transaction", build_transaction)
    monkeypatch.setattr(aleo.network, "submit_transaction", submit_transaction)
    from aleo.facade.async_client import AsyncProgram
    credits = AsyncProgram(aleo, RawProgram.from_source(CREDITS_SOURCE))
    call = credits.functions.transfer_private(aleo.record_provider.records[0]["record_plaintext"], str(account.address), 1)
    assert await call.transact() == "at1async"
    assert [s["id"] for s in submits] == ["at1async"]
    entry = aleo.inventory.store.list(aleo.inventory.scope())[0]
    assert (entry.status, entry.records) == ("submitted", ("1group",))
    with pytest.raises(RecordReservationConflictError):
        await credits.functions.transfer_private(aleo.record_provider.records[0]["record_plaintext"], str(account.address), 1).transact()
