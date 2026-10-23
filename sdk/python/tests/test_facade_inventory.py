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


class FakeTransaction:
    """Stands in for the proved network ``Transaction``: an id and a JSON form."""

    def __init__(self, tx: dict[str, Any]) -> None:
        self.id = tx["id"]
        self._tx = tx

    def to_json(self) -> str:
        return json.dumps(self._tx)


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

    def prove(self: BoundCall, acct: Any = None, *, priority_fee: int = 0, fee_record: Any = None,
              private_fee: bool = False, base_fee: int | None = None) -> FakeTransaction:
        sequence[0] += 1
        tag = None
        if private_fee or fee_record is not None:
            chosen = fee_record(fee + priority_fee) if callable(fee_record) else str(fee_record)
            tag = next(r["tag"] for r in aleo.record_provider.records if r["record_plaintext"] == chosen)
        tx = fake_transaction(f"at1tx{sequence[0]}", tag, fee + priority_fee)
        seen["builds"].append({"args": list(self.args), "private_fee": private_fee, "fee_record": fee_record, "tx": tx})
        return FakeTransaction(tx)

    # Replace only the proving step, so the managed build and submit halves run for real.
    monkeypatch.setattr(BoundCall, "_prove_transaction", prove)

    def submit_transaction(raw: Any, **_: Any) -> str:
        tx = json.loads(raw if isinstance(raw, str) else raw.to_json())
        seen["submits"].append(tx)
        return str(tx["id"])

    monkeypatch.setattr(aleo._client, "submit_transaction", submit_transaction)  # the raw node client
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


def test_build_and_submit_halves_are_managed_on_their_own(monkeypatch: pytest.MonkeyPatch) -> None:
    aleo, _, credits, seen = setup(monkeypatch)
    result = transfer(credits, aleo).build_transaction()
    assert result.checkpoint is not None and result.checkpoint.is_prepared
    entry = aleo.inventory.store.list(aleo.inventory.scope())[0]
    assert (entry.status, entry.transaction_id) == ("prepared", "at1tx1") and entry.transaction == seen["builds"][0]["tx"]
    assert seen["submits"] == []
    tx_id = aleo.network.submit_transaction(result)
    assert tx_id == "at1tx1" and aleo.inventory.store.list(aleo.inventory.scope())[0].status == "submitted"
    result.discard()                                                      # no effect after submission
    assert aleo.inventory.store.list(aleo.inventory.scope())[0].status == "submitted"


def test_a_built_transaction_can_be_discarded_before_submission(monkeypatch: pytest.MonkeyPatch) -> None:
    aleo, _, credits, _ = setup(monkeypatch)
    result = transfer(credits, aleo).build_transaction()
    assert aleo.inventory.reserved_record_ids() == {"1group"}
    result.discard()
    assert aleo.inventory.store.list(aleo.inventory.scope())[0].status == "cancelled"
    assert aleo.inventory.reserved_record_ids() == set()
    assert transfer(credits, aleo).transact() == "at1tx2"                 # the record is free again


def test_a_node_refusal_on_submit_releases_the_reservation(monkeypatch: pytest.MonkeyPatch) -> None:
    from aleo import AleoNetworkError
    aleo, _, credits, _ = setup(monkeypatch)

    def refuse(raw: Any, **_: Any) -> str:
        raise AleoNetworkError("POST /transaction/broadcast returned 400", status=400)

    monkeypatch.setattr(aleo._client, "submit_transaction", refuse)
    with pytest.raises(AleoNetworkError):
        transfer(credits, aleo).transact()
    assert aleo.inventory.store.list(aleo.inventory.scope())[0].status == "cancelled"


def test_a_proving_failure_releases_the_reservation(monkeypatch: pytest.MonkeyPatch) -> None:
    aleo, _, credits, _ = setup(monkeypatch)

    def fail(self: BoundCall, acct: Any = None, **kw: Any) -> Any:
        raise RuntimeError("proving failed")

    monkeypatch.setattr(BoundCall, "_prove_transaction", fail)
    with pytest.raises(RuntimeError):
        transfer(credits, aleo).build_transaction()
    assert aleo.inventory.store.list(aleo.inventory.scope())[0].status == "cancelled"


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


def test_private_fee_record_is_chosen_after_proving_and_held_in_the_same_entry_as_the_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    aleo, _, credits, seen = setup(monkeypatch)
    transfer(credits, aleo).transact(private_fee=True, priority_fee=20)
    build = seen["builds"][0]
    assert build["private_fee"] is True and callable(build["fee_record"])
    entry = aleo.inventory.store.list(aleo.inventory.scope())[0]
    assert (entry.records, entry.fee_record_id, entry.fee_microcredits, entry.status) == (("1group", "3group"), "3group", "120", "submitted")
    assert build["tx"]["fee"]["transition"]["inputs"][0]["tag"] == "3field"
    assert aleo.inventory.reserved_record_ids() == {"1group", "3group"}


def test_supplied_fee_record_is_preferred_when_eligible(monkeypatch: pytest.MonkeyPatch) -> None:
    aleo, _, credits, _ = setup(monkeypatch, records=[1_000_000, 500, 200])
    transfer(credits, aleo).transact(fee_record=aleo.record_provider.records[1]["record_plaintext"])
    assert aleo.inventory.store.list(aleo.inventory.scope())[0].fee_record_id == "2group"


def test_inventory_actions_prove_locally_on_a_managed_client_without_reserving_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    """The engine reserves the inputs itself, so the facade's managed build must not run again underneath it."""
    aleo, _, credits, seen = setup(monkeypatch, records=[1_000_000])
    monkeypatch.setattr(aleo.inventory, "proving", "local")
    monkeypatch.setattr(aleo.programs, "get", lambda program_id: credits)
    record = aleo.inventory.get_record_inventory(RecordAsset("credits.aleo")).available[0]
    tx_id = aleo.inventory.split_record(RecordAsset("credits.aleo"), record, 400_000)
    assert tx_id == "at1tx1" and [s["id"] for s in seen["submits"]] == ["at1tx1"]
    (entry,) = aleo.inventory.store.list(aleo.inventory.scope())
    assert (entry.function, entry.records, entry.status) == ("split", ("1group",), "submitted")


def test_discard_keeps_the_reservation_after_a_broadcast_attempt_with_no_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    aleo, _, credits, _ = setup(monkeypatch)
    result = transfer(credits, aleo).build_transaction()

    def lost(raw: Any, **_: Any) -> str:
        raise TimeoutError("no response")

    monkeypatch.setattr(aleo._client, "submit_transaction", lost)
    with pytest.raises(TimeoutError):
        aleo.network.submit_transaction(result)
    result.discard()                                                      # a cleanup path after the timeout
    entry = aleo.inventory.store.list(aleo.inventory.scope())[0]
    assert entry.status == "prepared" and aleo.inventory.reserved_record_ids() == {"1group"}


def test_a_refusal_after_a_retried_node_error_keeps_the_reservation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The raw client retries 5xx; the first attempt may have been admitted, so the later 4xx is not a clean refusal."""
    from aleo import AleoNetworkError
    aleo, _, credits, _ = setup(monkeypatch)
    answers = [AleoNetworkError("POST returned 503", status=503), AleoNetworkError("POST returned 400: exists", status=400)]
    attempts: list[bool] = []

    def flaky(raw: Any, *, retry: bool = True) -> str:
        attempts.append(retry)
        raise answers[len(attempts) - 1]

    monkeypatch.setattr(aleo._client, "submit_transaction", flaky)
    with pytest.raises(AleoNetworkError, match="400"):
        transfer(credits, aleo).transact()
    assert attempts == [False, False]                                     # one node attempt per report
    assert aleo.inventory.store.list(aleo.inventory.scope())[0].status == "prepared"


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

    def lost(raw: Any, **_: Any) -> str:
        raise ConnectionError("response lost")

    monkeypatch.setattr(aleo._client, "submit_transaction", lost)
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

    async def prove(self: AsyncBoundCall, acct: Any = None, **kw: Any) -> FakeTransaction:
        return FakeTransaction(fake_transaction("at1async"))

    async def submit_transaction(raw: Any, **_: Any) -> str:
        submits.append(json.loads(raw if isinstance(raw, str) else raw.to_json()))
        return "at1async"

    monkeypatch.setattr(AsyncBoundCall, "_prove_transaction", prove)
    monkeypatch.setattr(aleo.network_client, "submit_transaction", submit_transaction)
    from aleo.facade.async_client import AsyncProgram
    credits = AsyncProgram(aleo, RawProgram.from_source(CREDITS_SOURCE))
    def call() -> Any:  # the async program's callers are typed as the sync bound call
        return credits.functions.transfer_private(aleo.record_provider.records[0]["record_plaintext"], str(account.address), 1)
    assert await call().transact() == "at1async"
    assert [s["id"] for s in submits] == ["at1async"]
    entry = aleo.inventory.store.list(aleo.inventory.scope())[0]
    assert (entry.status, entry.records) == ("submitted", ("1group",))
    with pytest.raises(RecordReservationConflictError):
        await call().transact()
