"""Inventory orchestration, written once for the sync and async clients.

Every action here is a generator. Instead of calling the facade, it ``yield``s
a small request object (fetch a program, scan records, prove, broadcast,
look up a confirmation, sleep) and receives the answer from whichever client
is driving it. :class:`~aleo.inventory.client.RecordInventory` performs the
requests synchronously; :class:`~aleo.inventory.client.AsyncRecordInventory`
awaits them. The rules that protect funds (reserve before proving, save the
proved transaction before broadcast, keep a reservation when an outcome is
uncertain, feed only confirmed outputs into later steps) therefore exist in
one place and behave identically on both clients.

Reading the generators: ``value = yield Request(...)`` is a facade call whose
result is ``value``; ``yield from self.other(...)`` composes actions like an
ordinary call. Errors raised by the facade are thrown into the generator at
the ``yield`` that requested the work, so a ``try`` around a ``yield`` catches
them where they matter.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Generator, Literal, Mapping, Sequence, TypeVar, Union, cast

from .._client_common import AleoNetworkError
from .assets import (
    CREDITS_PROGRAM, AssetCapabilities, amount_width, decode_record, inventory_record_id, max_amount, scope_for,
    split_deduction, validate_asset,
)
from .interface import parse_program_interface
from .join_programs import (
    join_program_candidates, router_as_helper, supported_join_program, validate_record_join_function,
    validate_token_join_function, validate_token_join_router,
)
from .planner import build_inventory_plan, positive_integer
from .store import reserved_record_ids
from .types import (
    InventoryError, InventoryPlan, InventoryRecord, InventoryResult, InventoryStep, InventoryTarget, PlannedRecord,
    RecordAsset, RecordInventorySnapshot, RecordInventoryStore, RecordJoinProgram, RecordManagementConfig,
    RecordReservationConflictError, RecordSpend, ResolvedRecordAsset, TokenJoinRouter,
)

T = TypeVar("T")

# ── Requests the engine yields and the clients perform ───────────────────────


@dataclass(frozen=True)
class ProgramSource:
    """Fetch a deployed program's source; the answer is the text, or ``None`` when it is not deployed."""

    program_id: str


@dataclass(frozen=True)
class FindRecords:
    """Scan the account's unspent records of one program; the answer is the scanner's record list."""

    program: str


@dataclass(frozen=True)
class PreparedTransaction:
    """A proved transaction that has not been broadcast.

    Attributes:
        id: The transaction id.
        transaction: The transaction as JSON data, saved to the journal before
            broadcast so recovery can resubmit it.
    """

    id: str
    transaction: dict[str, Any]


FeeRecordSelector = Callable[[int], str]


@dataclass(frozen=True)
class BuildTransaction:
    """Prove a call without broadcasting; the answer is a :class:`PreparedTransaction`.

    Attributes:
        program: Program to call.
        function: Transition name.
        inputs: Aleo-encoded inputs, record plaintext included.
        imports: Programs reached by dynamic dispatch that the prover must
            register beside the called program.
        private_fee: Pay the fee from a credits record instead of the public
            balance.
        fee_record: Under a private fee, a callable that receives the exact
            fee in microcredits once the execution is proved and returns the
            plaintext of the credits record to spend. The engine reserves that
            record inside the callable.
    """

    program: str
    function: str
    inputs: tuple[str, ...]
    imports: tuple[str, ...] = ()
    private_fee: bool = False
    fee_record: FeeRecordSelector | None = None


@dataclass(frozen=True)
class Broadcast:
    """Submit a proved transaction; the answer is the transaction id the node returned."""

    transaction: Mapping[str, Any]


@dataclass(frozen=True)
class ConfirmedTransaction:
    """Look up a confirmed transaction; the answer is its JSON, or ``None`` while the node has none."""

    tx_id: str


@dataclass(frozen=True)
class Sleep:
    """Pause between polls."""

    seconds: float


@dataclass(frozen=True)
class WaitForScannerSync:
    """Wait until the record scanner reports itself synced, or raise after *timeout* seconds."""

    timeout: float


Request = Union[ProgramSource, FindRecords, BuildTransaction, Broadcast, ConfirmedTransaction, Sleep, WaitForScannerSync]
Step = Generator[Request, Any, T]
JsonDict = dict[str, Any]


class _Default:
    """Marker for "use the engine's configured value" where ``None`` is itself a valid setting."""


DEFAULT = _Default()


def _dict(value: Any) -> JsonDict:
    """A JSON object from untyped data, or an empty one."""
    return cast(JsonDict, value) if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    """A JSON array from untyped data, or an empty one."""
    return cast(list[Any], value) if isinstance(value, list) else []


_RECORD_LITERAL = re.compile(r"^\s*\{.*_nonce:\s*\d+group", re.DOTALL)
_U64 = re.compile(r"^\d+u64$")
_OWNER = re.compile(r"(?<![A-Za-z0-9_])owner:\s*([a-z0-9]+)")


@dataclass(frozen=True)
class Management:
    """Record management settings resolved for one client.

    Attributes:
        enabled: Reservations and recovery are on.
        store: The journal; ``None`` while disabled.
        chain_id: Scope chain identity.
        max_fee_microcredits: Per-transaction fee ceiling, or ``None``.
        token_join: Router policy (``None`` automatic, ``False`` native only).
        wait_for_sync: Wait for scanner synchronisation before each scan.
        sync_timeout: Seconds to wait for synchronisation.
    """

    enabled: bool
    store: RecordInventoryStore | None
    chain_id: str
    max_fee_microcredits: int | None
    token_join: TokenJoinRouter | Literal[False] | None
    wait_for_sync: bool
    sync_timeout: float

    @staticmethod
    def resolve(config: RecordManagementConfig, network: str, default_store: RecordInventoryStore | None) -> "Management":
        if config.token_join:
            validate_token_join_router(config.token_join)
        if config.max_fee_microcredits is not None and config.max_fee_microcredits < 0:
            raise InventoryError("Fee limit cannot be negative")
        return Management(
            enabled=config.enabled, store=default_store if config.enabled else None,
            chain_id=config.chain_id or network, max_fee_microcredits=config.max_fee_microcredits,
            token_join=config.token_join, wait_for_sync=config.wait_for_sync, sync_timeout=config.sync_timeout,
        )


# ── Pure helpers used by the engine and the facade hook ──────────────────────


def transaction_fee(transaction: Mapping[str, Any]) -> int:
    """Base plus priority fee in microcredits, read from the proved transaction.

    Both fee functions publish the amounts; ``fee_private`` puts its consumed
    credits record before them.

    Raises:
        InventoryError: The fee transition is missing or malformed, so it
            cannot be compared with a spending limit.
    """
    fee = _dict(transaction.get("fee"))
    if not fee:
        return 0
    transition = _dict(fee.get("transition"))
    offset = 1 if transition.get("function") == "fee_private" else 0
    inputs = _list(transition.get("inputs"))
    values = [_dict(item).get("value") for item in inputs[offset:offset + 2]]
    amounts = [value for value in values if isinstance(value, str) and _U64.match(value)]
    if len(values) != 2 or len(amounts) != 2:
        raise InventoryError("Cannot verify transaction fee before broadcast")
    return sum(int(value[:-3]) for value in amounts)


def inventory_output_commitments(
    transaction: Mapping[str, Any], asset: RecordAsset, step: InventoryStep, router: TokenJoinRouter | None,
) -> list[str]:
    """Record commitments a confirmed step produced, in planned output order.

    Native joins and splits expose their commitments directly. A static helper
    returns an external record and a dynamic router returns a dynamic id, so
    for both the commitment the scanner indexes comes from the final nested
    native ``join``, the second-to-last transition.

    Raises:
        InventoryError: The outer transition does not match the step, or the
            nested output cannot be matched unambiguously.
    """
    transitions = [_dict(item) for item in _list(_dict(transaction.get("execution")).get("transitions"))]
    outer = transitions[-1] if transitions else {}
    helper = step.join_program if step.kind == "join" else None
    routed = step.kind == "join" and asset.program != CREDITS_PROGRAM and router is not None
    target = helper.program if helper else (router.program if routed and router else asset.program)
    expected = f"join_{len(step.inputs)}" if (helper or routed) else step.kind
    if not outer or outer.get("program") != target or outer.get("function") != expected:
        raise InventoryError("Confirmed transition does not match the inventory plan")
    outputs = [_dict(item) for item in _list(outer.get("outputs"))]
    final = transitions[-2] if len(transitions) >= 2 else {}
    joined = [_dict(item) for item in _list(final.get("outputs"))]
    nested_join = final.get("program") == asset.program and final.get("function") == "join" and len(joined) == 1
    if helper is not None and helper.input_mode == "records":
        # A static helper returns an external record; the final nested native
        # join produces the encrypted record the scanner indexes.
        if (
            len(outputs) != 1 or outputs[0].get("type") != "external_record"
            or not nested_join or joined[0].get("type") not in ("record", "record_with_dynamic_id")
        ):
            raise InventoryError("Cannot resolve confirmed helper output to the underlying record commitment")
        return [str(joined[0]["id"])]
    if helper is None and not routed:
        return [str(item["id"]) for item in outputs if item.get("type") in ("record", "record_with_dynamic_id")]
    if len(outputs) != 1 or outputs[0].get("type") != "record_dynamic":
        raise InventoryError("Unexpected confirmed router output")
    # The router's own return slot carries a dynamic id, not a scanner
    # commitment; the final native join holds the encrypted record.
    if not nested_join or joined[0].get("type") != "record_with_dynamic_id":
        raise InventoryError("Cannot resolve confirmed router output to the underlying token commitment")
    return [str(joined[0]["id"])]


def record_input_ids(inputs: Sequence[str], address: str) -> list[str]:
    """Nonces of the record plaintext inputs the account owns.

    Raises:
        InventoryError: An input is a record ciphertext (no nonce to reserve),
            belongs to another account, or repeats.
    """
    ids: list[str] = []
    for item in inputs:
        if item.startswith("record1"):
            raise InventoryError("Managed spending requires a record nonce; ciphertext alone is insufficient")
        if not _RECORD_LITERAL.match(item):
            continue
        owner = _OWNER.search(item)
        if owner is None or owner.group(1) != address:
            raise InventoryError("Cannot reserve another account's record")
        ids.append(inventory_record_id({"record_plaintext": item}))
    if len(set(ids)) != len(ids):
        raise InventoryError("A transaction cannot spend the same record twice")
    return ids


def _now_ms() -> int:
    return int(time.time() * 1000)


class SpendCheckpoint:
    """The journal state of one spend, from reservation to submission.

    A spend is the two halves snarkVM separates: proving a transaction and
    broadcasting it. :meth:`open_spend` reserves the record inputs before
    proving; :meth:`prepared` saves the proved transaction before broadcast
    and enforces the fee ceiling; :meth:`submitted` records that the network
    accepted it. A spend that never reaches the network is released with
    :meth:`discard`, or by :meth:`broadcast_failed` when the node refuses
    the transaction outright. Under a private fee, :attr:`fee_record_selector`
    chooses and reserves the credits record once the exact fee is known.

    Attributes:
        entry: The journal entry holding the input reservation.
        entries: Every entry this spend holds; a pinned fee record adds one.
        transaction_id: The id recorded at preparation or submission.
    """

    def __init__(
        self, store: RecordInventoryStore, scope: str, scope_now: Callable[[], str], entry: RecordSpend, *,
        fee_limit: int | None,
    ) -> None:
        self._store = store
        self._scope = scope
        self._scope_now = scope_now
        self._fee_limit = fee_limit
        self._base = entry
        self.entry = entry
        self.entries: list[RecordSpend] = [entry]
        self.transaction_id: str | None = None
        self.is_prepared = False
        self.is_submitted = False
        self.is_closed = False
        self._fee_candidates: list[InventoryRecord] = []
        self._fee_choice: InventoryRecord | None = None
        self.fee_record_selector: FeeRecordSelector | None = None

    @property
    def input_ids(self) -> tuple[str, ...]:
        """The reserved execution input nonces."""
        return self._base.records

    def use_fee_candidates(self, available: Sequence[InventoryRecord], preferred: str | None = None) -> FeeRecordSelector:
        """Prepare private-fee selection from the credits records available now.

        The returned callable is handed to the prover as ``fee_record``. It
        runs once the execution is proved and the exact fee is known, picks
        the smallest sufficient record (the preferred nonce first when it is
        eligible), and reserves it as its own journal entry so a rejected
        execution can keep it held while releasing the inputs.
        """
        inputs = set(self._base.records)
        self._fee_candidates = sorted(
            (item for item in available if item.id not in inputs and item.plaintext and item.tag),
            key=lambda item: (item.id != preferred, item.amount, item.id),
        )

        def select(required: int) -> str:
            if self._fee_limit is not None and required > self._fee_limit:
                raise InventoryError("Estimated transaction exceeds max_fee_microcredits")
            if self._scope_now() != self._scope:
                raise InventoryError("Account or chain changed during private fee selection")
            for candidate in self._fee_candidates:
                if candidate.amount < required:
                    continue
                fee_entry = self._store.acquire(replace(self._base, records=(candidate.id,), fee_record_id=candidate.id))
                if fee_entry is not None:
                    self.entries.append(fee_entry)
                    self._fee_choice = candidate
                    return candidate.plaintext
            raise InventoryError(
                "No available credits record covers the private fee separately from the transaction inputs; "
                "refresh records or fund private credits"
            )

        self.fee_record_selector = select
        return select

    def prepared(self, tx_id: str, transaction: Mapping[str, Any], fee_microcredits: int | None = None) -> int:
        """Save the proved transaction before broadcast; returns the fee in microcredits.

        Raises:
            InventoryError: The account or chain changed, the prover did not
                spend the reserved fee record, the fee exceeds the pinned
                record or ``max_fee_microcredits``, or the fee cannot be read.
                Call :meth:`discard` (or let the caller's cleanup do it)
                because nothing was broadcast.
        """
        if self._scope_now() != self._scope:
            raise InventoryError("Account or chain changed during record spending")
        chosen = self._fee_choice
        if chosen is not None:
            # Check the prover honoured the pinned record before saving a
            # transaction that recovery could broadcast.
            fee_transition = _dict(_dict(transaction.get("fee")).get("transition"))
            fee_inputs = _list(fee_transition.get("inputs"))
            first = _dict(fee_inputs[0]) if fee_inputs else {}
            if (
                fee_transition.get("program") != CREDITS_PROGRAM or fee_transition.get("function") != "fee_private"
                or first.get("type") != "record" or first.get("tag") != chosen.tag
            ):
                raise InventoryError("Prover did not spend the reserved private fee record")
        fee = transaction_fee(transaction) if fee_microcredits is None else fee_microcredits
        if chosen is not None and fee > chosen.amount:
            raise InventoryError("Private fee exceeds the reserved credits record balance")
        if self._fee_limit is not None and fee > self._fee_limit:
            raise InventoryError("Proved transaction exceeds max_fee_microcredits")
        for item in self.entries:
            self._store.update(self._scope, item.id, {
                "status": "prepared", "transaction_id": tx_id, "transaction": dict(transaction), "fee_microcredits": str(fee),
            })
        self.transaction_id = tx_id
        self.is_prepared = True
        return fee

    def submitted(self, tx_id: str) -> None:
        """Record that the network accepted the broadcast; the reservation now outlives any error."""
        # Mark before touching the store: a storage failure must not release
        # inputs the network may already have accepted.
        self.is_submitted = True
        self.transaction_id = tx_id
        for item in self.entries:
            self._store.update(self._scope, item.id, {"status": "submitted", "transaction_id": tx_id})

    def confirmed(self, tx_id: str) -> None:
        """Record acceptance and drop the saved transaction."""
        for item in self.entries:
            self._store.update(self._scope, item.id, {"status": "confirmed", "transaction_id": tx_id, "transaction": None})

    def broadcast_failed(self, error: BaseException) -> None:
        """Decide what a failed broadcast means for the reservation.

        A 4xx from the broadcast endpoint means the node refused the
        transaction before admitting it, so nothing can reach the chain and
        the inputs are released. Any other failure (timeout, lost response,
        5xx) is uncertain and keeps them for reconciliation.
        """
        status = getattr(error, "status", None)
        if isinstance(error, AleoNetworkError) and isinstance(status, int) and 400 <= status < 500:
            self.discard()

    def discard(self) -> None:
        """Release the reservation of a spend that was never submitted.

        Safe to call more than once. Does nothing after :meth:`submitted`,
        because the network may have spent the records.
        """
        if self.is_submitted or self.is_closed:
            return
        self.is_closed = True
        for item in self.entries:
            self._store.update(self._scope, item.id, {"status": "cancelled"})


def open_spend(
    store: RecordInventoryStore, scope: str, scope_now: Callable[[], str], ids: Sequence[str], program: str, function: str,
    *, asset_program: str | None = None, fee_limit: int | None = None,
) -> SpendCheckpoint:
    """Reserve the input nonces of one spend before any proving starts.

    Raises:
        RecordReservationConflictError: Another operation holds one of them.
    """
    entry = store.acquire(RecordSpend(
        id="", scope=scope, records=tuple(ids), program=program, function=function, status="reserved",
        created_at=_now_ms(), asset_program=asset_program,
    ))
    if entry is None:
        raise RecordReservationConflictError(tuple(ids))
    return SpendCheckpoint(store, scope, scope_now, entry, fee_limit=fee_limit)


# ── The engine ───────────────────────────────────────────────────────────────


class InventoryEngine:
    """Sans-IO record inventory actions bound to one network, account, and management setting.

    Args:
        network: Network name, used for helper lookup and the default scope.
        address: Callable returning the signer's current address, so a
            changed default account is noticed mid-run.
        management: Resolved management settings.
        clock: Monotonic clock in seconds, injectable for tests.
    """

    def __init__(
        self, network: str, address: Callable[[], str], management: Management,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.network = network
        self._address = address
        self.management = management
        self.clock = clock

    # ── Scope and reservations ───────────────────────────────────────────────

    @property
    def address(self) -> str:
        return self._address()

    @property
    def scope(self) -> str:
        return scope_for(self.management.chain_id, self.address)

    def reserved_record_ids(self) -> set[str]:
        """Nonces excluded by the journal; empty while management is disabled."""
        return reserved_record_ids(self.management.store, self.scope)

    def _store(self, action: str) -> RecordInventoryStore:
        if not self.management.enabled or self.management.store is None:
            raise InventoryError(f"{action} requires record management to be enabled with a shared store")
        return self.management.store

    # ── Reads ────────────────────────────────────────────────────────────────

    def _validate_asset(self, asset: RecordAsset) -> Step[AssetCapabilities]:
        source = None if asset.program == CREDITS_PROGRAM else (yield ProgramSource(asset.program))
        return validate_asset(asset, source)

    def _find_records(self, program: str) -> Step[list[Mapping[str, Any]]]:
        if self.management.wait_for_sync:
            yield WaitForScannerSync(self.management.sync_timeout)
        records: list[Mapping[str, Any]] = yield FindRecords(program)
        return records

    def get_record_inventory(self, asset: RecordAsset) -> Step[RecordInventorySnapshot]:
        capabilities = yield from self._validate_asset(asset)
        excluded = self.reserved_record_ids()
        records = yield from self._find_records(asset.program)
        seen: dict[str, InventoryRecord] = {}
        address = self.address
        for record in records:
            decoded = decode_record(address, asset, record, capabilities)
            if decoded is not None and decoded.id not in seen:
                seen[decoded.id] = decoded
        available = tuple(item for item in seen.values() if item.id not in excluded)
        reserved = tuple(item for item in seen.values() if item.id in excluded)
        return RecordInventorySnapshot(capabilities.asset, available, reserved, sum(item.amount for item in available))

    def _join_programs(self, asset: ResolvedRecordAsset, token_join: TokenJoinRouter | Literal[False] | None) -> Step[tuple[RecordJoinProgram, ...]]:
        available: list[RecordJoinProgram] = []
        for helper in join_program_candidates(self.network, asset, token_join):
            # A missing deployment permits native joins; a transport outage
            # propagates instead of silently changing the transaction count.
            source = yield ProgramSource(helper.program)
            if source is None:
                continue
            supported = supported_join_program(helper, parse_program_interface(source), asset)
            if supported is not None:
                available.append(supported)
        return tuple(available)

    def plan_record_inventory(
        self, asset: RecordAsset, target: InventoryTarget, max_transactions: int | None = None,
    ) -> Step[InventoryPlan]:
        inventory = yield from self.get_record_inventory(asset)
        helpers: tuple[RecordJoinProgram, ...] = ()
        if len(inventory.available) > 2:
            helpers = yield from self._join_programs(inventory.asset, self.management.token_join)
        # Keep only nonces and amounts in the plan; plaintext stays in the
        # scanner result and is fetched again when each step executes.
        return build_inventory_plan(
            scope=self.scope, asset=inventory.asset, target=target,
            inputs=[PlannedRecord(item.id, item.amount) for item in inventory.available],
            max_transactions=max_transactions, token_join=self.management.token_join, join_programs=helpers,
        )

    # ── Single writes ────────────────────────────────────────────────────────

    def join_records(
        self, asset: RecordAsset, records: Sequence[InventoryRecord], join_program: RecordJoinProgram | Literal[False] | None = None,
        *, token_join: TokenJoinRouter | Literal[False] | None | _Default = DEFAULT,
        fee_limit: int | None | _Default = DEFAULT, observed: list[str] | None = None,
    ) -> Step[str]:
        router_policy = self.management.token_join if isinstance(token_join, _Default) else token_join
        capabilities = yield from self._validate_asset(asset)
        resolved = capabilities.asset
        router = router_policy if (resolved.program != CREDITS_PROGRAM and router_policy) else None
        helper: RecordJoinProgram | None
        if join_program is None and router:
            validate_token_join_router(router)
            source = yield ProgramSource(router.program)
            if source is None:
                raise InventoryError(f"{router.program} is not deployed")
            validate_token_join_function(router, parse_program_interface(source), len(records))
            helper = router_as_helper(router)
        elif join_program is None and len(records) > 2:
            helpers = yield from self._join_programs(resolved, router_policy)
            helper = next((item for item in helpers if len(records) in item.counts), None)
        elif join_program is False:
            helper = None
        else:
            helper = join_program
        if helper is not None:
            source = yield ProgramSource(helper.program)
            if source is None:
                raise InventoryError(f"{helper.program} is not deployed")
            validate_record_join_function(helper, parse_program_interface(source), resolved, len(records))
        elif len(records) != 2:
            raise InventoryError(
                "join_records requires two records without an available join helper; use auto_join for several transactions"
            )
        # Re-decode before proving; cached amounts do not establish ownership,
        # eligibility, or distinct inputs.
        address = self.address
        decoded = [decode_record(address, asset, item.record, capabilities) for item in records]
        eligible = [item for item in decoded if item is not None]
        if len(eligible) != len(records) or len({item.id for item in eligible}) != len(eligible):
            raise InventoryError("Join requires distinct eligible records owned by this account")
        if sum(item.amount for item in eligible) > max_amount(resolved):
            raise InventoryError("Joined amount overflows the token width")
        dynamic = helper is not None and helper.input_mode == "program_identifier_then_records"
        inputs = [f"'{resolved.program[:-5]}'"] if dynamic else []
        inputs.extend(item.plaintext for item in eligible)
        return (yield from self._spend(
            helper.program if helper else resolved.program, f"join_{len(eligible)}" if helper else "join", inputs,
            imports=(resolved.program,) if dynamic else (),
            asset_program=resolved.program if helper else None, fee_limit=fee_limit, observed=observed,
        ))

    def split_record(
        self, asset: RecordAsset, record: InventoryRecord, amount: int,
        *, fee_limit: int | None | _Default = DEFAULT, observed: list[str] | None = None,
    ) -> Step[str]:
        capabilities = yield from self._validate_asset(asset)
        resolved = capabilities.asset
        decoded = decode_record(self.address, asset, record.record, capabilities)
        # Both outputs must stay positive after the native split's own deduction.
        if decoded is None or isinstance(amount, bool) or amount <= 0 or amount + split_deduction(resolved) >= decoded.amount:
            raise InventoryError("Split must produce two positive eligible records")
        return (yield from self._spend(
            resolved.program, "split", [decoded.plaintext, f"{amount}{amount_width(resolved)}"],
            fee_limit=fee_limit, observed=observed,
        ))

    # ── Managed spending ─────────────────────────────────────────────────────

    def _spend(
        self, program: str, function: str, inputs: Sequence[str], *, imports: Sequence[str] = (),
        asset_program: str | None = None, private_fee: bool = False, preferred_fee_record: str | None = None,
        fee_limit: int | None | _Default = DEFAULT, observed: list[str] | None = None,
    ) -> Step[str]:
        """Prove and broadcast one call, reserving its inputs first when management is enabled.

        Returns the transaction id. With management enabled the journal
        records the reservation, the proved transaction, and the submission;
        a failure before the transaction is proved releases the reservation,
        a failure afterwards keeps it for reconciliation.
        """
        address = self.address
        ids = record_input_ids(inputs, address)
        limit = self.management.max_fee_microcredits if isinstance(fee_limit, _Default) else fee_limit
        if not self.management.enabled or self.management.store is None:
            prepared = yield BuildTransaction(program, function, tuple(inputs), tuple(imports), private_fee, None)
            if observed is not None:
                observed.append(prepared.id)
            return (yield Broadcast(prepared.transaction))
        # Acquire every input nonce together before any proving starts.
        checkpoint = open_spend(
            self.management.store, self.scope, lambda: self.scope, ids, program, function,
            asset_program=asset_program, fee_limit=limit,
        )
        selector: FeeRecordSelector | None = None
        try:
            if private_fee:
                credits = yield from self.get_record_inventory(RecordAsset(CREDITS_PROGRAM))
                preferred = inventory_record_id({"record_plaintext": preferred_fee_record}) if preferred_fee_record else None
                selector = checkpoint.use_fee_candidates(credits.available, preferred)
            prepared = yield BuildTransaction(program, function, tuple(inputs), tuple(imports), private_fee, selector)
            checkpoint.prepared(prepared.id, prepared.transaction)
            if observed is not None:
                observed.append(prepared.id)
            try:
                tx_id = yield Broadcast(prepared.transaction)
            except BaseException as exc:
                checkpoint.broadcast_failed(exc)
                raise
            checkpoint.submitted(tx_id)
            return tx_id
        except BaseException as exc:
            # Nothing was broadcast before the prepared checkpoint; afterwards
            # the outcome is uncertain and the reservations stay unless the
            # node refused the transaction outright.
            if not checkpoint.is_prepared:
                checkpoint.discard()
            if isinstance(exc, RecordReservationConflictError):
                # A nested write or a callback raised its own conflict after this
                # spend acquired its inputs; it must not authorise a retry.
                raise InventoryError("Record spending failed after acquiring inputs") from exc
            raise

    def managed_write(
        self, program: str, function: str, inputs: Sequence[str], *, imports: Sequence[str] = (),
        private_fee: bool = False, fee_record: str | None = None,
    ) -> Step[str]:
        """Run one caller-built write through the journal; used by the facade's managed verbs."""
        return (yield from self._spend(
            program, function, inputs, imports=imports, private_fee=private_fee or fee_record is not None,
            preferred_fee_record=fee_record,
        ))

    # ── Plan execution and recovery ──────────────────────────────────────────

    def rebalance_record_inventory(
        self, plan: InventoryPlan, *, timeout: float = 120.0, poll_interval: float = 1.0,
        max_fee_microcredits: int | None = None, cancelled: Callable[[], bool] | None = None,
    ) -> Step[InventoryResult]:
        store = self._store("Inventory execution")
        # Rebuild instead of trusting caller-supplied amounts or dependencies.
        rebuilt = build_inventory_plan(
            scope=plan.scope, asset=plan.asset, target=plan.target, inputs=plan.inputs,
            max_transactions=plan.max_transactions, token_join=plan.token_join, join_programs=plan.join_programs,
        )
        if rebuilt != plan:
            raise InventoryError("Inventory plan was modified; plan again")
        if plan.scope != self.scope:
            raise InventoryError("Inventory plan belongs to another account or chain")
        if timeout <= 0 or poll_interval <= 0:
            raise InventoryError("timeout and poll_interval must be positive")
        if max_fee_microcredits is not None and max_fee_microcredits < 0:
            raise InventoryError("Fee budget cannot be negative")
        transaction_ids: list[str] = []
        completed = 0
        # Plan inputs already use nonces; symbolic step outputs acquire real
        # nonces only after confirmation and scanner visibility.
        references = {item.id: item.id for item in plan.inputs}
        planned = {item.id: item for item in plan.inputs}
        for step in plan.steps:
            planned.update({item.id: item for item in step.outputs})
        fees = 0

        def check() -> None:
            if cancelled is not None and cancelled():
                raise InventoryError("Inventory execution cancelled; unresolved reservations retained")
            if self.scope != plan.scope:
                raise InventoryError("Account or chain changed during inventory execution")

        try:
            capabilities = (yield from self._validate_asset(plan.asset)) if plan.steps else None
            for step in plan.steps:
                check()
                # Rescan before every step so a competing spend or a changed
                # amount stops execution before a dependent proof is built.
                inventory = yield from self.get_record_inventory(plan.asset)
                inputs: list[InventoryRecord] = []
                for ref in step.inputs:
                    record = next((item for item in inventory.available if item.id == references.get(ref)), None)
                    expected = planned.get(ref)
                    if record is None or expected is None or record.amount != expected.amount:
                        raise InventoryError("Inventory changed; reconcile and plan again")
                    inputs.append(record)
                check()
                # The stricter of the remaining run budget and the per-transaction ceiling.
                remaining = None if max_fee_microcredits is None else max_fee_microcredits - fees
                if remaining is not None and remaining <= 0:
                    # Stop before reserving or proving a step the budget cannot pay for.
                    raise InventoryError("Fee budget exhausted; remaining steps were not submitted")
                per_transaction = self.management.max_fee_microcredits
                if remaining is None:
                    limit = per_transaction
                elif per_transaction is None or remaining < per_transaction:
                    limit = remaining
                else:
                    limit = per_transaction
                if step.kind == "join":
                    tx_id = yield from self.join_records(
                        plan.asset, inputs, step.join_program if step.join_program else (None if plan.token_join else False),
                        token_join=plan.token_join, fee_limit=limit, observed=transaction_ids,
                    )
                else:
                    assert step.amount is not None
                    tx_id = yield from self.split_record(plan.asset, inputs[0], step.amount, fee_limit=limit, observed=transaction_ids)
                if tx_id not in transaction_ids:
                    transaction_ids.append(tx_id)
                entry = next((item for item in store.list(plan.scope) if item.transaction_id == tx_id and item.fee_record_id is None), None)
                if entry is None:
                    raise InventoryError("Submitted transaction is missing its reservation")
                fees += int(entry.fee_microcredits or "0")
                deadline = self.clock() + timeout
                confirmed: Mapping[str, Any] | None = None
                # A missing transaction may still be propagating; keep its
                # reservation while polling and stop if acceptance cannot be
                # established within the deadline.
                while confirmed is None:
                    check()
                    if self.clock() >= deadline:
                        raise InventoryError(f"Timed out confirming {tx_id}; reservation retained")
                    confirmed = yield ConfirmedTransaction(tx_id)
                    if confirmed is None:
                        yield Sleep(poll_interval)
                if confirmed.get("status") != "accepted":
                    if confirmed.get("status") == "rejected":
                        store.update(plan.scope, entry.id, {"status": "rejected", "transaction": None})
                    raise InventoryError(f"Transaction {tx_id} was not accepted")
                store.update(plan.scope, entry.id, {"status": "confirmed", "transaction": None})
                commitments = inventory_output_commitments(_dict(confirmed.get("transaction")), plan.asset, step, plan.token_join)
                if len(commitments) != len(step.outputs):
                    raise InventoryError("Unexpected confirmed inventory output count")
                # The scanner must return these exact outputs, not an unrelated deposit.
                outputs: list[InventoryRecord] = []
                while len(outputs) != len(commitments):
                    check()
                    if self.clock() >= deadline:
                        raise InventoryError(f"Scanner has not indexed {tx_id}; reconcile before continuing")
                    records = yield from self._find_records(plan.asset.program)
                    outputs = []
                    for commitment in commitments:
                        record = next((item for item in records if str(item.get("commitment")) == commitment), None)
                        decoded = decode_record(self.address, plan.asset, record, capabilities) if record is not None and capabilities else None
                        if decoded is not None:
                            outputs.append(decoded)
                    if len(outputs) != len(commitments):
                        yield Sleep(poll_interval)
                # Bind each planned output to the scanned nonce and amount before
                # a dependent step may use it.
                for output, expected in zip(outputs, step.outputs):
                    if output.amount != expected.amount:
                        raise InventoryError("Confirmed output amount differs from the plan")
                    references[expected.id] = output.id
                completed += 1
        except Exception as exc:  # noqa: BLE001 - every interruption is reported, never swallowed
            # Earlier accepted steps stay on chain; report progress and let
            # reconciliation settle any pending spend before a new plan.
            return InventoryResult("interrupted", tuple(transaction_ids), completed, str(exc))
        return InventoryResult("complete", tuple(transaction_ids), completed)

    def reconcile_record_inventory(self, *, rebroadcast: bool = False, release_reserved: bool = False) -> Step[list[RecordSpend]]:
        store = self._store("Reconciliation")
        scope = self.scope
        for entry in store.list(scope):
            if entry.status in ("confirmed", "rejected", "cancelled"):
                continue
            if entry.transaction_id is None:
                # A reserved entry never reached the prepared checkpoint, so
                # nothing was broadcast. It is released only on request, because
                # another process may still be proving against it.
                if release_reserved and entry.status == "reserved":
                    store.update(scope, entry.id, {"status": "cancelled"})
                continue
            # Only a definitive chain result changes the reservation; a missing
            # transaction does not prove a prepared or submitted one failed.
            confirmed = yield ConfirmedTransaction(entry.transaction_id)
            if confirmed is not None:
                status = confirmed.get("status")
                if status not in ("accepted", "rejected"):
                    raise InventoryError("Unexpected transaction confirmation response")
                store.update(scope, entry.id, {"status": "confirmed" if status == "accepted" else "rejected", "transaction": None})
                continue
            if rebroadcast and entry.transaction is not None:
                # Reuse the saved proof and id; rebuilding could create a second
                # transaction while the first is still unresolved.
                tx_id = yield Broadcast(entry.transaction)
                if tx_id != entry.transaction_id:
                    raise InventoryError("Rebroadcast returned a different transaction id")
                store.update(scope, entry.id, {"status": "submitted"})
        return store.list(scope)

    def auto_join(
        self, asset: RecordAsset, *, min_amount: int | None = None, max_transactions: int | None = None,
        timeout: float = 120.0, poll_interval: float = 1.0, max_fee_microcredits: int | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Step[InventoryResult]:
        inventory = yield from self.get_record_inventory(asset)
        if min_amount is not None and min_amount <= 0:
            raise InventoryError("min_amount must be positive")
        records = list(inventory.available)
        if min_amount is not None:
            # Keep a sufficient existing record; otherwise take the largest
            # records until they cover the amount and leave the rest untouched.
            if any(item.amount >= min_amount for item in records):
                return InventoryResult("complete")
            records.sort(key=lambda item: (-item.amount, item.id))
            selected: list[InventoryRecord] = []
            total = 0
            for item in records:
                if total >= min_amount:
                    break
                selected.append(item)
                total += item.amount
            if total < min_amount:
                raise InventoryError("Insufficient available balance for auto_join")
            records = selected
        if len(records) <= 1:
            return InventoryResult("complete")
        helpers: tuple[RecordJoinProgram, ...] = ()
        if len(records) > 2:
            helpers = yield from self._join_programs(inventory.asset, self.management.token_join)
        plan = build_inventory_plan(
            scope=self.scope, asset=inventory.asset, target=InventoryTarget(records=1),
            inputs=[PlannedRecord(item.id, item.amount) for item in records], max_transactions=max_transactions,
            token_join=self.management.token_join, join_programs=helpers,
        )
        return (yield from self.rebalance_record_inventory(
            plan, timeout=timeout, poll_interval=poll_interval, max_fee_microcredits=max_fee_microcredits, cancelled=cancelled,
        ))


__all__ = [
    "DEFAULT", "Broadcast", "BuildTransaction", "ConfirmedTransaction", "FeeRecordSelector", "FindRecords",
    "InventoryEngine", "Management", "PreparedTransaction", "ProgramSource", "Request", "Sleep", "SpendCheckpoint", "Step",
    "WaitForScannerSync", "inventory_output_commitments", "open_spend", "positive_integer", "record_input_ids",
    "transaction_fee",
]
