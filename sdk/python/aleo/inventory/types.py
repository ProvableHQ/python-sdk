"""Core types for private record inventory management.

A private balance is held in encrypted records. These types describe one
asset's spendable records, the layout a caller wants, the joins and splits
that reach it, and the journal entries that keep participating writers from
spending the same record. They hold no keys. A plan or journal entry carries
nonces and amounts, never record plaintext, so either can be stored or shared
without exposing a balance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, Sequence, TypedDict

from .._client_common import AleoError

Standard = Literal["credits", "arc20", "arc22"]
AmountType = Literal["u8", "u16", "u32", "u64", "u128"]
Distribution = Literal["preserve", "balanced"]
InputMode = Literal["records", "program_identifier_then_records"]
StepKind = Literal["join", "split"]
RecordSpendStatus = Literal["reserved", "prepared", "submitted", "confirmed", "rejected", "cancelled"]
ResultStatus = Literal["complete", "interrupted"]

#: Most records a target may ask for.
MAX_TARGET_RECORDS = 1000
#: Most operations a plan may request.
MAX_PLAN_TRANSACTIONS = 10_000
#: Largest input count a static join helper exposes (``join_16``).
MAX_STATIC_JOIN = 16
#: Largest input count a dynamic ARC join router exposes (``join_15``).
MAX_DYNAMIC_JOIN = 15


class InventoryError(AleoError):
    """A record inventory operation could not proceed; the message says why."""


class InventoryPlanError(InventoryError):
    """A target cannot be reached from the available records within the configured bounds."""


class RecordReservationConflictError(InventoryError):
    """An execution input was already reserved or spent when the write tried to acquire it.

    Raised before signing or proving. The failed attempt holds no reservation
    and has submitted nothing, so the caller may select different inputs and
    try again. Proving, broadcast, and confirmation failures never raise this
    error, and a conflict raised by a nested write after the outer write
    acquired its inputs is reported as a plain :class:`InventoryError`.

    Attributes:
        record_ids: Execution input nonces involved in the failed acquisition.
            Fee-record contention is not reported here.
    """

    def __init__(self, record_ids: Sequence[str]) -> None:
        super().__init__("Record inventory changed: an input is already reserved or spent")
        self.record_ids: tuple[str, ...] = tuple(record_ids)


@dataclass(frozen=True)
class RecordAsset:
    """Select the program and fungible record balance to manage.

    Only ``program`` is required. The deployed program's ``join``/``split``
    interface supplies the record name and amount field; pass overrides for a
    program whose balance record is not named ``Token`` or whose amount field
    is not ``amount``. ``credits.aleo`` always uses ``credits.microcredits``.

    Attributes:
        program: Deployed program id, such as ``"credits.aleo"``.
        standard: Optional interface hint (``"credits"``, ``"arc20"``,
            ``"arc22"``), checked against the deployed code. Omit it to let the
            SDK detect the standard from the program.
        record_name: Record whose balance is managed. Defaults to ``credits``
            for ``credits.aleo``, otherwise to the record the program's ``join``
            accepts.
        amount_field: Unsigned integer field read as the balance. Defaults to
            ``amount``, then ``microcredits``. A named field must exist on the
            record; no other field is substituted.
        amount_type: Declared width of the amount field, set once the asset
            has been resolved against the program. Bounds every amount the
            planner accepts.
    """

    program: str
    standard: Standard | None = None
    record_name: str | None = None
    amount_field: str | None = None
    amount_type: AmountType | None = None


@dataclass(frozen=True, kw_only=True)
class ResolvedRecordAsset(RecordAsset):
    """A :class:`RecordAsset` whose record, amount field, and width were read from the program.

    Returned in :attr:`RecordInventorySnapshot.asset` and carried by plans so
    execution spends the same record and field the caller reviewed.
    """

    # The base class makes these optional; a resolved asset requires them. The
    # keyword-only override is what pyright flags, and it is intended.
    record_name: str  # pyright: ignore[reportGeneralTypeIssues]
    amount_field: str  # pyright: ignore[reportGeneralTypeIssues]
    amount_type: AmountType  # pyright: ignore[reportGeneralTypeIssues]


@dataclass(frozen=True)
class TokenJoinRouter:
    """Select a dynamic ARC join router explicitly.

    A router exposes ``join_2`` through ``join_15`` and joins records from any
    ARC-20 or ARC-22 token whose identifier is passed as the first input.
    Omit the router to let the SDK pick the deployed helper for the asset and
    network; set it to point at a custom deployment.

    Attributes:
        program: Router program id on the client's network.
        max_records: Most inputs one transaction may join, from 2 through 15.
            Defaults to 15.
    """

    program: str
    max_records: int = 15


@dataclass(frozen=True)
class RecordJoinProgram:
    """Describe a deployed helper that joins several records in one transaction.

    Attributes:
        program: Helper program id on the selected network.
        input_mode: ``"records"`` when ``join_N`` takes only records (credits
            and dedicated stablecoin helpers); ``"program_identifier_then_records"``
            when the token program identifier comes first (dynamic ARC routers).
        counts: Input counts the helper supports, each from 2 through 16
            (15 for dynamic routers).
    """

    program: str
    input_mode: InputMode
    counts: tuple[int, ...]


@dataclass(frozen=True)
class InventoryRecord:
    """One spendable record with its decoded amount.

    Attributes:
        id: The record nonce (``…group``), stable across scanners and the
            identity used in plans and journal entries.
        amount: Balance in integer base units, bounded by the asset's declared
            width (u64 for credits, u128 for ARC tokens).
        record: The scanner record this amount was decoded from. Its
            ``record_plaintext`` is what a join or split spends; it stays in
            memory and is never written to a plan or journal.
    """

    id: str
    amount: int
    record: Mapping[str, Any]

    @property
    def plaintext(self) -> str:
        """The decrypted record literal a transaction input needs."""
        return str(self.record.get("record_plaintext", ""))

    @property
    def commitment(self) -> str | None:
        """The commitment the scanner indexes this record under, when reported."""
        value = self.record.get("commitment")
        return None if value is None else str(value)

    @property
    def tag(self) -> str | None:
        """The record tag, used to recognise a consumed fee record in a proved transaction."""
        value = self.record.get("tag")
        return None if value is None else str(value)


@dataclass(frozen=True)
class InventoryTarget:
    """Describe the record layout a caller wants.

    Choose ``preserve`` to keep records that already fit and only add or
    consolidate as needed; choose ``balanced`` to reshape the inventory into
    roughly equal amounts, which costs more transactions.

    Attributes:
        records: Desired record count, from 1 through 1000.
        min_record_amount: Smallest acceptable output in base units. Defaults
            to 1.
        distribution: ``"preserve"`` (default) or ``"balanced"``.
        tolerance_bps: Allowed deviation from equal amounts under
            ``balanced``, in basis points from 0 through 10000. Defaults to
            1000 (10%).
    """

    records: int
    min_record_amount: int = 1
    distribution: Distribution = "preserve"
    tolerance_bps: int = 1000


@dataclass(frozen=True)
class PlannedRecord:
    """A record identity and amount as a plan sees it.

    Attributes:
        id: A scanned record nonce for plan inputs, or a symbolic
            ``step:<n>:<i>`` reference for an output a later step consumes.
        amount: Predicted amount in base units.
    """

    id: str
    amount: int


@dataclass(frozen=True)
class InventoryStep:
    """One join or split in a plan, with references to earlier planned outputs.

    Attributes:
        kind: ``"join"`` or ``"split"``.
        inputs: Record nonces or earlier step output references. A native
            join consumes two; a helper consumes up to 16 (15 for dynamic
            routers).
        outputs: Predicted outputs. None is spendable until the step is
            confirmed and its records appear in the scanner.
        amount: The split's first output amount; ``None`` for joins.
        deduction: Credits the split itself removes from its outputs
            (10,000 microcredits for ``credits.aleo``), separate from the
            transaction fee.
        join_program: Helper captured for this join. ``None`` means the
            native join, or the plan's router for ARC tokens.
    """

    kind: StepKind
    inputs: tuple[str, ...]
    outputs: tuple[PlannedRecord, ...]
    amount: int | None = None
    deduction: int = 0
    join_program: RecordJoinProgram | None = None


@dataclass(frozen=True)
class InventoryPlan:
    """A reviewable sequence of joins and splits bound to one account and chain.

    Review ``steps`` and ``deduction`` before passing the plan to
    ``rebalance_record_inventory``. Execution rebuilds the plan from its
    inputs and refuses one whose steps were edited.

    Attributes:
        scope: The chain and account the plan belongs to.
        asset: Program with its resolved record name, amount field, and width.
        target: Requested count, minimum amount, and distribution.
        inputs: Eligible records at planning time, as nonces and amounts.
        steps: Ordered operations; later steps may consume earlier outputs.
        deduction: Total credits removed by splits, excluding network fees.
        max_transactions: Operation ceiling the plan was built under.
        token_join: Dynamic router captured for ARC joins, when configured.
        join_programs: Deployed helpers available at planning time; ``None``
            or empty means native pairwise joins.
    """

    scope: str
    asset: RecordAsset
    target: InventoryTarget
    inputs: tuple[PlannedRecord, ...]
    steps: tuple[InventoryStep, ...]
    deduction: int
    max_transactions: int
    token_join: TokenJoinRouter | None = None
    join_programs: tuple[RecordJoinProgram, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise the plan to JSON-compatible data for review or storage.

        Amounts are rendered as decimal strings so a u128 balance survives
        JSON consumers that cannot represent large integers.
        """
        def record(item: PlannedRecord) -> dict[str, str]:
            return {"id": item.id, "amount": str(item.amount)}

        def helper(item: RecordJoinProgram) -> dict[str, Any]:
            return {"program": item.program, "input_mode": item.input_mode, "counts": list(item.counts)}

        asset: dict[str, Any] = {"program": self.asset.program}
        for name in ("standard", "record_name", "amount_field", "amount_type"):
            value = getattr(self.asset, name)
            if value is not None:
                asset[name] = value
        out: dict[str, Any] = {
            "scope": self.scope,
            "asset": asset,
            "target": {
                "records": self.target.records,
                "min_record_amount": str(self.target.min_record_amount),
                "distribution": self.target.distribution,
                "tolerance_bps": self.target.tolerance_bps,
            },
            "inputs": [record(item) for item in self.inputs],
            "steps": [
                {
                    "kind": step.kind,
                    "inputs": list(step.inputs),
                    "outputs": [record(item) for item in step.outputs],
                    **({"amount": str(step.amount)} if step.amount is not None else {}),
                    "deduction": str(step.deduction),
                    **({"join_program": helper(step.join_program)} if step.join_program else {}),
                }
                for step in self.steps
            ],
            "deduction": str(self.deduction),
            "max_transactions": self.max_transactions,
        }
        if self.token_join is not None:
            out["token_join"] = {"program": self.token_join.program, "max_records": self.token_join.max_records}
        if self.join_programs:
            out["join_programs"] = [helper(item) for item in self.join_programs]
        return out

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "InventoryPlan":
        """Rebuild a plan saved with :meth:`to_dict`.

        The result carries a plain :class:`RecordAsset`; execution re-resolves
        it against the deployed program before spending anything.
        """
        def record(item: Mapping[str, Any]) -> PlannedRecord:
            return PlannedRecord(str(item["id"]), int(item["amount"]))

        def helper(item: Mapping[str, Any]) -> RecordJoinProgram:
            return RecordJoinProgram(str(item["program"]), item["input_mode"], tuple(int(c) for c in item["counts"]))

        asset_data = data["asset"]
        target_data = data["target"]
        router = data.get("token_join")
        helpers = data.get("join_programs")
        return InventoryPlan(
            scope=str(data["scope"]),
            asset=RecordAsset(
                program=str(asset_data["program"]),
                standard=asset_data.get("standard"),
                record_name=asset_data.get("record_name"),
                amount_field=asset_data.get("amount_field"),
                amount_type=asset_data.get("amount_type"),
            ),
            target=InventoryTarget(
                records=int(target_data["records"]),
                min_record_amount=int(target_data.get("min_record_amount", 1)),
                distribution=target_data.get("distribution", "preserve"),
                tolerance_bps=int(target_data.get("tolerance_bps", 1000)),
            ),
            inputs=tuple(record(item) for item in data["inputs"]),
            steps=tuple(
                InventoryStep(
                    kind=step["kind"],
                    inputs=tuple(str(item) for item in step["inputs"]),
                    outputs=tuple(record(item) for item in step["outputs"]),
                    amount=None if step.get("amount") is None else int(step["amount"]),
                    deduction=int(step.get("deduction", 0)),
                    join_program=helper(step["join_program"]) if step.get("join_program") else None,
                )
                for step in data["steps"]
            ),
            deduction=int(data["deduction"]),
            max_transactions=int(data["max_transactions"]),
            token_join=None if router is None else TokenJoinRouter(str(router["program"]), int(router.get("max_records", 15))),
            join_programs=None if not helpers else tuple(helper(item) for item in helpers),
        )


@dataclass
class RecordSpend:
    """A journal entry: one operation's reserved records and its progress.

    Entries hold nonces, program and function names, and the proved
    transaction; never keys or record plaintext. A store may persist them
    anywhere.

    Attributes:
        id: Store-assigned identifier, unique within the store.
        scope: Chain and account the entry belongs to.
        records: Reserved record nonces, including the private fee record
            when one was pinned.
        program: Program the transaction calls.
        function: Transition name.
        status: Last persisted lifecycle boundary.
        created_at: Unix time in milliseconds.
        account_type: ``"local"`` for keys held by the SDK. Kept for journal
            compatibility with other Veil clients.
        asset_program: Underlying token for a routed join; ``None`` for
            native operations.
        fee_record_id: Nonce of the private fee record. It stays excluded
            after a rejected execution because the chain can still charge it.
        transaction_id: Known transaction id, retained on an uncertain
            submission.
        transaction: The proved transaction as JSON data, kept from
            preparation until confirmation so recovery can resubmit it
            without building a second spend.
        fee_microcredits: Total network fee as a decimal string.
    """

    id: str
    scope: str
    records: tuple[str, ...]
    program: str
    function: str
    status: RecordSpendStatus
    created_at: int
    account_type: str = "local"
    asset_program: str | None = None
    fee_record_id: str | None = None
    transaction_id: str | None = None
    transaction: dict[str, Any] | None = None
    fee_microcredits: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise the entry with Veil's field names, for a journal shared with other clients."""
        out: dict[str, Any] = {
            "id": self.id, "scope": self.scope, "records": list(self.records), "program": self.program,
            "function": self.function, "status": self.status, "createdAt": self.created_at,
            "accountType": self.account_type,
        }
        if self.asset_program is not None:
            out["assetProgram"] = self.asset_program
        if self.fee_record_id is not None:
            out["feeRecordId"] = self.fee_record_id
        if self.transaction_id is not None:
            out["transactionId"] = self.transaction_id
        if self.transaction is not None:
            out["transaction"] = self.transaction
        if self.fee_microcredits is not None:
            out["feeMicrocredits"] = self.fee_microcredits
        return out

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "RecordSpend":
        """Rebuild an entry serialised with :meth:`to_dict`."""
        return RecordSpend(
            id=str(data["id"]), scope=str(data["scope"]), records=tuple(str(r) for r in data["records"]),
            program=str(data["program"]), function=str(data["function"]), status=data["status"],
            created_at=int(data["createdAt"]), account_type=str(data.get("accountType", "local")),
            asset_program=data.get("assetProgram"), fee_record_id=data.get("feeRecordId"),
            transaction_id=data.get("transactionId"), transaction=data.get("transaction"),
            fee_microcredits=data.get("feeMicrocredits"),
        )


class RecordSpendPatch(TypedDict, total=False):
    """Fields :meth:`RecordInventoryStore.update` may change.

    A key that is present replaces the stored value; ``transaction: None``
    removes a stored transaction once it is no longer needed for recovery.
    ``records`` and ``fee_record_id`` change together when a private fee
    record is folded into the entry that holds the execution inputs, so one
    entry carries everything a crash must keep together.
    """

    status: RecordSpendStatus
    records: tuple[str, ...]
    fee_record_id: str | None
    transaction_id: str
    transaction: dict[str, Any] | None
    fee_microcredits: str


class RecordInventoryStore(Protocol):
    """Coordinate record spending across participating writers.

    Implement this protocol over any backend: a database shared by several
    processes, a key-value service, or a file. The SDK ships an in-memory
    implementation for a single process.

    Two guarantees are required. ``acquire`` MUST reserve every nonce in the
    entry or none of them, atomically with respect to other callers. Each
    mutation MUST be finished before the method returns, and ``update`` MUST
    apply every key present in the patch in one step. Entries whose status
    is not ``rejected`` or ``cancelled`` keep their records excluded from
    selection; a rejected entry keeps only its ``fee_record_id`` excluded.
    """

    def list(self, scope: str) -> list[RecordSpend]:
        """Return every entry recorded under *scope*."""
        ...

    def acquire(self, entry: RecordSpend) -> RecordSpend | None:
        """Reserve the entry's records and assign its id, or return ``None`` when any is already held."""
        ...

    def update(self, scope: str, entry_id: str, patch: RecordSpendPatch) -> None:
        """Persist a checkpoint for an existing entry."""
        ...


@dataclass(frozen=True)
class RecordManagementConfig:
    """Configure record reservations and recovery for a client.

    With ``enabled=True`` every write on the client reserves its record
    inputs before proving, saves the proved transaction before broadcast,
    and journals submission, so participating writers never spend the same
    record and an interrupted operation can be reconciled. Disabled, the
    configuration still supplies a chain identity and router for reads,
    plans, and single joins; nothing is journaled.

    Attributes:
        enabled: Turn reservations and recovery on. Defaults to off.
        store: Journal shared by participating writers. An enabled
            configuration without one uses a process-local in-memory store,
            which is lost on exit.
        chain_id: Stable chain identity for the reservation scope. Defaults
            to the client's network name. A devnode or custom chain MUST set a
            distinct value so two local chains do not share a scope.
        max_fee_microcredits: Ceiling checked against the proved transaction's
            fee before broadcast; a transaction over it is cancelled after
            proving. Under delegated proving with the fee master paying, the
            fee read is the prover's, so leave this ``None`` unless the
            account pays its own fees. ``None`` means no limit.
        token_join: ``None`` selects deployed helpers by asset and network;
            ``False`` uses native pairwise joins everywhere; a
            :class:`TokenJoinRouter` names a dynamic ARC router explicitly.
        wait_for_sync: Wait for the record scanner to report itself synced
            before each inventory read, so a lagging scanner does not cause
            needless joins. Defaults to off.
        sync_timeout: Seconds to wait for scanner synchronisation. Defaults
            to 300.
    """

    enabled: bool = False
    store: RecordInventoryStore | None = None
    chain_id: str | None = None
    max_fee_microcredits: int | None = None
    token_join: TokenJoinRouter | Literal[False] | None = None
    wait_for_sync: bool = False
    sync_timeout: float = 300.0


@dataclass(frozen=True)
class InventoryResult:
    """Outcome of executing a plan or an automatic join.

    Attributes:
        status: ``"complete"`` when every step confirmed, otherwise
            ``"interrupted"``. No accepted step is rolled back.
        transaction_ids: Every prepared or submitted transaction id,
            including ones whose acceptance is uncertain.
        completed_steps: Steps whose outputs were confirmed and seen by the
            scanner.
        reason: Why execution stopped; ``None`` on completion. After an
            interruption, reconcile, rescan, and plan again; the old plan
            describes records that completed steps have already spent.
    """

    status: ResultStatus
    transaction_ids: tuple[str, ...] = ()
    completed_steps: int = 0
    reason: str | None = None


@dataclass(frozen=True)
class RecordInventorySnapshot:
    """Spendable records of one asset, separated from records held by other operations.

    Attributes:
        asset: The resolved record name, amount field, and width.
        available: Eligible records no participating operation holds.
        reserved: Eligible records the journal excludes. A low available
            balance with a large reserved one means funds are committed, not
            missing.
        balance: Sum of the available amounts in base units.
    """

    asset: ResolvedRecordAsset
    available: tuple[InventoryRecord, ...]
    reserved: tuple[InventoryRecord, ...] = field(default=())
    balance: int = 0


__all__ = [
    "AmountType", "Distribution", "InputMode", "InventoryError", "InventoryPlan", "InventoryPlanError",
    "InventoryRecord", "InventoryResult", "InventoryStep", "InventoryTarget", "MAX_DYNAMIC_JOIN",
    "MAX_PLAN_TRANSACTIONS", "MAX_STATIC_JOIN", "MAX_TARGET_RECORDS", "PlannedRecord", "RecordAsset",
    "RecordInventorySnapshot", "RecordInventoryStore", "RecordJoinProgram", "RecordManagementConfig",
    "RecordReservationConflictError", "RecordSpend", "RecordSpendPatch", "RecordSpendStatus",
    "ResolvedRecordAsset", "ResultStatus", "Standard", "StepKind", "TokenJoinRouter",
]
