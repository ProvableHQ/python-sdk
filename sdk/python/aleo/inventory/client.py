"""Record inventory over an ``Aleo`` or ``AsyncAleo`` facade.

:class:`RecordInventory` and :class:`AsyncRecordInventory` expose the same
seven actions. Both drive the generators in :mod:`aleo.inventory._engine`;
the only difference is whether a request to the facade is called or awaited.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import threading
import time
from contextlib import contextmanager
from typing import Any, Awaitable, Callable, Generator, Literal, Mapping, Sequence, cast

from .._client_common import async_retry_with_backoff, retry_with_backoff
from .._scanner_common import net_module
from ..facade.errors import ProgramNotFound, TransactionNotFound
from ._engine import (
    Broadcast, BuildTransaction, ConfirmedTransaction, FindRecords, InventoryEngine, Management, PreparedTransaction,
    ProgramSource, Request, Sleep, SpendCheckpoint, Step, open_spend, record_input_ids,
)
from .assets import CREDITS_PROGRAM
from .interface import parse_program_interface
from .store import MemoryRecordInventoryStore
from .types import (
    InventoryError, InventoryPlan, InventoryRecord, InventoryResult, InventoryTarget, RecordAsset,
    RecordInventorySnapshot, RecordInventoryStore, RecordJoinProgram, RecordManagementConfig, RecordSpend,
)

Proving = Literal["delegated", "local"]
Builder = Callable[[BuildTransaction], PreparedTransaction]
AsyncBuilder = Callable[[BuildTransaction], "PreparedTransaction | Awaitable[PreparedTransaction]"]


class _InventoryBase:
    """Shared construction, scope, and journal access for both clients."""

    def __init__(
        self, aleo: Any, management: RecordManagementConfig | None = None, *, proving: Proving = "delegated",
        priority_fee: int = 0, pay_own_fee: bool = False,
    ) -> None:
        if proving not in ("delegated", "local"):
            raise ValueError("proving must be 'delegated' or 'local'")
        self._aleo = aleo
        self.config: RecordManagementConfig = management or RecordManagementConfig()
        self._store: RecordInventoryStore | None = self.config.store or (
            MemoryRecordInventoryStore() if self.config.enabled else None
        )
        self._management = Management.resolve(self.config, str(aleo.network_name), self._store)
        self.proving: Proving = proving
        self.priority_fee = priority_fee
        self.pay_own_fee = pay_own_fee

    @property
    def enabled(self) -> bool:
        """Whether writes reserve records and journal their progress."""
        return self._management.enabled

    @property
    def store(self) -> RecordInventoryStore | None:
        """The journal in use, or ``None`` while management is disabled."""
        return self._management.store

    @property
    def network(self) -> str:
        """The bound client's network name."""
        return str(self._aleo.network_name)

    def _account(self, account: Any) -> Any:
        resolved = account if account is not None else self._aleo.default_account
        if resolved is None:
            raise InventoryError("Inventory requires an account; pass account= or set aleo.default_account")
        return resolved

    def _engine(self, account: Any) -> InventoryEngine:
        if account is None:
            # Re-read the default account on every use so a change mid-run is noticed.
            address: Callable[[], str] = lambda: str(self._account(None).address)
        else:
            fixed = str(account.address)
            address = lambda: fixed
        return InventoryEngine(self.network, address, self._management)

    def scope(self, account: Any = None) -> str:
        """The reservation namespace (chain id and address) for *account* or the default account."""
        return self._engine(account).scope

    def reserved_record_ids(self, account: Any = None) -> set[str]:
        """Record nonces the journal currently excludes from selection.

        Reads the store only. Empty while management is disabled. Pass the
        result as ``exclude_nonces`` to any selector that chooses records
        outside this module.
        """
        return self._engine(account).reserved_record_ids()

    @contextmanager
    def reserve(
        self, program: str, function: str, record_ids: Sequence[str], *, asset_program: str | None = None,
        account: Any = None,
    ) -> Generator[SpendCheckpoint, None, None]:
        """Reserve records for a write this code does not build, and journal its progress.

        Use it around a write that proves and broadcasts through another
        path. Inside the block, call ``checkpoint.prepared`` once the proved
        transaction exists and ``checkpoint.submitted`` after broadcast.
        Leaving the block before ``submitted``, by return or by exception,
        releases the reservation because nothing can have reached the
        network through this checkpoint.

        Raises:
            InventoryError: Management is disabled.
            RecordReservationConflictError: A record is already held.
        """
        checkpoint = self._open_spend(record_ids, program, function, asset_program=asset_program, account=account)
        try:
            yield checkpoint
        finally:
            checkpoint.discard()

    def _open_spend(
        self, record_ids: Sequence[str], program: str, function: str, *, asset_program: str | None, account: Any,
    ) -> SpendCheckpoint:
        store = self._management.store
        if not self._management.enabled or store is None:
            raise InventoryError("Record reservation requires record management to be enabled with a shared store")
        engine = self._engine(account)
        return open_spend(
            store, engine.scope, lambda: engine.scope, record_ids, program, function,
            asset_program=asset_program, fee_limit=self._management.max_fee_microcredits,
        )

    # ── Program registration for local authorization ─────────────────────────

    def _register_sources(self, sources: Mapping[str, str]) -> None:
        """Add fetched programs to the client's process, dependencies first."""
        process = getattr(self._aleo, "process", None)
        if process is None:
            return
        net = net_module(self.network)
        added: set[str] = set()

        def add(program_id: str) -> None:
            if program_id == CREDITS_PROGRAM or program_id in added or program_id not in sources:
                return
            added.add(program_id)
            if process.contains_program(net.ProgramID.from_string(program_id)):
                return
            for dependency in parse_program_interface(sources[program_id]).imports:
                add(dependency)
            process.add_program(net.Program.from_source(sources[program_id]))

        for program_id in sources:
            add(program_id)


def prepared_from_transaction(transaction: Any) -> PreparedTransaction:
    """A proved network ``Transaction`` that has not been broadcast, ready for the journal."""
    return PreparedTransaction(str(transaction.id), json.loads(str(transaction.to_json())))


def prepared_from_result(result: Any) -> PreparedTransaction:
    """The unbroadcast transaction behind a facade ``TransactionResult``, ready for the journal."""
    return prepared_from_transaction(result.raw)


class _Submission:
    """A serialised transaction paired with its checkpoint, in the shape ``network.submit_transaction`` reads."""

    __slots__ = ("raw", "checkpoint")

    def __init__(self, raw: str, checkpoint: SpendCheckpoint | None) -> None:
        self.raw = raw
        self.checkpoint = checkpoint

    def __str__(self) -> str:
        return self.raw


def broadcast_checkpointed(submit_once: Callable[[], Any], checkpoint: Any) -> str:
    """Broadcast a managed transaction, reporting every attempt to its checkpoint.

    *submit_once* performs one broadcast without retrying. Transient node
    errors (5xx) are retried here with the SDK's usual backoff, and the
    checkpoint learns of each failed attempt first, so a refusal that follows
    an attempt with an unknown outcome keeps the reservation instead of
    releasing it. A successful answer is recorded as submitted.
    """
    def attempt() -> str:
        try:
            return str(submit_once())
        except BaseException as exc:
            checkpoint.broadcast_failed(exc)
            raise

    tx_id = retry_with_backoff(attempt)
    checkpoint.submitted(tx_id)
    return tx_id


async def async_broadcast_checkpointed(submit_once: Callable[[], Awaitable[Any]], checkpoint: Any) -> str:
    """Awaited twin of :func:`broadcast_checkpointed`."""
    async def attempt() -> str:
        try:
            return str(await submit_once())
        except BaseException as exc:
            checkpoint.broadcast_failed(exc)
            raise

    tx_id = str(await async_retry_with_backoff(attempt))
    checkpoint.submitted(tx_id)
    return tx_id


def _json_dict(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def prepared_from_payload(payload: Any) -> PreparedTransaction:
    """The unbroadcast transaction inside a delegated prover payload.

    Raises:
        InventoryError: The payload carries no transaction, so there is
            nothing to save before broadcast.
    """
    transaction = _json_dict(_json_dict(payload).get("transaction"))
    if not transaction.get("id"):
        raise InventoryError("The prover returned no transaction to save before broadcast")
    return PreparedTransaction(str(transaction["id"]), transaction)


def managed_inventory(aleo: Any) -> Any:
    """The client's inventory when record management is enabled on it, else ``None``.

    Writers use this to decide whether a write must go through the journal.
    A client without an ``inventory`` attribute, such as a test stub, counts
    as unmanaged.
    """
    inventory = getattr(aleo, "inventory", None)
    return inventory if inventory is not None and getattr(inventory, "enabled", False) else None


def _scanner_synced(status: Any) -> bool:
    report = _json_dict(status)
    return bool(report.get("ok")) and bool(_json_dict(report.get("data")).get("synced"))


class RecordInventory(_InventoryBase):
    """Inspect, reshape, and protect one account's private records.

    Reads go through the client's record provider, writes through its proving
    path (delegated by default, with the fee master paying; ``proving="local"``
    proves in-process and pays a public fee). With management enabled, every
    write reserves its inputs before proving and saves the proved transaction
    before broadcast; the same journal is shared by every writer given the
    same store.

    Args:
        aleo: The bound ``Aleo`` facade.
        management: Reservation and recovery settings. ``None`` disables
            both; reads, plans, and single joins or splits still work.
        proving: ``"delegated"`` (default) or ``"local"``.
        priority_fee: Extra fee in microcredits on every write. Defaults to 0.
        pay_own_fee: Under delegated proving, pay the public fee from the
            account instead of the prover's fee master. Defaults to False.
    """

    def _run(self, step: Step[Any], account: Any, builder: Builder | None = None) -> Any:
        try:
            request = next(step)
            while True:
                try:
                    result = self._perform(request, account, builder)
                except BaseException as exc:  # the engine decides whether a 404 or outage is fatal
                    request = step.throw(exc)
                else:
                    request = step.send(result)
        except StopIteration as done:
            return done.value

    def _perform(self, request: Request, account: Any, builder: Builder | None) -> Any:
        if isinstance(request, ProgramSource):
            try:
                return str(self._aleo.programs.get(request.program_id).source)
            except ProgramNotFound:
                return None
        if isinstance(request, FindRecords):
            provider = self._aleo.record_provider
            if provider is None:
                raise InventoryError("Inventory requires a record provider; register the account with aleo.records or set aleo.record_provider")
            return list(provider.find(account, program=request.program, unspent=True))
        if isinstance(request, BuildTransaction):
            return builder(request) if builder is not None else self._build(request, account)
        if isinstance(request, Broadcast):
            # The facade's broadcast reports each attempt to the checkpoint;
            # the engine records the final outcome itself for other drivers.
            return str(self._aleo.network.submit_transaction(_Submission(json.dumps(request.transaction), request.checkpoint)))
        if isinstance(request, ConfirmedTransaction):
            try:
                return self._aleo.network.get_confirmed_transaction(request.tx_id)
            except TransactionNotFound:
                return None
        if isinstance(request, Sleep):
            time.sleep(request.seconds)
            return None
        deadline = time.monotonic() + request.timeout
        while True:
            if _scanner_synced(self._aleo.records.status()):
                return None
            if time.monotonic() >= deadline:
                raise InventoryError("Record scanner synchronization timed out")
            time.sleep(1.0)

    def _build(self, request: BuildTransaction, account: Any) -> PreparedTransaction:
        self._register_programs([request.program, *request.imports])
        bound = self._aleo.programs.get(request.program).functions[request.function](*request.inputs)
        # The engine has already reserved this call's inputs, so prove through
        # the facade's unmanaged paths: ``build_transaction`` would try to
        # reserve them a second time, and ``delegate`` is only managed when it
        # broadcasts. This is the Python form of Veil's ``withoutManagement``.
        if self.proving == "local":
            transaction = bound._prove_transaction(  # pyright: ignore[reportPrivateUsage]
                account, priority_fee=self.priority_fee, fee_record=request.fee_record, private_fee=request.private_fee,
            )
            return prepared_from_transaction(transaction)
        payload = bound.delegate(
            account, broadcast=False, pay_own_fee=self.pay_own_fee, fee_record=request.fee_record, priority_fee=self.priority_fee,
        )
        return prepared_from_payload(payload)

    def _register_programs(self, program_ids: Sequence[str]) -> None:
        if getattr(self._aleo, "process", None) is None:
            return
        sources: dict[str, str] = {}
        stack = [pid for pid in program_ids if pid != CREDITS_PROGRAM]
        while stack:
            program_id = stack.pop()
            if program_id in sources or program_id == CREDITS_PROGRAM:
                continue
            sources[program_id] = str(self._aleo.programs.get(program_id).source)
            stack.extend(parse_program_interface(sources[program_id]).imports)
        self._register_sources(sources)

    def begin_spend(
        self, program: str, function: str, inputs: Sequence[str], *, account: Any = None,
        private_fee: bool = False, fee_record: str | None = None,
    ) -> SpendCheckpoint:
        """Reserve a call's record inputs before it is proved; the first half of a managed spend.

        The facade's ``build_transaction`` calls this on a managed client.
        Record plaintexts among *inputs* owned by the signer are reserved
        together. With a private fee, the credits inventory is scanned now and
        ``checkpoint.fee_record_selector`` chooses and reserves the fee record
        once the exact fee is known (*fee_record* is preferred when eligible).
        Call ``checkpoint.prepared`` after proving, pass the result to
        ``aleo.network.submit_transaction`` to record the broadcast, or
        ``checkpoint.discard()`` to release a spend that was never sent;
        after a broadcast attempt, ``discard`` does nothing.

        Raises:
            RecordReservationConflictError: An input is already held.
        """
        engine = self._engine(account)
        checkpoint = self._open_spend(record_input_ids(inputs, engine.address), program, function, asset_program=None, account=account)
        if private_fee or fee_record is not None:
            try:
                credits = self.get_record_inventory(RecordAsset(CREDITS_PROGRAM), account=account)
                preferred = record_input_ids([fee_record], engine.address)[0] if fee_record else None
                checkpoint.use_fee_candidates(credits.available, preferred)
            except BaseException:
                checkpoint.discard()
                raise
        return checkpoint

    # ── Actions ──────────────────────────────────────────────────────────────

    def get_record_inventory(self, asset: RecordAsset, *, account: Any = None) -> RecordInventorySnapshot:
        """Read the account's spendable records of one asset.

        Checks the deployed program's record interface, scans the record
        provider, and separates records held by other participating
        operations from available ones. Nothing is signed or submitted.

        Args:
            asset: Program with optional record and amount-field overrides.
            account: Signer whose records to read. Defaults to ``aleo.default_account``.

        Raises:
            InventoryError: The program's interface is incompatible, a record
                arrived without decrypted plaintext, or no record provider is
                configured.

        Example::

            snapshot = aleo.inventory.get_record_inventory(RecordAsset("credits.aleo"))
            print(snapshot.balance, len(snapshot.available), len(snapshot.reserved))
        """
        return self._run(self._engine(account).get_record_inventory(asset), account)

    def plan_record_inventory(
        self, asset: RecordAsset, target: InventoryTarget, *, max_transactions: int | None = None, account: Any = None,
    ) -> InventoryPlan:
        """Plan the joins and splits that reach *target* from the current inventory.

        Reads the scanner and the deployed program, including any join
        helpers deployed on this network. Nothing is reserved, signed, or
        submitted. Review ``plan.steps`` and ``plan.deduction`` (credits a
        native split removes, separate from network fees), then pass the
        plan unchanged to :meth:`rebalance_record_inventory`.

        Args:
            asset: Program with optional record and amount-field overrides.
            target: Desired count, minimum amount, and distribution.
            max_transactions: Most operations the plan may take, 1 through
                10000. Defaults to 100.
            account: Signer; defaults to ``aleo.default_account``.

        Raises:
            InventoryPlanError: The target cannot be reached within the bounds.

        Example::

            plan = aleo.inventory.plan_record_inventory(
                RecordAsset("credits.aleo"),
                InventoryTarget(records=4, min_record_amount=1_000_000, distribution="balanced"),
            )
            for step in plan.steps:
                print(step.kind, len(step.inputs), [o.amount for o in step.outputs])
        """
        return self._run(self._engine(account).plan_record_inventory(asset, target, max_transactions), account)

    def join_records(
        self, asset: RecordAsset, records: Sequence[InventoryRecord], *, join_program: RecordJoinProgram | Literal[False] | None = None,
        account: Any = None,
    ) -> str:
        """Join records into one transaction and return its id.

        Two records use the program's native ``join``. More than two use a
        deployed helper when one accepts that count (up to 16 for credits and
        dedicated stablecoins, 15 for other ARC tokens); pass
        ``join_program=False`` to force the native join. Signs, proves, and
        broadcasts; acceptance is asynchronous.

        Args:
            asset: Program with optional overrides.
            records: Distinct records from :meth:`get_record_inventory`.
            join_program: Helper override, ``False`` for native, ``None`` to select.
            account: Signer; defaults to ``aleo.default_account``.

        Raises:
            InventoryError: Records are not distinct and eligible, the joined
                amount overflows the width, or no helper accepts the count.
            RecordReservationConflictError: An input is already reserved.

        Example::

            snapshot = aleo.inventory.get_record_inventory(asset)
            tx_id = aleo.inventory.join_records(asset, snapshot.available[:2])
            aleo.network.wait_for_transaction(tx_id, timeout=120)
        """
        return self._run(self._engine(account).join_records(asset, records, join_program), account)

    def split_record(self, asset: RecordAsset, record: InventoryRecord, amount: int, *, account: Any = None) -> str:
        """Split one record into two and return the transaction id.

        The first output holds *amount*; the second holds the remainder, less
        10,000 microcredits for ``credits.aleo``. Both outputs must stay
        positive. Signs, proves, and broadcasts; rescan after confirmation
        before spending either output.

        Example::

            tx_id = aleo.inventory.split_record(asset, snapshot.available[0], 500_000)
        """
        return self._run(self._engine(account).split_record(asset, record, amount), account)

    def rebalance_record_inventory(
        self, plan: InventoryPlan, *, timeout: float = 120.0, poll_interval: float = 1.0,
        max_fee_microcredits: int | None = None, cancel: threading.Event | None = None, account: Any = None,
    ) -> InventoryResult:
        """Execute a reviewed plan one confirmed transaction at a time.

        Requires management to be enabled. Each step rescans, reserves its
        inputs, proves, broadcasts, waits for confirmation and for the scanner
        to show the exact outputs, and only then feeds them to the next step.
        No accepted step is rolled back.

        Args:
            plan: The plan from :meth:`plan_record_inventory`, unchanged.
            timeout: Seconds allowed per step for confirmation and scanner
                visibility. Defaults to 120.
            poll_interval: Seconds between polls. Defaults to 1.
            max_fee_microcredits: Total network-fee budget for this run,
                excluding split deductions. ``None`` means no budget.
            cancel: Set the event to stop before the next step; unresolved
                reservations are kept.
            account: Signer; defaults to ``aleo.default_account``.

        Returns:
            ``complete`` when every step confirmed, otherwise ``interrupted``
            with the reason and every known transaction id. After an
            interruption, reconcile, rescan, and plan again.

        Raises:
            InventoryError: Before any submission, when the plan was edited,
                belongs to another scope, or management is disabled.

        Example::

            result = aleo.inventory.rebalance_record_inventory(plan, max_fee_microcredits=1_000_000)
            if result.status == "interrupted":
                aleo.inventory.reconcile_record_inventory()
        """
        return self._run(self._engine(account).rebalance_record_inventory(
            plan, timeout=timeout, poll_interval=poll_interval, max_fee_microcredits=max_fee_microcredits,
            cancelled=cancel.is_set if cancel is not None else None,
        ), account)

    def reconcile_record_inventory(
        self, *, rebroadcast: bool = False, release_reserved: bool = False, account: Any = None,
    ) -> list[RecordSpend]:
        """Settle pending journal entries against the chain after an interruption.

        Reads each unresolved entry's transaction: an accepted one becomes
        ``confirmed``, a rejected one ``rejected``. A transaction the node has
        not seen keeps its reservation. With ``rebroadcast=True`` the saved
        proved transaction is resubmitted with its original id; nothing is
        rebuilt. Requires management to be enabled.

        Args:
            rebroadcast: Resubmit saved proved transactions the node has not seen.
            release_reserved: Cancel entries still in ``reserved`` status. Such
                an entry never reached the prepared checkpoint, so nothing was
                broadcast; set this after a crash, once no other process is
                proving against this journal.
            account: Signer whose scope to reconcile; defaults to ``aleo.default_account``.

        Returns:
            The journal for this scope after reconciliation.

        Example::

            journal = aleo.inventory.reconcile_record_inventory()
            pending = [e for e in journal if e.status not in ("confirmed", "rejected", "cancelled")]
        """
        return self._run(self._engine(account).reconcile_record_inventory(
            rebroadcast=rebroadcast, release_reserved=release_reserved), account)

    def auto_join(
        self, asset: RecordAsset, *, min_amount: int | None = None, max_transactions: int | None = None,
        timeout: float = 120.0, poll_interval: float = 1.0, max_fee_microcredits: int | None = None,
        cancel: threading.Event | None = None, account: Any = None,
    ) -> InventoryResult:
        """Join records until one covers *min_amount*, or consolidate everything when it is omitted.

        A record that already covers the amount means nothing is submitted.
        Otherwise the largest records are joined, through helpers where
        available, and the rest are left untouched. Requires management to be
        enabled; executes like :meth:`rebalance_record_inventory`.

        Raises:
            InventoryError: Before any submission when the available balance
                cannot cover *min_amount*.

        Example::

            result = aleo.inventory.auto_join(RecordAsset("credits.aleo"), min_amount=1_000_000)
        """
        return self._run(self._engine(account).auto_join(
            asset, min_amount=min_amount, max_transactions=max_transactions, timeout=timeout,
            poll_interval=poll_interval, max_fee_microcredits=max_fee_microcredits,
            cancelled=cancel.is_set if cancel is not None else None,
        ), account)

    def managed_write(
        self, program: str, function: str, inputs: Sequence[str], *, imports: Sequence[str] = (), account: Any = None,
        private_fee: bool = False, fee_record: str | None = None, builder: Builder | None = None,
    ) -> str:
        """Run one write through the journal: reserve, prove, checkpoint, broadcast.

        The facade's ``transact`` and ``delegate`` call this on a managed
        client. *builder* proves the call and returns the unbroadcast
        transaction; without one the inventory's own proving settings apply.
        Under a private fee the credits record is chosen once the exact fee is
        known and reserved beside the inputs; *fee_record* is preferred when
        it is still available and sufficient.

        Returns:
            The transaction id the node returned.
        """
        return self._run(self._engine(account).managed_write(
            program, function, inputs, imports=imports, private_fee=private_fee, fee_record=fee_record,
        ), account, builder)


class AsyncRecordInventory(_InventoryBase):
    """Async twin of :class:`RecordInventory` over an ``AsyncAleo`` facade.

    Every action is awaited. Store calls are synchronous and brief; a custom
    store that blocks should wrap its own I/O.
    """

    async def _run(self, step: Step[Any], account: Any, builder: AsyncBuilder | None = None) -> Any:
        try:
            request = next(step)
            while True:
                try:
                    result = await self._perform(request, account, builder)
                except BaseException as exc:
                    request = step.throw(exc)
                else:
                    request = step.send(result)
        except StopIteration as done:
            return done.value

    async def _perform(self, request: Request, account: Any, builder: AsyncBuilder | None) -> Any:
        if isinstance(request, ProgramSource):
            try:
                return str((await self._aleo.programs.get(request.program_id)).source)
            except ProgramNotFound:
                return None
        if isinstance(request, FindRecords):
            provider = self._aleo.record_provider
            if provider is None:
                raise InventoryError("Inventory requires a record provider; register the account with aleo.records or set aleo.record_provider")
            return list(await provider.find(account, program=request.program, unspent=True))
        if isinstance(request, BuildTransaction):
            if builder is None:
                return await self._build(request, account)
            built = builder(request)
            return await built if inspect.isawaitable(built) else built
        if isinstance(request, Broadcast):
            return str(await self._aleo.network.submit_transaction(_Submission(json.dumps(request.transaction), request.checkpoint)))
        if isinstance(request, ConfirmedTransaction):
            try:
                return await self._aleo.network.get_confirmed_transaction(request.tx_id)
            except TransactionNotFound:
                return None
        if isinstance(request, Sleep):
            await asyncio.sleep(request.seconds)
            return None
        deadline = time.monotonic() + request.timeout
        while True:
            if _scanner_synced(await self._aleo.records.status()):
                return None
            if time.monotonic() >= deadline:
                raise InventoryError("Record scanner synchronization timed out")
            await asyncio.sleep(1.0)

    async def _build(self, request: BuildTransaction, account: Any) -> PreparedTransaction:
        await self._register_programs([request.program, *request.imports])
        program = await self._aleo.programs.get(request.program)
        bound = program.functions[request.function](*request.inputs)
        # Unmanaged proving paths only; see RecordInventory._build.
        if self.proving == "local":
            transaction = await bound._prove_transaction(  # pyright: ignore[reportPrivateUsage]
                account, priority_fee=self.priority_fee, fee_record=request.fee_record, private_fee=request.private_fee,
            )
            return prepared_from_transaction(transaction)
        payload = await bound.delegate(
            account, broadcast=False, pay_own_fee=self.pay_own_fee, fee_record=request.fee_record, priority_fee=self.priority_fee,
        )
        return prepared_from_payload(payload)

    async def _register_programs(self, program_ids: Sequence[str]) -> None:
        if getattr(self._aleo, "process", None) is None:
            return
        sources: dict[str, str] = {}
        stack = [pid for pid in program_ids if pid != CREDITS_PROGRAM]
        while stack:
            program_id = stack.pop()
            if program_id in sources or program_id == CREDITS_PROGRAM:
                continue
            sources[program_id] = str((await self._aleo.programs.get(program_id)).source)
            stack.extend(parse_program_interface(sources[program_id]).imports)
        self._register_sources(sources)

    async def begin_spend(
        self, program: str, function: str, inputs: Sequence[str], *, account: Any = None,
        private_fee: bool = False, fee_record: str | None = None,
    ) -> SpendCheckpoint:
        """Reserve a call's record inputs before it is proved (awaited). See :meth:`RecordInventory.begin_spend`."""
        engine = self._engine(account)
        checkpoint = self._open_spend(record_input_ids(inputs, engine.address), program, function, asset_program=None, account=account)
        if private_fee or fee_record is not None:
            try:
                credits = await self.get_record_inventory(RecordAsset(CREDITS_PROGRAM), account=account)
                preferred = record_input_ids([fee_record], engine.address)[0] if fee_record else None
                checkpoint.use_fee_candidates(credits.available, preferred)
            except BaseException:
                checkpoint.discard()
                raise
        return checkpoint

    async def get_record_inventory(self, asset: RecordAsset, *, account: Any = None) -> RecordInventorySnapshot:
        """Read the account's spendable records of one asset (awaited). See :meth:`RecordInventory.get_record_inventory`."""
        return await self._run(self._engine(account).get_record_inventory(asset), account)

    async def plan_record_inventory(
        self, asset: RecordAsset, target: InventoryTarget, *, max_transactions: int | None = None, account: Any = None,
    ) -> InventoryPlan:
        """Plan joins and splits without submitting (awaited). See :meth:`RecordInventory.plan_record_inventory`."""
        return await self._run(self._engine(account).plan_record_inventory(asset, target, max_transactions), account)

    async def join_records(
        self, asset: RecordAsset, records: Sequence[InventoryRecord], *, join_program: RecordJoinProgram | Literal[False] | None = None,
        account: Any = None,
    ) -> str:
        """Join records in one transaction and return its id (awaited). See :meth:`RecordInventory.join_records`."""
        return await self._run(self._engine(account).join_records(asset, records, join_program), account)

    async def split_record(self, asset: RecordAsset, record: InventoryRecord, amount: int, *, account: Any = None) -> str:
        """Split one record into two and return the transaction id (awaited). See :meth:`RecordInventory.split_record`."""
        return await self._run(self._engine(account).split_record(asset, record, amount), account)

    async def rebalance_record_inventory(
        self, plan: InventoryPlan, *, timeout: float = 120.0, poll_interval: float = 1.0,
        max_fee_microcredits: int | None = None, cancel: threading.Event | asyncio.Event | None = None, account: Any = None,
    ) -> InventoryResult:
        """Execute a reviewed plan step by step (awaited). See :meth:`RecordInventory.rebalance_record_inventory`."""
        return await self._run(self._engine(account).rebalance_record_inventory(
            plan, timeout=timeout, poll_interval=poll_interval, max_fee_microcredits=max_fee_microcredits,
            cancelled=cancel.is_set if cancel is not None else None,
        ), account)

    async def reconcile_record_inventory(
        self, *, rebroadcast: bool = False, release_reserved: bool = False, account: Any = None,
    ) -> list[RecordSpend]:
        """Settle pending journal entries against the chain (awaited). See :meth:`RecordInventory.reconcile_record_inventory`."""
        return await self._run(self._engine(account).reconcile_record_inventory(
            rebroadcast=rebroadcast, release_reserved=release_reserved), account)

    async def auto_join(
        self, asset: RecordAsset, *, min_amount: int | None = None, max_transactions: int | None = None,
        timeout: float = 120.0, poll_interval: float = 1.0, max_fee_microcredits: int | None = None,
        cancel: threading.Event | asyncio.Event | None = None, account: Any = None,
    ) -> InventoryResult:
        """Join enough records to cover *min_amount*, or all of them (awaited). See :meth:`RecordInventory.auto_join`."""
        return await self._run(self._engine(account).auto_join(
            asset, min_amount=min_amount, max_transactions=max_transactions, timeout=timeout,
            poll_interval=poll_interval, max_fee_microcredits=max_fee_microcredits,
            cancelled=cancel.is_set if cancel is not None else None,
        ), account)

    async def managed_write(
        self, program: str, function: str, inputs: Sequence[str], *, imports: Sequence[str] = (), account: Any = None,
        private_fee: bool = False, fee_record: str | None = None, builder: AsyncBuilder | None = None,
    ) -> str:
        """Run one write through the journal (awaited). See :meth:`RecordInventory.managed_write`.

        *builder* may be a plain function or a coroutine function; an
        awaitable result is awaited.
        """
        return await self._run(self._engine(account).managed_write(
            program, function, inputs, imports=imports, private_fee=private_fee, fee_record=fee_record,
        ), account, builder)


__all__ = [
    "AsyncBuilder", "AsyncRecordInventory", "Builder", "Proving", "RecordInventory", "SpendCheckpoint",
    "managed_inventory", "prepared_from_payload", "prepared_from_result",
]
