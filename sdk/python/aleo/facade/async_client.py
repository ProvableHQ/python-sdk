"""Aleo — AsyncAleo async facade (F7).

Mirrors the sync :class:`~aleo.facade.client.Aleo` surface with ``async def``
wherever real network I/O occurs.  Blocking :class:`~aleo.mainnet.Process` ops
(``authorize``, ``execute``, ``prove_execution``, ``prove_fee``,
``trace.prepare``) run inside ``await asyncio.to_thread(...)`` so they do not
block the event loop; all network calls are ``await``-ed directly through
:class:`~aleo.async_network_client.AsyncAleoNetworkClient` and
:class:`~aleo.async_record_scanner.AsyncRecordScanner`.

**Sync surfaces on AsyncAleo:**
- ``account.*`` — all local crypto, reuses :class:`~aleo.facade.account.AccountModule`.
- ``is_valid_address``, ``to_microcredits``, ``from_microcredits`` — local helpers.
- ``network_id``, ``network_name`` — local reads from the provider.
- ``authorize`` / ``simulate`` / ``call`` on :class:`AsyncBoundCall` — building
  the :class:`Authorization` via ``process.authorize`` is a blocking Rust call
  with no async variant; it does NOT touch the network and returns promptly, so
  these are synchronous on the async facade.

**Async surfaces on AsyncAleo (all await-able):**
- ``is_connected`` — lightweight ``get_latest_height`` probe.
- ``get_balance`` — mapping read.
- ``network.*`` — all reads and submit.
- ``programs.get`` — fetches source from the network.
- ``programs.mapping(name).get(key)`` — mapping read.
- ``records.register`` / ``revoke`` / ``status`` / ``find`` / ``find_credits`` / ``get_unspent``
  — all delegate to :class:`~aleo.async_record_scanner.AsyncRecordScanner`.
- ``program.functions.<name>(...).build_transaction`` / ``transact`` / ``delegate``
  — Process ops run in ``asyncio.to_thread``; submit / DPS calls are awaited.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from .._client_common import AleoNetworkError
from .._facade_common import credits_to_microcredits, microcredits_to_credits
from .._scanner_common import OwnedRecord, RecordNotFoundError
from .errors import (
    ExecutionError,
    ProgramNotFound,
    TransactionConfirmationTimeout,
    TransactionNotFound,
)
from .programs import PreparedCall, ProgramFunctions
from .provider import HTTPProvider
from .call import AuthorizationResult, TransactionResult


# ---------------------------------------------------------------------------
# Async context managers mirroring the sync ones
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _async_program_404(program_id: str) -> AsyncGenerator[None, None]:
    """Map a 404 ``AleoNetworkError`` during a program fetch to ``ProgramNotFound``."""
    try:
        yield
    except AleoNetworkError as exc:
        if exc.status == 404:
            raise ProgramNotFound(program_id) from exc
        raise


# ---------------------------------------------------------------------------
# Async programs
# ---------------------------------------------------------------------------


class AsyncMapping:
    """Async handle to a single on-chain mapping."""

    def __init__(self, client: Any, program_id: str, name: str) -> None:
        self._client = client
        self.program_id = program_id
        self.name = name

    async def get(self, key: str | Any) -> str:
        """Return the current value at (*program*, *mapping*, *key*) (awaited)."""
        async with _async_program_404(self.program_id):
            return await self._client.network_client.get_program_mapping_value(
                self.program_id, self.name, str(key)
            )

    def __repr__(self) -> str:
        return f"AsyncMapping({self.program_id}/{self.name})"


class AsyncProgram:
    """Async bound facade program — returned by ``aleo.programs.get(id)`` (awaited).

    ``.functions`` namespace is built synchronously from the fetched source
    (no additional network calls).  ``.mapping(name).get(key)`` is async.
    """

    def __init__(self, client: Any, raw: Any) -> None:
        self._client = client
        self._raw = raw
        self.id: str = str(raw.id)
        inputs_by_fn: dict[str, list[dict[str, Any]]] = {}
        for ident in raw.functions:
            fn_name = str(ident)
            inputs_by_fn[fn_name] = list(raw.get_function_inputs(fn_name))
        # _AsyncProgramFunctions is defined later in this module (after AsyncBoundCall).
        # Python resolves names at call time, so this forward reference is safe.
        self.functions: ProgramFunctions = _AsyncProgramFunctions(  # type: ignore[name-defined]
            self.id, inputs_by_fn, client
        )

    @property
    def raw(self) -> Any:
        """The underlying network ``Program`` object (escape hatch)."""
        return self._raw

    @property
    def source(self) -> str:
        return str(self._raw.source)

    @property
    def imports(self) -> list[str]:
        """The program's import identifiers as strings (local — no network call)."""
        return [str(i) for i in self._raw.imports]

    def mappings(self) -> list[str]:
        """Return the mapping names declared by this program (local — no network call).

        Reads from the program definition via ``program.get_mappings()``.
        """
        return [str(m["name"]) for m in self._raw.get_mappings()]

    def abi(self) -> dict[str, Any]:
        """Generate the ABI for this program (object path, sync — local).

        Delegates to :func:`aleo.abi.generate_abi` with the underlying network
        Program object.  Raises :exc:`ImportError` if ``aleo-abi`` is absent.
        """
        from .. import abi as _abi
        return _abi.generate_abi(self._raw, self._client.network_name)

    def mapping(self, name: str) -> AsyncMapping:
        """Return an :class:`AsyncMapping` handle for *name*."""
        return AsyncMapping(self._client, self.id, name)

    def __repr__(self) -> str:
        return f"AsyncProgram({self.id}, functions={len(self.functions)})"


class AsyncProgramsModule:
    """Async namespaced program operations (``aleo.programs``)."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def _net(self) -> Any:
        network: str = self._client.provider.network
        if network == "testnet":
            import aleo.testnet as _testnet  # type: ignore[attr-defined]
            return _testnet
        import aleo.mainnet as _mainnet
        return _mainnet

    async def get(self, program_id: str, edition: int | None = None) -> AsyncProgram:
        """Fetch *program_id* from the network and return an :class:`AsyncProgram`."""
        async with _async_program_404(program_id):
            source: str = await self._client.network_client.get_program(
                program_id, edition
            )
        net = self._net()
        raw: Any = net.Program.from_source(source)
        return AsyncProgram(self._client, raw)

    async def abi(self, program_id: str, edition: int | None = None) -> dict[str, Any]:
        """Generate the ABI for *program_id* by fetching its source (web path, async).

        Fetches the deployed source via ``aleo.network.get_program`` (awaited),
        then funnels the string to :func:`aleo.abi.generate_abi` (sync/local).

        Parameters
        ----------
        program_id:
            Aleo program identifier.
        edition:
            Optional edition number.

        Raises
        ------
        ProgramNotFound
            If the network has no such program (a 404 from the node).
        ImportError
            If the ``aleo-abi`` package is not installed.
        """
        async with _async_program_404(program_id):
            source: str = await self._client.network_client.get_program(
                program_id, edition
            )
        from .. import abi as _abi
        return _abi.generate_abi(source, self._client.network_name)

    def __repr__(self) -> str:
        return f"AsyncProgramsModule(network={self._client.provider.network!r})"


# ---------------------------------------------------------------------------
# Async records + async record provider
# ---------------------------------------------------------------------------


class AsyncRecordsModule:
    """Async namespaced record operations (``aleo.records``).

    Implements the async :class:`RecordProvider` contract:
    ``get_unspent`` is ``async def`` and ``await``s the scanner.
    """

    def __init__(self, client: Any) -> None:
        self._client = client
        self._scanner: Any = None
        self._account: Any = None

    def _net(self) -> Any:
        network: str = self._client.provider.network
        if network == "testnet":
            import aleo.testnet as _testnet  # type: ignore[attr-defined]
            return _testnet
        import aleo.mainnet as _mainnet
        return _mainnet

    def _build_scanner(self) -> Any:
        from ..async_record_scanner import AsyncRecordScanner

        provider = self._client.provider
        base = str(provider.url).rstrip("/")
        for suffix in ("/mainnet", "/testnet"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        return AsyncRecordScanner(
            base,
            network=provider.network,
            api_key=provider.api_key,
            transport=getattr(provider, "_transport", None),
        )

    @property
    def scanner(self) -> Any:
        """The underlying :class:`~aleo.async_record_scanner.AsyncRecordScanner` (lazy)."""
        if self._scanner is None:
            self._scanner = self._build_scanner()
        return self._scanner

    @scanner.setter
    def scanner(self, scanner: Any) -> None:
        self._scanner = scanner

    async def register(self, account: Any, start: int = 0) -> dict[str, Any]:
        """Register *account* for delegated async scanning from block *start*."""
        scanner = self.scanner
        self._account = account
        scanner.set_account(account)
        scanner.set_decrypt_enabled(True)
        return await scanner.register(account.view_key, start)

    async def revoke(self) -> dict[str, Any]:
        return await self.scanner.revoke()

    async def status(self) -> dict[str, Any]:
        return await self.scanner.status()

    async def find(
        self,
        account: Any = None,
        *,
        program: str | None = None,
        record: str | None = None,
        unspent: bool = True,
        amounts: list[int] | None = None,
        nonces: list[str] | None = None,
        **_extra: Any,
    ) -> list[OwnedRecord]:
        """Find owned records for *account* matching the given filters (async)."""
        acct = account if account is not None else self._account
        scanner = self.scanner
        if acct is not None:
            scanner.set_account(acct)
            scanner.set_decrypt_enabled(True)

        from .._scanner_common import build_owned_filter, compute_uuid

        uuid = (
            str(compute_uuid(acct.view_key, self._client.provider.network))
            if acct is not None
            else None
        )
        owned_filter = build_owned_filter(
            uuid, program=program, record=record, unspent=unspent, nonces=nonces
        )

        if amounts is not None:
            return await scanner.find_credits_records(amounts, owned_filter)
        return await scanner.find_records(owned_filter)

    async def find_credits(
        self, account: Any = None, at_least: int | None = None
    ) -> list[OwnedRecord]:
        """Find unspent credits records for *account* (async)."""
        acct = account if account is not None else self._account
        scanner = self.scanner
        if acct is not None:
            scanner.set_account(acct)
            scanner.set_decrypt_enabled(True)

        from .._scanner_common import build_owned_filter, compute_uuid

        uuid = (
            str(compute_uuid(acct.view_key, self._client.provider.network))
            if acct is not None
            else None
        )

        if at_least is not None:
            try:
                rec = await scanner.find_credits_record(
                    int(at_least), build_owned_filter(uuid)
                )
            except RecordNotFoundError:
                return []
            return [rec]

        owned_filter = build_owned_filter(
            uuid, program="credits.aleo", record="credits"
        )
        return await scanner.find_records(owned_filter)

    async def get_unspent(
        self,
        *,
        program: str,
        record: str,
        min_microcredits: int | None = None,
        exclude_nonces: tuple[str, ...] = (),
    ) -> Any | None:
        """Return one unspent record as a network ``RecordPlaintext`` (or ``None``).

        Async version of the ``RecordProvider`` protocol.
        """
        net = self._net()

        if program == "credits.aleo" and record == "credits":
            if exclude_nonces:
                candidates = await self.find_credits(at_least=None)
            else:
                candidates = await self.find_credits(at_least=min_microcredits)
        else:
            candidates = await self.find(program=program, record=record, unspent=True)

        excluded = set(exclude_nonces)
        for owned in candidates:
            pt_str = owned.get("record_plaintext", "")
            if not pt_str:
                continue
            try:
                plaintext = net.RecordPlaintext.from_string(pt_str)
            except Exception:
                continue
            if excluded and str(plaintext.nonce) in excluded:
                continue
            if (
                min_microcredits is not None
                and program == "credits.aleo"
                and record == "credits"
                and int(plaintext.microcredits) < int(min_microcredits)
            ):
                continue
            return plaintext
        return None

    def __repr__(self) -> str:
        return f"AsyncRecordsModule(network={self._client.provider.network!r})"


# ---------------------------------------------------------------------------
# Async network module
# ---------------------------------------------------------------------------


class AsyncNetworkModule:
    """Async namespaced network operations (``aleo.network``)."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def _nc(self) -> Any:
        return self._client.network_client

    async def get_latest_height(self) -> int:
        return int(await self._nc().get_latest_height())

    async def get_latest_block(self) -> Any:
        return await self._nc().get_latest_block()

    async def get_block(self, height: int) -> Any:
        return await self._nc().get_block(height)

    async def get_block_by_hash(self, block_hash: str) -> Any:
        return await self._nc().get_block_by_hash(block_hash)

    async def get_block_range(self, start: int, end: int) -> list[Any]:
        return await self._nc().get_block_range(start, end)

    async def get_latest_block_hash(self) -> str:
        return str(await self._nc().get_latest_block_hash())

    async def get_state_root(self) -> str:
        return str(await self._nc().get_state_root())

    async def get_program(self, program_id: str, edition: int | None = None) -> str:
        return await self._nc().get_program(program_id, edition)

    async def get_program_mapping_names(self, program_id: str) -> list[str]:
        return await self._nc().get_program_mapping_names(program_id)

    async def get_program_mapping_value(
        self, program_id: str, mapping_name: str, key: str
    ) -> str:
        return await self._nc().get_program_mapping_value(
            program_id, mapping_name, key
        )

    async def get_public_balance(self, address: str) -> int:
        return int(await self._nc().get_public_balance(address))

    async def get_transaction(self, tx_id: str) -> Any:
        try:
            return await self._nc().get_transaction(tx_id)
        except AleoNetworkError as exc:
            if exc.status == 404:
                raise TransactionNotFound(tx_id) from exc
            raise

    async def get_confirmed_transaction(self, tx_id: str) -> Any:
        try:
            return await self._nc().get_confirmed_transaction(tx_id)
        except AleoNetworkError as exc:
            if exc.status == 404:
                raise TransactionNotFound(tx_id) from exc
            raise

    async def get_transaction_object(self, tx_id: str) -> Any:
        """Return a ``Transaction`` object for *tx_id* (network object path, async).

        Raises :exc:`~aleo.facade.errors.TransactionNotFound` on a 404.
        """
        try:
            return await self._nc().get_transaction_object(tx_id)
        except AleoNetworkError as exc:
            if exc.status == 404:
                raise TransactionNotFound(tx_id) from exc
            raise

    async def get_transactions(self, block_height: int) -> list[Any]:
        return await self._nc().get_transactions(block_height)

    async def submit_transaction(self, transaction: Any) -> str:
        return str(await self._nc().submit_transaction(transaction))

    send_raw_transaction = submit_transaction

    async def wait_for_transaction(
        self,
        tx_id: str,
        *,
        timeout: float = 45.0,
        poll_interval: float = 2.0,
    ) -> Any:
        try:
            return await self._nc().wait_for_transaction_confirmation(
                tx_id,
                check_interval=poll_interval,
                timeout=timeout,
            )
        except TimeoutError:
            raise TransactionConfirmationTimeout(tx_id, timeout)

    def __repr__(self) -> str:
        return f"AsyncNetworkModule(network={self._client.provider.network!r})"


# ---------------------------------------------------------------------------
# Async bound call (verb ladder)
# ---------------------------------------------------------------------------


class AsyncBoundCall(PreparedCall):
    """Async verb ladder for a prepared call.

    ``authorize`` / ``simulate`` / ``call`` are SYNCHRONOUS — ``process.authorize``
    is a blocking Rust call that does not touch the network, so there is no
    benefit (and some awkwardness) making them async.

    ``build_transaction``, ``transact``, and ``delegate`` are ``async def`` because
    they either prove (blocking CPU, run via ``asyncio.to_thread``) or submit to
    the network (true async I/O).
    """

    __slots__ = ()

    def _net(self) -> Any:
        network: str = self._client.provider.network
        if network == "testnet":
            import aleo.testnet as _testnet  # type: ignore[attr-defined]
            return _testnet
        import aleo.mainnet as _mainnet
        return _mainnet

    def _resolve_private_key(self, account: Any) -> Any:
        acct = account
        if acct is None:
            acct = self._client.default_account
        if acct is None:
            raise ValueError(
                "No account provided and aleo.default_account is not set. "
                "Pass an account explicitly or set aleo.default_account first."
            )
        pk = getattr(acct, "private_key", None)
        return pk if pk is not None else acct

    @property
    def _locator(self) -> str:
        return f"{self.program_id}/{self.function_name}"

    @property
    def _query_url(self) -> str:
        return str(self._client.provider.url)

    def _build_authorization(self, account: Any) -> Any:
        """Build the raw ``Authorization`` (local, sync — no network, no proof)."""
        net = self._net()
        pk = self._resolve_private_key(account)
        program_id = net.ProgramID.from_string(self.program_id)
        function_name = net.Identifier.from_string(self.function_name)
        inputs = [net.Value.parse(a) for a in self.args]
        try:
            return self._client.process.authorize(pk, program_id, function_name, inputs)
        except ValueError:
            raise
        except Exception as exc:
            raise ExecutionError(
                f"Failed to authorize {self._locator}: {exc}", detail=str(exc)
            ) from exc

    # ── Sync verbs (no network, no proof) ───────────────────────────────────

    def authorize(self, account: Any = None) -> AuthorizationResult:
        """Build the Authorization locally (sync — no proof, no network)."""
        return AuthorizationResult(self._build_authorization(account))

    def simulate(self, account: Any = None) -> AuthorizationResult:
        """Alias of :meth:`authorize` (local, sync)."""
        return self.authorize(account)

    def call(self, account: Any = None) -> AuthorizationResult:
        """Alias of :meth:`authorize` (local, sync)."""
        return self.authorize(account)

    # ── Async verbs ──────────────────────────────────────────────────────────

    async def build_transaction(
        self,
        account: Any = None,
        *,
        priority_fee: int = 0,
        fee_record: Any = None,
        private_fee: bool = False,
        base_fee: int | None = None,
    ) -> TransactionResult:
        """Run the full prove ladder and return an assembled transaction (async).

        Process ops run inside ``asyncio.to_thread``; network calls are awaited.
        """
        net = self._net()
        locator = self._locator
        query_url = self._query_url
        process = self._client.process
        auth = self._build_authorization(account)

        # All blocking Process ops in a thread
        def _prove_execution() -> Any:
            try:
                _, trace = process.execute(auth)
                trace.prepare(net.Query.rest(query_url))
                return trace.prove_execution(locator)
            except Exception as exc:
                raise ExecutionError(
                    f"Failed to execute/prove {locator}: {exc}", detail=str(exc)
                ) from exc

        execution = await asyncio.to_thread(_prove_execution)

        # Fee authorization (sync, local)
        fee_auth = await self._authorize_fee_async(
            account,
            execution,
            priority_fee=priority_fee,
            fee_record=fee_record,
            private_fee=private_fee,
            base_fee=base_fee,
        )

        def _prove_fee_and_assemble() -> TransactionResult:
            try:
                _, fee_trace = process.execute(fee_auth)
                fee_trace.prepare(net.Query.rest(query_url))
                fee = fee_trace.prove_fee()
                tx = net.Transaction.from_execution(execution, fee)
                return TransactionResult(tx)
            except Exception as exc:
                raise ExecutionError(
                    f"Failed to prove fee / assemble transaction for {locator}: {exc}",
                    detail=str(exc),
                ) from exc

        return await asyncio.to_thread(_prove_fee_and_assemble)

    prove = build_transaction  # documented alias

    async def transact(
        self,
        account: Any = None,
        *,
        priority_fee: int = 0,
        fee_record: Any = None,
        private_fee: bool = False,
        base_fee: int | None = None,
    ) -> str:
        """Build and broadcast; return the transaction id (async)."""
        result = await self.build_transaction(
            account,
            priority_fee=priority_fee,
            fee_record=fee_record,
            private_fee=private_fee,
            base_fee=base_fee,
        )
        return await self._client.network.submit_transaction(result.raw)

    async def delegate(
        self,
        account: Any = None,
        *,
        broadcast: bool = True,
        pay_own_fee: bool = False,
        fee_record: Any = None,
        priority_fee: int = 0,
    ) -> Any:
        """Delegate proving to a DPS (async).

        By default ``fee_authorization=None`` — the prover's fee master pays.
        Self-paid fees require proving locally (blocking, in ``asyncio.to_thread``).
        The only self-paid fee knob is *priority_fee*; delegate never overrides
        the base fee.
        """
        net = self._net()
        auth = self._build_authorization(account)

        fee_authorization: Any = None
        if pay_own_fee or fee_record is not None:
            locator = self._locator
            query_url = self._query_url
            process = self._client.process

            def _prove_for_fee() -> Any:
                try:
                    _, trace = process.execute(auth)
                    trace.prepare(net.Query.rest(query_url))
                    return trace.prove_execution(locator)
                except Exception as exc:
                    raise ExecutionError(
                        f"Failed to execute/prove {locator} for self-paid "
                        f"delegate fee: {exc}",
                        detail=str(exc),
                    ) from exc

            execution = await asyncio.to_thread(_prove_for_fee)
            fee_authorization = await self._authorize_fee_async(
                account,
                execution,
                priority_fee=priority_fee,
                fee_record=fee_record,
                private_fee=fee_record is not None,
            )

        request = net.ProvingRequest(auth, fee_authorization, bool(broadcast))
        return await self._client.network_client.submit_proving_request(request)

    # ── Fee helpers (async where record sourcing is async) ───────────────────

    async def _authorize_fee_async(
        self,
        account: Any,
        execution: Any,
        *,
        priority_fee: int,
        fee_record: Any,
        private_fee: bool,
        base_fee: int | None = None,
    ) -> Any:
        pk = self._resolve_private_key(account)
        process = self._client.process
        execution_id = execution.execution_id
        if base_fee is None:
            total, _ = process.execution_cost(execution)
            base_fee = int(total)
        else:
            base_fee = int(base_fee)

        use_private = private_fee or fee_record is not None
        if not use_private:
            try:
                return process.authorize_fee_public(
                    pk, base_fee, int(priority_fee), execution_id
                )
            except Exception as exc:
                raise ExecutionError(
                    f"Failed to authorize public fee for {self._locator}: {exc}",
                    detail=str(exc),
                ) from exc

        record = await self._resolve_fee_record_async(
            fee_record, min_microcredits=base_fee + int(priority_fee)
        )
        try:
            return process.authorize_fee_private(
                pk, record, base_fee, int(priority_fee), execution_id
            )
        except Exception as exc:
            raise ExecutionError(
                f"Failed to authorize private fee for {self._locator}: {exc}",
                detail=str(exc),
            ) from exc

    async def _resolve_fee_record_async(
        self, fee_record: Any, *, min_microcredits: int | None = None
    ) -> Any:
        net = self._net()
        if fee_record is not None:
            if isinstance(fee_record, str):
                return net.RecordPlaintext.from_string(fee_record)
            return fee_record

        provider = getattr(self._client, "record_provider", None)
        if provider is None:
            raise ExecutionError(
                "A private fee was requested but no record provider is configured. "
                "Pass fee_record=<credits record> explicitly, set "
                "aleo.record_provider, or omit private_fee to pay a public fee."
            )

        record = await provider.get_unspent(
            program="credits.aleo",
            record="credits",
            min_microcredits=min_microcredits,
        )
        if record is None:
            amount = (
                "the fee"
                if min_microcredits is None
                else f"{min_microcredits} microcredits"
            )
            raise ExecutionError(
                f"No unspent credits record covering {amount} was found via "
                "the record provider. Fund the account, register it for scanning "
                "(aleo.records.register), or pass fee_record= explicitly."
            )
        return record


# ---------------------------------------------------------------------------
# Async function caller — returns AsyncBoundCall
# ---------------------------------------------------------------------------


class _AsyncFunctionCaller:
    """Like ``_FunctionCaller`` but produces an :class:`AsyncBoundCall`."""

    __slots__ = ("program_id", "function_name", "inputs", "_client")

    def __init__(
        self,
        program_id: str,
        function_name: str,
        inputs: list[dict[str, Any]],
        client: Any = None,
    ) -> None:
        self.program_id = program_id
        self.function_name = function_name
        self.inputs = inputs
        self._client = client

    def __call__(self, *args: Any) -> AsyncBoundCall:
        return AsyncBoundCall(
            self.program_id,
            self.function_name,
            self.inputs,
            args,
            self._client,
        )

    def __repr__(self) -> str:
        from .programs import input_type_name

        parts = [input_type_name(i) for i in self.inputs]
        sig = f"{self.function_name}({', '.join(parts)})"
        return f"<async function {self.program_id}/{sig}>"


class _AsyncProgramFunctions(ProgramFunctions):
    """ProgramFunctions variant that vends :class:`AsyncBoundCall` instances.

    Overrides only ``_make`` — everything else (``__dir__``, ``__iter__``,
    ``__contains__``, ``__getattr__``, ``__getitem__``) is inherited unchanged.
    """

    def _make(self, name: str) -> "_AsyncFunctionCaller":  # type: ignore[override]
        return _AsyncFunctionCaller(
            self._program_id, name, self._inputs_by_fn[name], self._client
        )


# ---------------------------------------------------------------------------
# AsyncAleo — the main async facade client
# ---------------------------------------------------------------------------


class AsyncAleo:
    """Async Web3.py-style client for the Aleo blockchain (F7).

    Mirrors :class:`~aleo.facade.client.Aleo` with ``async def`` where there
    is real network I/O.  Account crypto, coercion, and ABI generation remain
    synchronous (they are local Rust/Python operations).

    Construct with an :class:`~aleo.facade.provider.HTTPProvider`::

        from aleo import AsyncAleo
        aleo = AsyncAleo(AsyncAleo.HTTPProvider("https://api.provable.com/v2"))

    Parameters
    ----------
    provider:
        An :class:`~aleo.facade.provider.HTTPProvider`.
    """

    # Expose HTTPProvider as a nested class attribute for ``AsyncAleo.HTTPProvider(...)``
    HTTPProvider = HTTPProvider

    def __init__(self, provider: object) -> None:
        if not isinstance(provider, HTTPProvider):
            raise TypeError(
                f"provider must be an HTTPProvider, got {type(provider).__name__}"
            )
        self._provider: HTTPProvider = provider
        # Async network client (httpx-based)
        self._async_client: Any = provider._build_async_client()  # pyright: ignore[reportPrivateUsage]
        self._process: Any = None  # lazy
        self._default_account: Any = None

        # Namespaced modules
        from .account import AccountModule
        self.account: AccountModule = AccountModule(self)
        self.network: AsyncNetworkModule = AsyncNetworkModule(self)
        self.programs: AsyncProgramsModule = AsyncProgramsModule(self)
        self.records: AsyncRecordsModule = AsyncRecordsModule(self)

        # The async record provider (default = self.records; get_unspent is async)
        self._record_provider: Any = self.records

    # ── Escape hatches ─────────────────────────────────────────────────────

    @property
    def provider(self) -> HTTPProvider:
        return self._provider

    @property
    def network_client(self) -> Any:
        """The raw :class:`~aleo.async_network_client.AsyncAleoNetworkClient`."""
        return self._async_client

    @property
    def process(self) -> Any:
        """Lazily-loaded :class:`~aleo.mainnet.Process` (blocking; safe to call sync)."""
        if self._process is None:
            net = self._provider.network
            if net == "testnet":
                from ..testnet import Process  # type: ignore[attr-defined]
            else:
                from ..mainnet import Process  # type: ignore[attr-defined]
            self._process = Process.load()
        return self._process

    # ── Default account ────────────────────────────────────────────────────

    @property
    def default_account(self) -> Any:
        return self._default_account

    @default_account.setter
    def default_account(self, account: Any) -> None:
        self._default_account = account

    # ── Record provider ────────────────────────────────────────────────────

    @property
    def record_provider(self) -> Any:
        """The async record provider (``get_unspent`` is ``async def`` here)."""
        return self._record_provider

    @record_provider.setter
    def record_provider(self, provider: Any) -> None:
        self._record_provider = provider

    # ── Network identity (sync — local) ────────────────────────────────────

    @property
    def network_id(self) -> int:
        """Numeric network identifier (0 = mainnet, 1 = testnet)."""
        net = self._provider.network
        if net == "testnet":
            from ..testnet import Network  # type: ignore[attr-defined]
        else:
            from ..mainnet import Network  # type: ignore[attr-defined]
        return int(Network.id())

    @property
    def network_name(self) -> str:
        return self._provider.network

    # ── Connectivity (async) ────────────────────────────────────────────────

    async def is_connected(self) -> bool:
        """Return ``True`` if the node is reachable (awaited)."""
        try:
            await self._async_client.get_latest_height()
            return True
        except Exception:
            return False

    # ── Balance (async) ────────────────────────────────────────────────────

    async def get_balance(self, address: str) -> int:
        """Return the public credits balance for *address* in microcredits (async)."""
        try:
            raw: Any = await self._async_client.get_program_mapping_value(
                "credits.aleo", "account", address
            )
            if raw is None:
                return 0
            val = str(raw).strip().strip('"')
            if not val or val == "null":
                return 0
            if val.endswith("u64"):
                val = val[:-3]
            return int(val)
        except (AleoNetworkError, ValueError):
            return 0
        except Exception:
            return 0

    # ── Unit conversions (sync) ─────────────────────────────────────────────

    def to_microcredits(self, credits: float | int) -> int:
        return credits_to_microcredits(credits)

    def from_microcredits(self, microcredits: int) -> float:
        return microcredits_to_credits(microcredits)

    # ── Address validation (sync) ───────────────────────────────────────────

    def is_valid_address(self, s: str) -> bool:
        try:
            net = self._provider.network
            if net == "testnet":
                from ..testnet import Address  # type: ignore[attr-defined]
            else:
                from ..mainnet import Address  # type: ignore[attr-defined]
            return bool(Address.is_valid(s))  # pyright: ignore[reportAttributeAccessIssue]
        except Exception:
            return False

    # ── ABI generation (local, sync) ────────────────────────────────────────

    def generate_abi(
        self, source_or_program: Any, *, network: str | None = None
    ) -> dict[str, Any]:
        """Generate an ABI dict from a source string or a Program (local — no await).

        Accepts a raw ``.aleo`` source string, an :class:`AsyncProgram`, or a
        raw network ``Program`` object, and funnels to
        :func:`aleo.abi.generate_abi`.

        Parameters
        ----------
        source_or_program:
            An Aleo source string, an :class:`AsyncProgram`, or a raw ``Program``.
        network:
            Network name for ABI generation.  Defaults to this client's
            provider network.

        Raises
        ------
        ImportError
            If the ``aleo-abi`` package is not installed.
        """
        from .. import abi as _abi

        net = network if network is not None else self.network_name
        target: Any = source_or_program
        if isinstance(source_or_program, AsyncProgram):
            target = source_or_program.raw
        return _abi.generate_abi(target, net)

    # ── Transition decoding ─────────────────────────────────────────────────

    async def decode_transition(self, transition_or_id: Any) -> dict[str, Any]:
        """Decode a ``Transition`` (or a transaction id) to a plain dict (async).

        Accepts a raw network :class:`Transition` object directly (sync path),
        or a transaction id string.  For an id, the transaction is fetched via
        ``aleo.network.get_transaction_object`` (awaited) and the transition
        matching *transition_or_id* (or the first one) is decoded.

        Returns ``{program, function, inputs, outputs}``.

        Parameters
        ----------
        transition_or_id:
            A network ``Transition`` object, or a transaction/transition id
            string.

        Raises
        ------
        TransactionNotFound
            If a string id resolves to no transaction (404 from the node).
        ExecutionError
            If no decodable transition is found.
        """
        from .call import decode_transition_object

        # A Transition object exposes program_id/function_name/outputs().
        if not isinstance(transition_or_id, str):
            return decode_transition_object(transition_or_id)

        tid = transition_or_id
        tx: Any = await self.network.get_transaction_object(tid)
        transitions: list[Any] = list(tx.transitions())
        for t in transitions:
            if str(t.id) == tid:
                return decode_transition_object(t)
        if transitions:
            return decode_transition_object(transitions[0])
        raise ExecutionError(f"No decodable transition found for {tid!r}.")

    # ── Repr ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"AsyncAleo(provider={self._provider!r})"


__all__ = ["AsyncAleo"]
