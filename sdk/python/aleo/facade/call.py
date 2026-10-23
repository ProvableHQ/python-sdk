"""Aleo — facade bound-call verb ladder (F5).

Extends the F4 :class:`~aleo.facade.programs.PreparedCall` seam into the full
Web3.py-style verb ladder.  ``program.functions.<name>(*args)`` now returns a
:class:`BoundCall` (a :class:`PreparedCall` subclass) that carries the same
coerced arguments and adds the verbs:

* :meth:`~BoundCall.simulate` / :meth:`~BoundCall.call` / :meth:`~BoundCall.authorize`
  — build a local :class:`~aleo.facade.call.AuthorizationResult` (no proof, no
  network).  ``simulate``/``call`` are documented aliases of ``authorize``.
* :meth:`~BoundCall.build_transaction` (alias :meth:`~BoundCall.prove`) — the
  full authorize → execute → prepare → prove → fee → assemble ladder, returning
  a :class:`TransactionResult`.
* :meth:`~BoundCall.transact` — build + broadcast, returning the tx id string.
* :meth:`~BoundCall.delegate` — the **flagship** delegated-proving path: build
  the main authorization only, hand it to a Delegated Proving Service (DPS) that
  does the proving.  **By default the prover's fee master pays** (no fee
  authorization is attached) — this frictionlessness is the whole point.

Result inspection: :class:`AuthorizationResult` and :class:`TransactionResult`
both expose ``.outputs`` / ``.decoded()`` (computed over per-transition
``outputs()``) plus a ``.raw`` escape hatch.  The facade also attaches
``aleo.decode_transition(t_or_id)``.
"""
from __future__ import annotations

from typing import Any

from .._client_common import inclusion_height
from .errors import ExecutionError
from .programs import PreparedCall


# ── Result inspection wrappers ───────────────────────────────────────────────


def decode_transition_object(transition: Any) -> dict[str, Any]:
    """Decode a raw network ``Transition`` to a plain dict.

    Returns ``{program, function, inputs, outputs}`` where ``inputs``/``outputs``
    are the transition's own decoded lists.
    """
    return {
        "program": str(transition.program_id),
        "function": str(transition.function_name),
        "inputs": list(transition.inputs()),
        "outputs": list(transition.outputs()),
    }


class AuthorizationResult:
    """Inspectable wrapper around a raw network ``Authorization``.

    Returned by :meth:`BoundCall.authorize` / :meth:`BoundCall.simulate` /
    :meth:`BoundCall.call`.  Lets a caller SEE the outputs/futures a call will
    produce *before* proving or broadcasting anything.

    Attributes
    ----------
    raw:
        The underlying network :class:`Authorization` (escape hatch).
    """

    __slots__ = ("_raw",)

    def __init__(self, raw: Any) -> None:
        self._raw = raw

    @property
    def raw(self) -> Any:
        """The underlying network ``Authorization`` object (escape hatch)."""
        return self._raw

    @property
    def function_name(self) -> str:
        """The authorization's root function name."""
        return str(self._raw.function_name())

    @property
    def execution_id(self) -> Any:
        """The execution id (``Field``) this authorization binds to."""
        return self._raw.to_execution_id()

    def transitions(self) -> list[Any]:
        """The raw network transitions in this authorization."""
        return list(self._raw.transitions())

    @property
    def outputs(self) -> list[list[dict[str, Any]]]:
        """Per-transition output lists (each transition's ``outputs()``)."""
        return [list(t.outputs()) for t in self._raw.transitions()]

    def decoded(self) -> list[dict[str, Any]]:
        """Per-transition decoded ``{program, function, inputs, outputs}`` dicts."""
        return [decode_transition_object(t) for t in self._raw.transitions()]

    def __repr__(self) -> str:
        return f"AuthorizationResult({self.function_name})"


class TransactionResult:
    """Inspectable wrapper around a built network ``Transaction``.

    Returned by :meth:`BoundCall.build_transaction` / :meth:`BoundCall.prove`.
    Exposes the same ``.outputs`` / ``.decoded()`` surface as
    :class:`AuthorizationResult` so downstream code (and F7's async mirror) can
    treat both uniformly, plus ``.id`` and a ``.raw`` hatch.

    Attributes
    ----------
    raw:
        The underlying network :class:`Transaction` (escape hatch).
    checkpoint:
        On a client with record management enabled, the journal entry that
        reserved this transaction's record inputs and saved the proved
        transaction. ``aleo.network.submit_transaction(result)`` records the
        broadcast against it; ``None`` on an unmanaged client.
    """

    __slots__ = ("_raw", "checkpoint")

    def __init__(self, raw: Any, checkpoint: Any = None) -> None:
        self._raw = raw
        self.checkpoint = checkpoint

    @property
    def raw(self) -> Any:
        """The underlying network ``Transaction`` object (escape hatch)."""
        return self._raw

    def discard(self) -> None:
        """Release the record reservation of a built transaction that will not be submitted.

        Only meaningful on a managed client. It does nothing once
        ``submit_transaction`` has started, whether the attempt is still in
        flight, the node accepted the transaction, or the attempt ended
        without an answer, because the network may have spent the records;
        use ``reconcile_record_inventory`` to settle such a spend. A result
        that was discarded is refused by ``submit_transaction`` afterwards,
        since its records may already belong to another transaction.
        """
        if self.checkpoint is not None:
            self.checkpoint.discard()

    @property
    def id(self) -> str:
        """The transaction id string."""
        return str(self._raw.id)

    def transitions(self) -> list[Any]:
        """The raw network transitions in this transaction."""
        return list(self._raw.transitions())

    @property
    def outputs(self) -> list[list[dict[str, Any]]]:
        """Per-transition output lists (each transition's ``outputs()``)."""
        return [list(t.outputs()) for t in self._raw.transitions()]

    def decoded(self) -> list[dict[str, Any]]:
        """Per-transition decoded ``{program, function, inputs, outputs}`` dicts."""
        return [decode_transition_object(t) for t in self._raw.transitions()]

    def __repr__(self) -> str:
        return f"TransactionResult({self.id})"


# ── Bound call (the verb ladder) ─────────────────────────────────────────────


class BoundCall(PreparedCall):
    """A :class:`PreparedCall` extended with the F5 verb ladder.

    Produced by ``program.functions.<name>(*args)``.  Subclasses
    :class:`PreparedCall` so it inherits the F4 coercion done in the constructor
    (no duplication) and the ``program_id`` / ``function_name`` / ``inputs`` /
    ``args`` / ``_client`` slots — then layers the authorize/execute/prove/
    broadcast/delegate verbs on top.

    ``self._client`` is the parent :class:`~aleo.facade.client.Aleo` facade.
    """

    __slots__ = ()

    # ── Shared internals ────────────────────────────────────────────────────

    def _net(self) -> Any:
        """Return the network module (``aleo.mainnet`` / ``aleo.testnet``)."""
        network: str = self._client._provider.network
        if network == "testnet":
            import aleo.testnet as _testnet  # type: ignore[attr-defined]
            return _testnet
        import aleo.mainnet as _mainnet
        return _mainnet

    def _resolve_private_key(self, account: Any) -> Any:
        """Resolve *account* (or ``aleo.default_account``) to a ``PrivateKey``.

        Raises :exc:`ValueError` when neither is set — mirrors the account
        module's resolve pattern.
        """
        acct = account
        if acct is None:
            acct = self._client.default_account
        if acct is None:
            raise ValueError(
                "No account provided and aleo.default_account is not set. "
                "Pass an account explicitly or set aleo.default_account first."
            )
        # Accept an Account (has .private_key) or a bare PrivateKey.
        pk = getattr(acct, "private_key", None)
        return pk if pk is not None else acct

    @property
    def _locator(self) -> str:
        """The ``program_id/function_name`` locator used by ``prove_execution``."""
        return f"{self.program_id}/{self.function_name}"

    @property
    def _query_url(self) -> str:
        """The provider base url handed to ``Query.rest`` for trace preparation."""
        return str(self._client._provider.url)

    def _build_authorization(self, account: Any) -> Any:
        """Build the raw network ``Authorization`` for this call (local only)."""
        net = self._net()
        pk = self._resolve_private_key(account)
        program_id = net.ProgramID.from_string(self.program_id)
        function_name = net.Identifier.from_string(self.function_name)
        inputs = [net.Value.parse(a) for a in self.args]
        try:
            return self._client.process.authorize(
                pk, program_id, function_name, inputs
            )
        except ValueError:
            raise
        except Exception as exc:  # snarkvm surfaces failures as RuntimeError
            raise ExecutionError(
                f"Failed to authorize {self._locator}: {exc}", detail=str(exc)
            ) from exc

    # ── Verb: authorize / simulate / call (local only) ──────────────────────

    def authorize(self, account: Any = None) -> AuthorizationResult:
        """Build the :class:`Authorization` for this call (local, no proof/send).

        The signer defaults to ``aleo.default_account`` when *account* is
        ``None``; a clear :exc:`ValueError` is raised if neither is set.

        Returns
        -------
        AuthorizationResult
            An inspectable wrapper — call ``.decoded()`` / ``.outputs`` to see
            the transitions before proving.
        """
        return AuthorizationResult(self._build_authorization(account))

    def simulate(self, account: Any = None) -> AuthorizationResult:
        """Alias of :meth:`authorize` — build the authorization locally.

        Named for the Web3.py ``call``/``simulate`` idiom: it performs no proof
        and touches no network, so it is a safe dry-run of the call's outputs.
        """
        return self.authorize(account)

    def call(self, account: Any = None) -> AuthorizationResult:
        """Alias of :meth:`authorize` (Web3.py ``contract.functions.f().call()``)."""
        return self.authorize(account)

    # ── Fee sourcing ────────────────────────────────────────────────────────

    def _current_height(self) -> int | None:
        """The chain head, for consensus-version-dependent estimates and
        proofs — on any node that follows the real network's schedule.

        A transaction is judged by the rules in force at its inclusion
        height, so fee estimates and the Varuna proof version use the version
        active *now* (resolved through the bindings' per-network activation
        table), not the newest one the network has scheduled — those differ
        between a snarkvm release scheduling a version and its activation.

        A devnode or custom node runs its own activation schedule, so its
        height must not be mapped through the SDK's table; ``None`` for those
        (see :func:`inclusion_height`), and ``None`` when the node cannot
        answer.
        """
        try:
            head = int(self._client.network.get_latest_height())
        except Exception:  # noqa: BLE001 - estimate still possible without it
            return None
        return inclusion_height(self._client._provider.url, head)

    def _authorize_fee(
        self,
        account: Any,
        execution: Any,
        *,
        priority_fee: int,
        fee_record: Any,
        private_fee: bool,
        base_fee: int | None = None,
    ) -> Any:
        """Build a fee :class:`Authorization` bound to *execution*'s id.

        Public by default (``authorize_fee_public``, base fee from
        ``process.execution_cost``).  Private when *fee_record* is supplied or
        *private_fee* is requested — an explicit *fee_record* wins, otherwise the
        record is auto-sourced from ``aleo.record_provider`` (which defaults to
        ``aleo.records``) for at least ``base_fee + priority_fee`` microcredits.

        *base_fee* overrides the auto-computed storage cost.  The SDK's
        ``execution_cost`` can undershoot the fee a node's consensus actually
        requires (version skew); pass an explicit *base_fee* to overpay.
        """
        pk = self._resolve_private_key(account)
        process = self._client.process
        execution_id = execution.execution_id
        if base_fee is None:
            total, _ = process.execution_cost(execution, self._current_height())
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

        record = self._resolve_fee_record(
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

    def _resolve_fee_record(
        self, fee_record: Any, *, min_microcredits: int | None = None
    ) -> Any:
        """Resolve an unspent credits :class:`RecordPlaintext` for a private fee.

        A supplied *fee_record* (string or object) wins; otherwise the client's
        ``record_provider`` (``aleo.record_provider``, which defaults to
        ``aleo.records``) is asked for an unspent ``credits.aleo``/``credits``
        record covering *min_microcredits*.  Raises a clear facade error when a
        private fee is requested but no record can be sourced.
        """
        net = self._net()
        if callable(fee_record):
            # Record management chooses the fee record once the exact fee is
            # known; the callable reserves it and returns its plaintext.
            fee_record = fee_record(int(min_microcredits or 0))
        if fee_record is not None:
            if isinstance(fee_record, str):
                return net.RecordPlaintext.from_string(fee_record)
            return fee_record

        provider = getattr(self._client, "record_provider", None)
        if provider is None:
            raise ExecutionError(
                "A private fee was requested but no record provider is "
                "configured. Pass fee_record=<credits record> explicitly, set "
                "aleo.record_provider, or omit private_fee to pay a public fee."
            )

        # A transaction sources exactly one fee record, so there is nothing to
        # exclude here (exclude_nonces is for multi-record builds, which the
        # verb ladder does not yet produce).
        record = provider.get_unspent_credits_record(
            min_microcredits=min_microcredits,
        )
        if record is None:
            amount = "the fee" if min_microcredits is None else f"{min_microcredits} microcredits"
            raise ExecutionError(
                f"No unspent credits record covering {amount} was found via "
                "the record provider. Fund the account, register it for "
                "scanning (aleo.records.register), or pass fee_record= explicitly."
            )
        return record

    # ── Verb: build_transaction / prove (full ladder, local prove) ──────────

    def build_transaction(
        self,
        account: Any = None,
        *,
        priority_fee: int = 0,
        fee_record: Any = None,
        private_fee: bool = False,
        base_fee: int | None = None,
    ) -> TransactionResult:
        """Run the full ladder and return a proven, assembled transaction.

        authorize → ``process.execute`` → ``trace.prepare(Query.rest(url))`` →
        ``trace.prove_execution(locator)`` → fee (public by default; private via
        *fee_record* / *private_fee*) → ``Transaction.from_execution``.

        The fee authorization is bound to the *execution*'s id (the footgun this
        method encapsulates).

        Parameters
        ----------
        account:
            Signer; defaults to ``aleo.default_account``.
        priority_fee:
            Extra priority fee in microcredits (added atop the base cost).
        fee_record:
            Optional credits :class:`RecordPlaintext` (or string) → private fee.
        private_fee:
            Force a private fee sourced from ``aleo.record_provider``.

        On a client built with record management enabled, the call's record
        inputs are reserved before proving and the proved transaction is
        saved to the journal before this returns; a private fee record is
        chosen and reserved once the exact fee is known. Submit the result
        with ``aleo.network.submit_transaction(result)`` to record the
        broadcast, or call ``result.discard()`` to release the reservation of
        a transaction that was never sent. A record another writer holds
        raises :class:`~aleo.inventory.RecordReservationConflictError` before
        any proof is built.

        Returns
        -------
        TransactionResult
            Inspectable wrapper over the assembled ``Transaction``.
        """
        managed = _managed_inventory(self._client)
        if managed is None:
            return TransactionResult(self._prove_transaction(
                account, priority_fee=priority_fee, fee_record=fee_record, private_fee=private_fee, base_fee=base_fee,
            ))
        checkpoint = managed.begin_spend(
            self.program_id, self.function_name, self.args, account=account,
            private_fee=private_fee or fee_record is not None, fee_record=_fee_record_text(fee_record),
        )
        try:
            tx = self._prove_transaction(
                account, priority_fee=priority_fee, fee_record=checkpoint.fee_record_selector or fee_record,
                private_fee=private_fee, base_fee=base_fee,
            )
            checkpoint.prepared(str(tx.id), _transaction_json(tx))
        except BaseException as exc:
            checkpoint.discard()
            raise _after_acquisition(exc) from exc
        return TransactionResult(tx, checkpoint)

    def _prove_transaction(
        self,
        account: Any = None,
        *,
        priority_fee: int = 0,
        fee_record: Any = None,
        private_fee: bool = False,
        base_fee: int | None = None,
    ) -> Any:
        """Authorize, execute, prove, and assemble the ``Transaction`` locally; no journal, no broadcast."""
        net = self._net()
        process = self._client.process
        auth = self._build_authorization(account)
        # Proofs carry the Varuna version in force at the inclusion height, so
        # prove at the chain head (real-network nodes only; see _current_height).
        height = self._current_height()

        try:
            _, trace = process.execute(auth)
            trace.prepare(net.Query.rest(self._query_url))
            execution = trace.prove_execution(self._locator, block_height=height)
        except Exception as exc:
            raise ExecutionError(
                f"Failed to execute/prove {self._locator}: {exc}",
                detail=str(exc),
            ) from exc

        fee_auth = self._authorize_fee(
            account,
            execution,
            priority_fee=priority_fee,
            fee_record=fee_record,
            private_fee=private_fee,
            base_fee=base_fee,
        )

        try:
            _, fee_trace = process.execute(fee_auth)
            fee_trace.prepare(net.Query.rest(self._query_url))
            fee = fee_trace.prove_fee(block_height=height)
            tx = net.Transaction.from_execution(execution, fee)
        except Exception as exc:
            raise ExecutionError(
                f"Failed to prove fee / assemble transaction for {self._locator}: "
                f"{exc}",
                detail=str(exc),
            ) from exc

        return tx

    # ``.prove`` is the documented alias of build_transaction.
    prove = build_transaction

    # ── Verb: transact (build + broadcast) ──────────────────────────────────

    def transact(
        self,
        account: Any = None,
        *,
        priority_fee: int = 0,
        fee_record: Any = None,
        private_fee: bool = False,
        base_fee: int | None = None,
    ) -> str:
        """Build the transaction (:meth:`build_transaction`) and broadcast it.

        On a client built with record management enabled this is the two
        managed halves in sequence: the build reserves the record inputs and
        saves the proved transaction, the broadcast records the submission.

        Returns
        -------
        str
            The transaction id returned by the node.
        """
        result = self.build_transaction(
            account,
            priority_fee=priority_fee,
            fee_record=fee_record,
            private_fee=private_fee,
            base_fee=base_fee,
        )
        return self._client.network.submit_transaction(result)

    # ── Verb: delegate (the flagship DPS path) ──────────────────────────────

    def delegate(
        self,
        account: Any = None,
        *,
        broadcast: bool = True,
        pay_own_fee: bool = False,
        fee_record: Any = None,
        priority_fee: int = 0,
    ) -> Any:
        """Delegate proving to a Delegated Proving Service (DPS) — the flagship.

        Builds only the main :class:`Authorization` locally (the DPS does the
        expensive proving), packs it into a ``ProvingRequest`` and submits it via
        ``aleo.network_client.submit_proving_request`` (which raises
        :exc:`~aleo.facade.errors.AleoProvingError` on failure).

        **Fee behaviour.**  By default ``fee_authorization`` is ``None`` — the
        prover's *fee master* pays.  No record, no public fee, no friction; this
        is the whole point of the flagship path.  A self-paid fee authorization
        is attached only when *pay_own_fee* is ``True`` (public fee) or
        *fee_record* is supplied (private fee); either requires proving the
        execution locally first to bind the fee to its execution id.

        Parameters
        ----------
        account:
            Signer; defaults to ``aleo.default_account``.
        broadcast:
            Whether the prover should broadcast the resulting transaction.
        pay_own_fee:
            Pay a public fee yourself instead of using the prover's fee master.
        fee_record:
            Pay a private fee yourself from this credits record.

        Returns
        -------
        Any
            The prover's result payload (the data dict) from the DPS.

        On a client built with record management enabled, a broadcasting
        call asks the prover not to broadcast, reserves the record inputs
        before proving, saves the returned transaction, and submits it from
        this client; the payload then carries the proved ``transaction``.
        """
        managed = _managed_inventory(self._client) if broadcast else None
        if managed is not None:
            payloads: list[Any] = []

            def build(request: Any) -> Any:
                payload = self.delegate(
                    account, broadcast=False, pay_own_fee=pay_own_fee, fee_record=request.fee_record,
                    priority_fee=priority_fee,
                )
                payloads.append(payload)
                return _prepared_from_payload(payload)
            managed.managed_write(
                self.program_id, self.function_name, self.args, account=account,
                private_fee=fee_record is not None, fee_record=_fee_record_text(fee_record), builder=build,
            )
            return payloads[0]
        net = self._net()
        auth = self._build_authorization(account)

        fee_authorization: Any = None
        if pay_own_fee or fee_record is not None:
            # Self-paid fee: bind to the real execution id, which requires
            # proving the execution locally first.
            process = self._client.process
            try:
                _, trace = process.execute(auth)
                trace.prepare(net.Query.rest(self._query_url))
                execution = trace.prove_execution(
                    self._locator, block_height=self._current_height()
                )
            except Exception as exc:
                raise ExecutionError(
                    f"Failed to execute/prove {self._locator} for self-paid "
                    f"delegate fee: {exc}",
                    detail=str(exc),
                ) from exc
            # Delegate never overrides the base fee — only a priority fee is a
            # valid delegate knob; base fee stays the auto-computed storage cost.
            fee_authorization = self._authorize_fee(
                account,
                execution,
                priority_fee=priority_fee,
                fee_record=fee_record,
                private_fee=fee_record is not None,
            )

        request = net.ProvingRequest(auth, fee_authorization, bool(broadcast))
        return self._client.network_client.submit_proving_request(request)


# ── Record management hook helpers (shared with the async client) ────────────


def _managed_inventory(client: Any) -> Any:
    """The client's inventory when record management is enabled, else ``None``."""
    from ..inventory.client import managed_inventory
    return managed_inventory(client)


def _fee_record_text(fee_record: Any) -> str | None:
    """A caller's preferred fee record as plaintext, for the journal's selector."""
    return None if fee_record is None else str(fee_record)


def _transaction_json(transaction: Any) -> dict[str, Any]:
    import json
    return json.loads(str(transaction.to_json()))


def _after_acquisition(error: BaseException) -> BaseException:
    """A conflict raised after this call reserved its inputs must not look retryable."""
    from ..inventory import InventoryError, RecordReservationConflictError
    if isinstance(error, RecordReservationConflictError):
        return InventoryError("Record spending failed after acquiring inputs")
    return error


def _prepared_from_payload(payload: Any) -> Any:
    from ..inventory.client import prepared_from_payload
    return prepared_from_payload(payload)


__all__ = [
    "BoundCall",
    "AuthorizationResult",
    "TransactionResult",
]
