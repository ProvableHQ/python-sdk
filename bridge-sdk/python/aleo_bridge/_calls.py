"""AleoCall — a facade BoundCall plus a typed result builder (contract §_calls.py).

The caller picks the proving path; every path harvests the ROOT transition's outputs (the last
transition whose program/function match the call — Aleo orders child transitions first) so the typed
result is complete without waiting for confirmation. ``delegate(broadcast=False)`` is the
checkpointable path: prove on the DPS, hand back the serialized transaction, then ``submit_prepared``
— where a duplicate-transaction response counts as success (invariant 3).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

from .errors import BridgeError, ConfigurationError
from .types import PreparedTx

R = TypeVar("R")
_DUPLICATE_MARKER = "already exists"


def _network_module(aleo: Any) -> Any:
    import aleo as aleo_pkg
    return getattr(aleo_pkg, aleo.network_name)


def extract_tx_id(payload: Any) -> str:
    """Transaction id from a DPS result payload (dict variants or a bare id string)."""
    if isinstance(payload, str) and payload.strip():
        return payload.strip()
    if isinstance(payload, dict):
        tx = payload.get("transaction")
        if isinstance(tx, dict) and tx.get("id"):
            return str(tx["id"])
        for key in ("transaction_id", "transactionId", "id", "txid", "tx_id"):
            if payload.get(key):
                return str(payload[key])
    raise ValueError(f"Cannot find a transaction id in DPS payload: {payload!r}")


def payload_transitions(payload: Any) -> "list[dict[str, Any]] | None":
    """Decoded transitions from a DPS payload that carries the whole transaction; None when it is id-only."""
    if not isinstance(payload, dict):
        return None
    return _transaction_transitions(payload.get("transaction"))


def _transaction_transitions(tx: Any) -> "list[dict[str, Any]] | None":
    if not isinstance(tx, dict):
        return None
    transitions = (tx.get("execution") or {}).get("transitions")
    if not isinstance(transitions, list):
        return None
    return [{"program": str(t.get("program")), "function": str(t.get("function")), "outputs": t.get("outputs", [])}
            for t in transitions]


def output_values(outputs: Any) -> list[str]:
    values: list[str] = []
    for out in outputs:
        if isinstance(out, dict):
            value = out.get("value")
            values.append(value if isinstance(value, str) else str(value))
        else:
            values.append(str(out))
    return values


def root_outputs(decoded: list[dict[str, Any]], program: str, function: str) -> list[str]:
    """Output values of the LAST transition matching *program*/*function* (the root)."""
    for entry in reversed(decoded):
        if str(entry.get("program")) == program and str(entry.get("function")) == function:
            return output_values(entry.get("outputs", []))
    return []


def is_duplicate_submission(exc: BaseException) -> bool:
    """True only for a node's "already exists" rejection (an idempotent rebroadcast).

    Deliberately narrow: a message like "duplicate serial number" or "duplicate output id" is a
    REAL double-spend failure and must propagate, not be swallowed as success.
    """
    return _DUPLICATE_MARKER in str(exc).lower()


class AleoCall(Generic[R]):
    """A prepared Aleo write. Nothing touches the network until a verb runs."""

    def __init__(self, aleo: Any, bound: Any, build_result: Callable[[str, list[str]], R], *,
                 imports: "dict[str, str] | None" = None) -> None:
        self._aleo = aleo
        self._bound = bound
        self._build = build_result
        self._imports = dict(imports or {})
        self._imports_registered = False

    def __repr__(self) -> str:
        # Never print .inputs here: a record plaintext (private_burn arg 0) or a secret nonce
        # (private_mint arg 3) can be an input literal, and repr() output tends to end up in logs.
        return f"AleoCall({self.program_id}/{self.function_name}, inputs={len(self.inputs)} literals)"

    @property
    def program_id(self) -> str:
        return str(self._bound.program_id)

    @property
    def function_name(self) -> str:
        return str(self._bound.function_name)

    @property
    def inputs(self) -> list[str]:
        """The exact Aleo input literals as they will be submitted."""
        return list(self._bound.args)

    # ── import registration (program sources the process must know before authorizing) ──
    def _register_imports(self) -> None:
        if self._imports_registered or not self._imports:
            return
        process = self._aleo.process
        net = _network_module(self._aleo)
        for program_id, source in self._imports.items():
            if process.contains_program(net.ProgramID.from_string(program_id)):
                continue
            process.add_program(net.Program.from_source(source))
        self._imports_registered = True

    # ── verbs ──
    def simulate(self, account: Any = None) -> Any:
        """Local authorization — no proof, no network send; inspect outputs before spending."""
        self._register_imports()
        return self._bound.simulate(account)

    def prove(self, account: Any = None, **fee: Any) -> PreparedTx:
        """Prove locally, do NOT broadcast; returns the serialized transaction for checkpointing."""
        self._register_imports()
        tx = self._bound.build_transaction(account, **fee)
        return PreparedTx(transaction_id=str(tx.id), serialized=str(tx.raw))

    def transact(self, account: Any = None, **fee: Any) -> R:
        """Prove locally, harvest root outputs, broadcast, build the typed result."""
        self._register_imports()
        tx = self._bound.build_transaction(account, **fee)
        outputs = root_outputs(tx.decoded(), self.program_id, self.function_name)
        self._aleo.network.submit_transaction(tx.raw)
        return self._build(str(tx.id), outputs)

    def delegate(self, account: Any = None, *, wait: bool = True, wait_timeout: float = 180.0,
                 broadcast: bool = True, **fee: Any) -> R:
        """Delegate proving to the DPS (fee master pays by default).

        ``broadcast=True``: the prover broadcasts; outputs come from the returned transaction, or after
        waiting and fetching when the payload is id-only. ``broadcast=False``: ``delegate_prepared`` then
        ``submit_prepared`` so the exact bytes exist locally before the network sees them.

        When the payload is id-only, ``wait`` is effectively forced ``True`` regardless of what was
        passed: the transaction body must be fetched after confirmation to harvest outputs, so a wait
        happens either way in that branch.

        If ``wait`` (or the forced wait above) times out, ``aleo.facade.errors.TransactionConfirmationTimeout``
        propagates AFTER the transaction has already been broadcast by the DPS — the transaction id is
        recoverable from the exception's ``tx_id`` attribute (or from re-deriving it) for later polling;
        the transfer itself was not rolled back.
        """
        if not broadcast:
            return self.submit_prepared(self.delegate_prepared(account, **fee), wait=wait, wait_timeout=wait_timeout)
        self._register_imports()
        payload = self._bound.delegate(account, broadcast=True, **fee)
        tx_id = extract_tx_id(payload)
        decoded = payload_transitions(payload)
        if decoded is None:
            self._aleo.network.wait_for_transaction(tx_id, timeout=wait_timeout)
            tx = self._aleo.network.get_transaction_object(tx_id)
            decoded = [{"program": str(t.program_id), "function": str(t.function_name), "outputs": list(t.outputs())}
                       for t in tx.transitions()]
        elif wait:
            self._aleo.network.wait_for_transaction(tx_id, timeout=wait_timeout)
        return self._build(tx_id, root_outputs(decoded, self.program_id, self.function_name))

    def delegate_prepared(self, account: Any = None, **fee: Any) -> PreparedTx:
        """DPS proves with ``broadcast=False``; returns the serialized transaction for checkpointing."""
        self._register_imports()
        payload = self._bound.delegate(account, broadcast=False, **fee)
        tx = payload.get("transaction") if isinstance(payload, dict) else None
        if not isinstance(tx, dict) or not tx.get("id"):
            raise ConfigurationError(
                "The delegated prover did not return the transaction body; cannot checkpoint an unbroadcast "
                "transaction. Use delegate(broadcast=True) or prove() instead.")
        return PreparedTx(transaction_id=str(tx["id"]), serialized=json.dumps(tx))

    def submit_prepared(self, prepared: PreparedTx, *, wait: bool = True, wait_timeout: float = 180.0) -> R:
        """Broadcast a prepared transaction; a duplicate-transaction rejection is success (idempotent rebroadcast).

        If ``wait`` is true and confirmation does not land within ``wait_timeout``,
        ``aleo.facade.errors.TransactionConfirmationTimeout`` propagates AFTER the transaction has already
        been broadcast (the ``submit_transaction`` call above already returned/succeeded) — this is not a
        submission failure. The transaction id is recoverable from ``prepared.transaction_id`` or from the
        exception's own ``tx_id`` attribute, for later polling or a checkpoint. Keeping this raise (rather
        than swallowing it) is consistent with the rest of the facade; the lifecycle layer calls
        ``submit_prepared(wait=False)`` and does its own status polling instead of relying on this wait.
        """
        try:
            self._aleo.network.submit_transaction(prepared.serialized)
        except Exception as exc:  # noqa: BLE001 — the node's error type varies by transport
            if not is_duplicate_submission(exc):
                raise
        if wait:
            self._aleo.network.wait_for_transaction(prepared.transaction_id, timeout=wait_timeout)
        try:
            decoded = _transaction_transitions(json.loads(prepared.serialized)) or []
        except (TypeError, ValueError):
            decoded = []
        return self._build(prepared.transaction_id, root_outputs(decoded, self.program_id, self.function_name))


@dataclass(frozen=True)
class EvmStep:
    """One unsigned EVM transaction the call will broadcast, in order."""

    kind: str          # "approve" | "main"
    to: str
    data: str          # 0x calldata
    value: int = 0     # wei (msg.value)


@dataclass(frozen=True)
class EvmOutcome:
    """What the step runner observed; the module's ``finish`` turns it into the typed result."""

    status: str                        # "SOURCE_APPROVAL_PENDING" | "SOURCE_CONFIRMING" | "CONFIRMED"
    sender: str
    approval_tx_ids: tuple[str, ...]
    source_tx_id: str | None
    receipt: Any | None                # web3 receipt when status == "CONFIRMED"


def _assert_evm_success(receipt: Any, tx_hash: str) -> None:
    if int(receipt["status"]) == 0:
        raise BridgeError(f"EVM transaction reverted: {tx_hash}")


class EvmCall(Generic[R]):
    """A prepared Ethereum write: ``build()`` for unsigned transaction dicts, ``send()`` to broadcast.

    ``steps(sender)`` is evaluated at ``build``/``send`` time so allowances and router
    fees are read at the last responsible moment. ``send`` broadcasts approvals then the
    main call, emits a ``Checkpoint`` after every broadcast (before polling) and after
    confirmation, and returns a pending result when a receipt does not arrive within
    ``timeout_seconds`` — a timeout is not a failure.
    """

    def __init__(self, conn: Any, *, plan: "Plan", registry: "Registry",
                 steps: Callable[[str], list[EvmStep]], finish: Callable[[EvmOutcome], R],
                 store: "CheckpointStore | None" = None) -> None:
        self._conn, self.plan, self._registry = conn, plan, registry
        self._steps, self._finish, self._store = steps, finish, store

    def _sender(self) -> str:
        sender = self._conn.require_address()
        if self.plan.sender:
            Web3 = self._conn.w3.__class__
            if Web3.to_checksum_address(self.plan.sender) != sender:
                raise ConfigurationError(
                    f"Prepared sender {self.plan.sender} does not match connected account {sender}")
        return sender

    def build(self) -> list[dict]:
        """Unsigned transaction dicts in submission order (approvals then main). Reads only.

        With an account configured the plan's sender must be that account (same rule ``send()``
        applies), so a mismatched plan fails here rather than producing calldata nobody can sign.
        """
        sender = self._sender() if self._conn.address is not None else self.plan.sender
        if sender is None:
            raise ConfigurationError("build() needs a sender: configure a signer or set plan.sender")
        nonce = int(self._conn.w3.eth.get_transaction_count(sender, "pending"))
        return [{"from": sender, "to": step.to, "data": step.data, "value": step.value,
                 "chainId": self._conn.chain_id, "nonce": nonce + i}
                for i, step in enumerate(self._steps(sender))]

    def _checkpoint(self, result: R, on_checkpoint: Callable[["Checkpoint"], None] | None, tx_hash: str) -> None:
        """Emit the checkpoint for a just-broadcast *tx_hash* to the caller first, then the store.

        The caller's callback runs before the store because the transaction is already on the wire:
        if persistence fails, the hash must still have reached the one channel that can act on it.
        A store failure is then fatal and names the hash — losing it silently would strand funds.
        """
        from .checkpoint import create_checkpoint

        checkpoint = create_checkpoint(self.plan, result.receipt, self._registry)  # type: ignore[attr-defined]
        if on_checkpoint is not None:
            on_checkpoint(checkpoint)                     # the caller's own callback: errors are theirs
        if self._store is not None:
            try:
                self._store.save(checkpoint)
            except Exception as exc:  # noqa: BLE001 — any store backend failure
                raise BridgeError(
                    f"Transaction {tx_hash} WAS broadcast but its checkpoint {checkpoint.id} could not be saved "
                    f"({exc}); record the transaction hash before retrying — resending would double-spend") from exc

    def send(self, *, wait: bool = True, timeout_seconds: float = 120.0, poll_seconds: float = 1.0,
             on_checkpoint: Callable[["Checkpoint"], None] | None = None) -> R:
        """Broadcast every step in order; checkpoint each hash before polling; pending on timeout.

        ``wait=False`` broadcasts only the first step and returns its pending result; call
        ``bridge.eth.source_status`` (or plan 4's ``resume``) to continue.
        """
        sender = self._sender()
        approvals: list[str] = []
        for step in self._steps(sender):
            tx_hash = self._conn.send_transaction({"from": sender, "to": step.to, "data": step.data, "value": step.value})
            if step.kind == "approve":
                approvals.append(tx_hash)
                pending = self._finish(EvmOutcome("SOURCE_APPROVAL_PENDING", sender, tuple(approvals), None, None))
                self._checkpoint(pending, on_checkpoint, tx_hash)
                if not wait:
                    return pending
                receipt = self._conn.wait_for_receipt(tx_hash, timeout_seconds=timeout_seconds, poll_seconds=poll_seconds)
                if receipt is None:
                    return pending
                _assert_evm_success(receipt, tx_hash)
                continue
            pending = self._finish(EvmOutcome("SOURCE_CONFIRMING", sender, tuple(approvals), tx_hash, None))
            self._checkpoint(pending, on_checkpoint, tx_hash)
            if not wait:
                return pending
            receipt = self._conn.wait_for_receipt(tx_hash, timeout_seconds=timeout_seconds, poll_seconds=poll_seconds)
            if receipt is None:
                return pending
            _assert_evm_success(receipt, tx_hash)
            confirmed = self._finish(EvmOutcome("CONFIRMED", sender, tuple(approvals), tx_hash, receipt))
            self._checkpoint(confirmed, on_checkpoint, tx_hash)
            return confirmed
        raise BridgeError("EvmCall has no main step")


class SolCall(Generic[R]):
    """A prepared Solana-origin call (spec §7).

    ``build()`` re-quotes, compiles the v0 transaction and signs it with the ephemeral
    unique-message keypair only — a preview that spends nothing. ``send()`` rebuilds with a
    fresh quote, unique key and blockhash, checks the wallet balance, adds the fee-payer
    signature through the connection's signer, broadcasts, hands the ``Checkpoint`` built from the
    SOURCE_CONFIRMING receipt to ``on_checkpoint`` and then to the bound store — both before the
    first confirmation poll — and returns the typed result. A polling timeout is not a failure:
    the pending receipt comes back with the signature and blockhash lifetime.
    """

    def __init__(self, module: Any, *, route: Any, recipient: str, amount_atomic: int, plan: Any,
                 build_result: Callable[[Any], R], store: "CheckpointStore | None" = None) -> None:
        self._module = module
        self.route = route
        self.recipient = recipient
        self.amount_atomic = amount_atomic
        self.plan = plan
        self._build_result = build_result
        self._store = store
        self.quote: Any = None
        self._built: Any = None

    def build(self) -> Any:
        """Partially signed ``VersionedTransaction`` (unique-message signer only); sets ``self.quote``."""
        self._built = self._module._build_transaction(route=self.route, recipient=self.recipient,
                                                      amount_atomic=self.amount_atomic, plan=self.plan)
        self.quote = self._built.quote
        return self._built.transaction

    def send(self, *, wait: bool = True, timeout_seconds: float = 120.0, poll_seconds: float = 1.0,
             on_checkpoint: Callable[[Any], None] | None = None) -> R:
        self.build()
        receipt = self._module._submit(self._built, wait=wait, timeout_seconds=timeout_seconds,
                                       poll_seconds=poll_seconds, on_checkpoint=on_checkpoint,
                                       store=self._store)
        return self._build_result(receipt)


__all__ = ["AleoCall", "EvmCall", "EvmOutcome", "EvmStep", "SolCall", "extract_tx_id", "is_duplicate_submission",
           "output_values", "payload_transitions", "root_outputs"]
