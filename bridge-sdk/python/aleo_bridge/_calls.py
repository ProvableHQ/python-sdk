"""AleoCall — a facade BoundCall plus a typed result builder (contract §_calls.py).

The caller picks the proving path; every path harvests the ROOT transition's outputs (the last
transition whose program/function match the call — Aleo orders child transitions first) so the typed
result is complete without waiting for confirmation. ``delegate(broadcast=False)`` is the
checkpointable path: prove on the DPS, hand back the serialized transaction, then ``submit_prepared``
— where a duplicate-transaction response counts as success (invariant 3).
"""
from __future__ import annotations

import json
from typing import Any, Callable, Generic, TypeVar

from .errors import ConfigurationError
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
        return f"AleoCall({self.program_id}/{self.function_name}, inputs={self.inputs!r})"

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


class EvmCall:
    """Completed in Task 2."""


__all__ = ["AleoCall", "EvmCall", "extract_tx_id", "is_duplicate_submission", "output_values", "payload_transitions",
           "root_outputs"]
