"""DexCall — a facade BoundCall plus a typed result builder.

Preserves the facade's verb ladder on every DEX write: the consumer picks
the proving path (``simulate`` / ``transact`` / ``delegate``) and gets a
typed result back instead of a bare transaction id.

Output harvesting is scoped to the ROOT transition — the one whose program
and function match the call.  Aleo executions order transitions child-first
(token-program transfers precede the shield_swap root, and a fee transition
may follow), so "first output of the transaction" would read a child's
output.
"""
from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

R = TypeVar("R")


def extract_tx_id(payload: Any) -> str:
    """Transaction id from a DPS result payload (dict variants or bare id)."""
    if isinstance(payload, str) and payload.strip():
        return payload.strip()
    if isinstance(payload, dict):
        for key in ("transaction_id", "transactionId", "id", "txid", "tx_id"):
            if payload.get(key):
                return str(payload[key])
    raise ValueError(f"Cannot find a transaction id in DPS payload: {payload!r}")


def output_values(outputs: Any) -> list[str]:
    """Normalize transition outputs (dicts or raw objects) to value strings."""
    vals: list[str] = []
    for o in outputs:
        if isinstance(o, dict):
            v = o.get("value")
            vals.append(v if isinstance(v, str) else str(v))
        else:
            vals.append(str(o))
    return vals


def root_outputs(decoded_transitions: list[dict[str, Any]],
                 program: str, function: str) -> list[str]:
    """Output values of the LAST transition matching *program*/*function*."""
    for entry in reversed(decoded_transitions):
        if (str(entry.get("program")) == program
                and str(entry.get("function")) == function):
            return output_values(entry.get("outputs", []))
    return []


class DexCall(Generic[R]):
    """A prepared DEX write.  ``build_result(transaction_id, root_outputs)
    -> R`` turns the submitted transaction into the verb's typed result."""

    def __init__(self, aleo: Any, bound: Any,
                 build_result: Callable[[str, list[str]], R]) -> None:
        self._aleo = aleo
        self._bound = bound
        self._build = build_result

    def __repr__(self) -> str:
        return f"DexCall({self._bound!r})"

    def simulate(self, account: Any = None) -> Any:
        """Local authorization — no proof, no send; inspect before spending."""
        return self._bound.simulate(account)

    def transact(self, account: Any = None, **fee_kwargs: Any) -> R:
        """Prove locally, broadcast, and build the typed result.

        Root-transition outputs are harvested from the built transaction
        before broadcast, so the result is complete without waiting for
        confirmation.
        """
        tx = self._bound.build_transaction(account, **fee_kwargs)
        outputs = root_outputs(tx.decoded(), self._bound.program_id,
                               self._bound.function_name)
        # Same submit path the facade's BoundCall.transact uses.
        self._aleo.network.submit_transaction(tx.raw)
        return self._build(tx.id, outputs)

    def delegate(self, account: Any = None, **fee_kwargs: Any) -> R:
        """Delegate proving to the DPS (fee master pays by default), wait for
        the transaction, and build the typed result from its root transition."""
        payload = self._bound.delegate(account, **fee_kwargs)
        tx_id = extract_tx_id(payload)
        self._aleo.network.wait_for_transaction(tx_id)
        tx = self._aleo.network.get_transaction_object(tx_id)
        decoded = [
            {"program": str(t.program_id), "function": str(t.function_name),
             "outputs": list(t.outputs())}
            for t in tx.transitions()
        ]
        outputs = root_outputs(decoded, self._bound.program_id,
                               self._bound.function_name)
        return self._build(tx_id, outputs)
