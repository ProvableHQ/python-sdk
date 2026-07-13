"""DexCall — a facade BoundCall plus a typed result builder.

Preserves the facade's verb ladder on every DEX write: the consumer picks
the proving path (``simulate`` / ``transact`` / ``delegate``) and gets a
typed result back instead of a bare transaction id.
"""
from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

R = TypeVar("R")


class DexCall(Generic[R]):
    """A prepared DEX write.  ``build_result(transaction_id, outputs) -> R``
    turns the submitted transaction into the verb's typed result."""

    def __init__(self, aleo: Any, bound: Any,
                 build_result: Callable[[str, list[Any]], R]) -> None:
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

        Outputs are harvested from the built transaction before broadcast, so
        the result is complete without waiting for confirmation.
        """
        tx = self._bound.build_transaction(account, **fee_kwargs)
        outputs = [o.get("value") for group in tx.outputs() for o in group]
        # Same submit path the facade's BoundCall.transact uses.
        self._aleo.network.submit_transaction(tx.raw)
        return self._build(tx.id, outputs)

    def delegate(self, account: Any = None, **fee_kwargs: Any) -> R:
        """Delegate proving to the DPS (fee master pays by default), wait for
        the transaction, and build the typed result from its transitions."""
        payload = self._bound.delegate(account, **fee_kwargs)
        tx_id = self._extract_tx_id(payload)
        self._aleo.network.wait_for_transaction(tx_id)
        decoded = self._aleo.decode_transition(tx_id)
        outputs = [o.get("value") if isinstance(o, dict) else o
                   for o in decoded.get("outputs", [])]
        return self._build(tx_id, outputs)

    @staticmethod
    def _extract_tx_id(payload: Any) -> str:
        if isinstance(payload, str) and payload.strip():
            return payload.strip()
        if isinstance(payload, dict):
            for key in ("transaction_id", "transactionId", "id", "txid", "tx_id"):
                if payload.get(key):
                    return str(payload[key])
        raise ValueError(f"Cannot find a transaction id in DPS payload: {payload!r}")
