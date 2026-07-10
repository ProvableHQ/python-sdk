"""Shared pure helpers for the Aleo facade (no side-effects, no imports of heavy modules).

This module is intentionally lightweight so it can be imported without pulling
in the compiled extension modules or the network client.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

_MICROCREDITS_PER_CREDIT: int = 1_000_000


@runtime_checkable
class RecordProvider(Protocol):
    """Source of unspent records for the facade (the F5 fee-sourcing seam).

    The default implementation is :class:`~aleo.facade.records.RecordsModule`
    (``aleo.records``), which wraps a delegated :class:`~aleo.record_scanner.RecordScanner`.
    Any object that satisfies this Protocol can be assigned to
    ``aleo.record_provider`` — e.g. a self-hosted scanner wrapper — so callers
    who do not want to share their view key with a hosted scanning service can
    plug in their own source of records.

    Implementations are consumed by :meth:`~aleo.facade.call.BoundCall.build_transaction`
    (and the rest of the verb ladder) to auto-source a credits record for a
    private fee when the caller does not pass ``fee_record`` explicitly.
    """

    def get_unspent(
        self,
        *,
        program: str,
        record: str,
        min_microcredits: int | None = None,
        exclude_nonces: tuple[str, ...] = (),
    ) -> Any | None:
        """Return one unspent record ready to hand to ``process.authorize_fee_private``.

        Returns a *network* ``RecordPlaintext`` (parsed from the scanner's
        ``record_plaintext`` string) for the first unspent ``program``/``record``
        that covers *min_microcredits* (for credits records) and whose ``_nonce``
        is not in *exclude_nonces* — or ``None`` when nothing qualifies.
        """
        ...

    def find(self, **filters: Any) -> list[Any]:
        """Return the list of records matching *filters* (implementation-defined)."""
        ...


def credits_to_microcredits(credits: float | int) -> int:
    """Convert a credits amount (float or int) to integer microcredits.

    Examples
    --------
    >>> credits_to_microcredits(1)
    1000000
    >>> credits_to_microcredits(1.5)
    1500000
    """
    return int(credits * _MICROCREDITS_PER_CREDIT)


def microcredits_to_credits(microcredits: int) -> float:
    """Convert an integer microcredits amount to a credits float.

    Examples
    --------
    >>> microcredits_to_credits(1_000_000)
    1.0
    >>> microcredits_to_credits(1_500_000)
    1.5
    """
    return microcredits / _MICROCREDITS_PER_CREDIT
