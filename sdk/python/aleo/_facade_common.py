"""Shared pure helpers for the Aleo facade (no side-effects, no imports of heavy modules).

This module is intentionally lightweight so it can be imported without pulling
in the compiled extension modules or the network client.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_DOWN
from typing import Any, Protocol, Union, runtime_checkable

_MICROCREDITS_PER_CREDIT: int = 1_000_000

#: What a credits amount may be written as.  ``str`` and ``Decimal`` are exact;
#: ``float`` is not (see :func:`credits_to_microcredits`).
CreditsAmount = Union[str, int, Decimal, float]


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

    def get_unspent_credits_record(
        self,
        *,
        min_microcredits: int | None = None,
        exclude_nonces: tuple[str, ...] = (),
    ) -> Any | None:
        """Return one unspent credits record ready for ``process.authorize_fee_private``.

        Returns a *network* ``RecordPlaintext`` (parsed from the scanner's
        ``record_plaintext`` string) for the first unspent ``credits.aleo``/
        ``credits`` record that covers *min_microcredits* and whose ``_nonce`` is
        not in *exclude_nonces* — or ``None`` when nothing qualifies.
        """
        ...

    def find(self, **filters: Any) -> list[Any]:
        """Return the list of records matching *filters* (implementation-defined)."""
        ...


def _to_decimal(credits: CreditsAmount) -> Decimal:
    """Interpret a credits amount exactly.

    A ``float`` has already lost the value the caller wrote — ``1.005`` is really
    ``1.00499999999999989...`` — so it is routed through ``str()``, which yields
    the shortest representation that round-trips, i.e. the literal as typed.
    ``str``, ``int``, and ``Decimal`` are exact already.
    """
    if isinstance(credits, Decimal):
        return credits
    if isinstance(credits, int):
        return Decimal(credits)
    return Decimal(str(credits))


def credits_to_microcredits(
    credits: CreditsAmount, *, allow_rounding: bool = False
) -> int:
    """Convert a credits amount to integer microcredits, exactly.

    Computed in decimal, never binary floating point: ``1.005`` credits is
    1_005_000 microcredits, where a float multiply would truncate to 1_004_999
    and silently lose value.  Pass *credits* as ``str`` or ``Decimal`` for full
    exactness; a ``float`` is interpreted as the decimal literal it prints as.

    Args:
        credits: Credits amount — ``"1.005"``, ``Decimal("1.005")``, ``1``, or
            ``1.005``.
        allow_rounding: Permit input finer than one microcredit, truncating
            toward zero.  Off by default so lost precision is an error rather
            than a silent underpayment.

    Returns:
        The amount in microcredits.

    Raises:
        ValueError: If *credits* carries sub-microcredit precision and
            *allow_rounding* is False, or is not a usable number.

    Examples
    --------
    >>> credits_to_microcredits(1)
    1000000
    >>> credits_to_microcredits(1.5)
    1500000
    >>> credits_to_microcredits("1.005")
    1005000
    """
    try:
        scaled = _to_decimal(credits) * _MICROCREDITS_PER_CREDIT
    except (ArithmeticError, ValueError, TypeError) as exc:
        raise ValueError(f"{credits!r} is not a valid credits amount") from exc
    if not scaled.is_finite():
        raise ValueError(f"{credits!r} is not a finite credits amount")
    rounded = scaled.to_integral_value(rounding=ROUND_DOWN)
    if scaled != rounded and not allow_rounding:
        raise ValueError(
            f"{credits!r} credits is {scaled} microcredits — finer than the "
            "chain's smallest unit. Round it yourself, or pass "
            "allow_rounding=True to truncate toward zero."
        )
    return int(rounded)


def microcredits_to_credits(microcredits: int) -> Decimal:
    """Convert integer microcredits to credits, exactly.

    Returns a :class:`~decimal.Decimal` rather than a float: microcredits are a
    ``u64``, and past 2**53 a float cannot even hold the integer, so a float
    round-trip is not the identity.  Compares equal to the obvious float
    (``microcredits_to_credits(1_500_000) == 1.5``), but mixing it into float
    arithmetic raises — convert deliberately with ``float(...)`` if you want
    that, accepting the precision loss.

    Args:
        microcredits: Amount in microcredits.

    Returns:
        The amount in credits, exact at any ``u64`` magnitude.

    Examples
    --------
    >>> microcredits_to_credits(1_000_000)
    Decimal('1.000000')
    >>> microcredits_to_credits(1_500_000)
    Decimal('1.500000')
    """
    return Decimal(int(microcredits)).scaleb(-6)
