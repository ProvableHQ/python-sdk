"""shield / unshield for Aleo assets with a privacy capability (port of veil actions/shield.ts, unshield.ts,
internal/aleoPrivacy.ts) plus local record selection through ``aleo.records.find``."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from ._calls import AleoCall
from .errors import ConfigurationError, InsufficientBalanceError, InvalidAmountError, InvalidRecipientError, UnsupportedRouteError
from .registry import Asset
from .types import PrivacyReceipt
from .units import format_decimal_amount, resolve_amount

if TYPE_CHECKING:  # pragma: no cover
    from .client import Bridge

_AMOUNT_RE = re.compile(r"\bamount:\s*(\d+)u128")


def record_amount(plaintext: str) -> int | None:
    """The ``amount`` member of a Token record plaintext, or None when absent."""
    match = _AMOUNT_RE.search(plaintext or "")
    return int(match.group(1)) if match else None


class PrivacyModule:
    """``bridge.privacy`` — convert between public balances and private records."""

    def __init__(self, bridge: "Bridge") -> None:
        self._bridge = bridge

    def _asset(self, asset: Any, operation: str) -> Asset:
        resolved = self._bridge.registry.asset(asset)
        chain = self._bridge.registry.chain(resolved.chain_id)
        if chain.family != "aleo" or resolved.privacy is None:
            raise UnsupportedRouteError(f'Bridge asset "{resolved.id}" does not support {operation}')
        return resolved

    def _amount(self, asset: Asset, amount: Any, amount_atomic: int | None, operation: str) -> tuple[int, str]:
        atomic = resolve_amount(amount=amount, amount_atomic=amount_atomic, decimals=asset.decimals)
        if atomic <= 0:
            raise InvalidAmountError(f"{operation} amount must be greater than zero")
        return atomic, f"{atomic}u128"

    def _recipient(self, asset: Asset, recipient: str | None) -> str:
        value = recipient if recipient is not None else self._bridge.aleo_address()
        if not asset.matches_address(value):
            raise InvalidRecipientError(f"Recipient does not match the Aleo address format: {value}")
        return value

    def select_record(self, program: str, amount_atomic: int, account: Any = None) -> str:
        """Smallest unspent ``program``/``Token`` record covering *amount_atomic* (plaintext string).

        Reads the account's records through the HOSTED RECORD SCANNER, which answers nothing for an
        account that was never registered with it (the Aleo SDK's ``UUIDError``). That is re-raised
        here as a :class:`ConfigurationError` naming the one call that fixes it — registering is a
        privacy decision (it shares the account's VIEW KEY with the scanning service), so the SDK
        never does it for the caller.
        """
        try:
            rows = self._bridge.aleo.records.find(account, program=program, record="Token")
        except Exception as exc:                # matched by NAME: the aleo SDK is an optional import here
            if type(exc).__name__ != "UUIDError":
                raise
            raise ConfigurationError(
                f"The hosted record scanner has no registration for this account, so it returns no "
                f"{program} records ({exc}). Register it explicitly — "
                "`bridge.aleo.records.register(bridge.aleo.default_account)` — which SHARES THIS "
                "ACCOUNT'S VIEW KEY with the scanning service (it can then decrypt every record the "
                "account owns); the SDK never does that for you. Or avoid record selection entirely: "
                "pass record= yourself, or use a public transfer/burn.") from exc
        amounts: list[tuple[int, str]] = []
        for row in rows:
            plaintext = row.get("record_plaintext") if isinstance(row, dict) else getattr(row, "record_plaintext", None)
            value = record_amount(plaintext) if plaintext else None
            if value is not None:
                amounts.append((value, plaintext))
        covering = [entry for entry in amounts if entry[0] >= amount_atomic]
        if not covering:
            largest = max((value for value, _ in amounts), default=0)
            raise InsufficientBalanceError(
                f"No unspent {program} Token record covers {amount_atomic}; largest available is {largest}. "
                "Shield more, join records, or lower the amount.")
        return min(covering)[1]

    def shield(self, asset: Any, *, amount: Any = None, amount_atomic: int | None = None,
               recipient: str | None = None) -> AleoCall[PrivacyReceipt]:
        """Public balance → private record. ARC-22 names the private recipient; ARC-20 always credits the caller."""
        resolved = self._asset(asset, "shielding")
        atomic, literal = self._amount(resolved, amount, amount_atomic, "Shielding")
        privacy = resolved.privacy
        assert privacy is not None
        if privacy.kind == "arc22":
            function, inputs = "transfer_public_to_private", [self._recipient(resolved, recipient), literal]
        else:
            if recipient is not None and recipient != self._bridge.aleo_address():
                raise ConfigurationError("ARC-20 shield always credits the caller; omit recipient=")
            function, inputs = "shield", [literal]
        human = format_decimal_amount(atomic, resolved.decimals)

        def build(tx_id: str, _outputs: list[str]) -> PrivacyReceipt:
            return PrivacyReceipt(tx_id, resolved.id, human, atomic, "shield")

        return self._bridge._call(privacy.program, function, inputs, build)

    def unshield(self, asset: Any, *, amount: Any = None, amount_atomic: int | None = None, record: str | None = None,
                 merkle_proof: str | None = None, recipient: str | None = None) -> AleoCall[PrivacyReceipt]:
        """Private record → public balance. The record defaults to the smallest covering one; ARC-22 also needs the
        freeze-list exclusion proof (computed for the signer; veil's empty pair when the list is empty)."""
        resolved = self._asset(asset, "unshielding")
        atomic, literal = self._amount(resolved, amount, amount_atomic, "Unshielding")
        privacy = resolved.privacy
        assert privacy is not None
        record = record if record is not None else self.select_record(privacy.program, atomic)
        if privacy.kind == "arc22":
            proof = merkle_proof if merkle_proof is not None else \
                self._bridge.freezelist.exclusion_proof(self._bridge.aleo_address(), privacy.program)
            function, inputs = "transfer_private_to_public", [self._recipient(resolved, recipient), literal, record, proof]
        else:
            if recipient is not None and recipient != self._bridge.aleo_address():
                raise ConfigurationError("ARC-20 unshield always credits the caller; omit recipient=")
            if merkle_proof is not None:
                raise ConfigurationError("ARC-20 unshield takes no Merkle proof; omit merkle_proof=")
            function, inputs = "unshield", [record, literal]
        human = format_decimal_amount(atomic, resolved.decimals)

        def build(tx_id: str, _outputs: list[str]) -> PrivacyReceipt:
            return PrivacyReceipt(tx_id, resolved.id, human, atomic, "unshield")

        return self._bridge._call(privacy.program, function, inputs, build)


__all__ = ["PrivacyModule", "record_amount"]
