"""Shield-swap error taxonomy — every package error is a ShieldSwapError.

The base subclasses the facade's :class:`~aleo.AleoError`, so ``except
AleoError`` catches everything from both layers.  Facade errors
(``AleoProvingError``, ``TransactionConfirmationTimeout``, …) pass through
untranslated.
"""
from __future__ import annotations

from aleo import AleoError


class ShieldSwapError(AleoError):
    """Base class for all shield-swap-sdk errors."""


class SwapOutputNotFinalizedError(ShieldSwapError):
    """``swap_outputs[swap_id]`` is empty — the request has not finalized yet
    (retry shortly), or the output was already claimed (a claim consumes the
    entry, so a second claim of the same handle sees the same absence)."""

    def __init__(self, swap_id: str) -> None:
        super().__init__(
            f"swap_outputs[{swap_id}] is empty — the request transaction has "
            "not finalized yet (retry shortly), or this output was already claimed."
        )
        self.swap_id = swap_id


class PoolNotFoundError(ShieldSwapError):
    """No pool exists at this key.

    Usually a key derived for the wrong token order, fee tier, or network — the
    derivation succeeds regardless, so a bad key only surfaces on the first read.
    """

    def __init__(self, pool_key: str) -> None:
        super().__init__(f"Pool {pool_key} does not exist on-chain.")
        self.pool_key = pool_key


class PoolNotInitializedError(ShieldSwapError):
    """The pool exists but has no slot yet, so it cannot quote or trade.

    Distinct from :class:`PoolNotFoundError`: the pool was created but never
    initialized. Trading against it only works once someone initializes it.
    """

    def __init__(self, pool_key: str) -> None:
        super().__init__(f"Pool {pool_key} exists but is not initialized.")
        self.pool_key = pool_key


class InsufficientRecordsError(ShieldSwapError):
    """The record provider found no unspent record covering the amount."""


class InvalidFeeTierError(ShieldSwapError):
    """The fee tier is not registered on-chain (checked before submission)."""


class DexApiError(ShieldSwapError):
    """A DEX REST API request failed; carries the HTTP status and body.

    ``code`` is the API's machine-readable error code when the body is its
    JSON error envelope (``{"error": …, "code": …, "ref": …}``), else None —
    branch on it rather than on the human-readable message.
    """

    def __init__(self, status: int, body: str,
                 message: "str | None" = None) -> None:
        super().__init__(message or f"DEX API error {status}: {body[:200]}")
        self.status = status
        self.body = body
        self.code: "str | None" = None
        self.ref: "str | None" = None
        try:
            import json
            envelope = json.loads(body)
        except (TypeError, ValueError):
            envelope = None
        if isinstance(envelope, dict):
            code = envelope.get("code")
            self.code = str(code) if code is not None else None
            ref = envelope.get("ref")
            self.ref = str(ref) if ref is not None else None


class NotAuthenticatedError(DexApiError):
    """The DEX API rejected the request for lack of a valid JWT (401).

    Subclasses :class:`DexApiError` so existing ``except DexApiError`` /
    ``.status`` call sites keep working."""

    def __init__(self, body: str = "") -> None:
        super().__init__(
            401, body,
            "Not authenticated with the DEX API — run dex.onboard() (or "
            "api.authenticate(address, sign) for manual control)."
        )


class NotFundedError(ShieldSwapError):
    """The account holds none of the token required for this action."""

    def __init__(self, detail: str = "") -> None:
        super().__init__(
            "This account holds no usable tokens — run dex.onboard() to "
            "request the airdrop, or check dex.get_balances()."
            + (f" ({detail})" if detail else "")
        )


class AirdropPendingError(ShieldSwapError):
    """An airdrop job was accepted but its records have not landed yet."""

    def __init__(self, job_id: "str | None" = None) -> None:
        super().__init__(
            "Airdrop requested but not landed yet — check dex.status() and "
            "retry shortly."
        )
        self.job_id = job_id


class AirdropRateLimitedError(DexApiError):
    """The DEX API returned 429 — for ``POST /airdrop``, one claim per
    address per 15 minutes."""

    def __init__(self, body: str = "") -> None:
        super().__init__(
            429, body,
            "Airdrop already claimed for this address in the last 15 minutes "
            "— wait and retry, or proceed if dex.get_balances() shows funds."
        )


class CredentialsMissingError(ShieldSwapError):
    """Provable API (delegated-proving/scanner) credentials could not be
    provisioned or found."""

    def __init__(self, detail: str = "") -> None:
        super().__init__(
            "Could not obtain Provable API credentials — automatic "
            "provisioning failed and ALEO_E2E_API_KEY/ALEO_E2E_CONSUMER_ID "
            "are not set. Retry dex.onboard(), or set those env vars (they "
            "are persisted to the profile)."
            + (f" ({detail})" if detail else "")
        )
