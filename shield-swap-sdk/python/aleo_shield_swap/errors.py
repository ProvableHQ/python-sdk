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
    def __init__(self, pool_key: str) -> None:
        super().__init__(f"Pool {pool_key} does not exist on-chain.")
        self.pool_key = pool_key


class PoolNotInitializedError(ShieldSwapError):
    def __init__(self, pool_key: str) -> None:
        super().__init__(f"Pool {pool_key} exists but is not initialized.")
        self.pool_key = pool_key


class InsufficientRecordsError(ShieldSwapError):
    """The record provider found no unspent record covering the amount."""


class InvalidFeeTierError(ShieldSwapError):
    """The fee tier is not registered on-chain (checked before submission)."""


class DexApiError(ShieldSwapError):
    """A DEX REST API request failed; carries the HTTP status and body."""

    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"DEX API error {status}: {body[:200]}")
        self.status = status
        self.body = body


class NotAuthenticatedError(ShieldSwapError):
    """The DEX API rejected the request for lack of a valid JWT (401)."""

    def __init__(self) -> None:
        super().__init__(
            "Not authenticated with the DEX API — run dex.onboard() (or "
            "api.authenticate(address, sign) for manual control)."
        )


class NotRedeemedError(ShieldSwapError):
    """Authenticated, but the account has not redeemed an invite code (403)."""

    def __init__(self) -> None:
        super().__init__(
            "This account has not redeemed an invite code — run "
            "dex.onboard(invite_code=...). Codes are distributed by the team."
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


class AirdropRateLimitedError(ShieldSwapError):
    """``POST /airdrop`` returned 429 — one claim per address per 15 minutes."""

    def __init__(self) -> None:
        super().__init__(
            "Airdrop already claimed for this address in the last 15 minutes "
            "— wait and retry, or proceed if dex.get_balances() shows funds."
        )


class CredentialsMissingError(ShieldSwapError):
    """Delegated-proving/scanner credentials are not configured."""

    def __init__(self) -> None:
        super().__init__(
            "No delegated-proving credentials — set ALEO_E2E_API_KEY and "
            "ALEO_E2E_CONSUMER_ID in the environment (they are persisted to "
            "the profile on next onboard())."
        )
