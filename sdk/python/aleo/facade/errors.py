"""Typed exception hierarchy for the Aleo facade.

All facade methods raise subclasses of :class:`AleoError` rather than raw
``RuntimeError`` or the internal ``AleoNetworkError``/``AleoProvingError``.
The internal errors are re-exported here so callers can catch everything with
a single ``except AleoError``.
"""
from __future__ import annotations

from .._client_common import AleoError, AleoNetworkError, AleoProvingError
from .._scanner_common import (
    RecordScannerRequestError,
    DecryptionNotEnabledError,
    ViewKeyNotStoredError,
    RecordNotFoundError,
    UUIDError,
)


class TransactionNotFound(AleoError):
    """Raised when a requested transaction does not exist on-chain."""

    def __init__(self, tx_id: str) -> None:
        super().__init__(f"Transaction not found: {tx_id}")
        self.tx_id = tx_id


class ProgramNotFound(AleoError):
    """Raised when a requested program does not exist on-chain."""

    def __init__(self, program_id: str) -> None:
        super().__init__(f"Program not found: {program_id}")
        self.program_id = program_id


class ExecutionError(AleoError):
    """Raised when authorize/execute fails (the ContractLogicError analog).

    Carries the underlying snarkvm error message in ``detail``.
    """

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.detail = detail


class TransactionConfirmationTimeout(AleoError):
    """Raised when waiting for transaction confirmation exceeds the timeout."""

    def __init__(self, tx_id: str, timeout: float) -> None:
        super().__init__(
            f"Transaction {tx_id} did not confirm within {timeout}s"
        )
        self.tx_id = tx_id
        self.timeout = timeout


__all__ = [
    "AleoError",
    "TransactionNotFound",
    "ProgramNotFound",
    "ExecutionError",
    "TransactionConfirmationTimeout",
    # Re-exported internal errors — they subclass AleoError (defined in
    # _client_common), so `except AleoError` genuinely catches them.
    "AleoNetworkError",
    "AleoProvingError",
    "RecordScannerRequestError",
    "DecryptionNotEnabledError",
    "ViewKeyNotStoredError",
    "RecordNotFoundError",
    "UUIDError",
]
