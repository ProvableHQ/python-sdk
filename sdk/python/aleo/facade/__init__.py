"""Aleo facade package — Web3.py-style client, provider, and typed exceptions."""
from __future__ import annotations

from .client import Aleo as Aleo
from .provider import HTTPProvider as HTTPProvider
from .account import AccountModule as AccountModule
from .network import NetworkModule as NetworkModule
from .errors import (
    AleoError as AleoError,
    TransactionNotFound as TransactionNotFound,
    ProgramNotFound as ProgramNotFound,
    ExecutionError as ExecutionError,
    TransactionConfirmationTimeout as TransactionConfirmationTimeout,
    # Re-exported internal errors
    AleoNetworkError as AleoNetworkError,
    AleoProvingError as AleoProvingError,
    RecordScannerRequestError as RecordScannerRequestError,
    DecryptionNotEnabledError as DecryptionNotEnabledError,
    ViewKeyNotStoredError as ViewKeyNotStoredError,
    RecordNotFoundError as RecordNotFoundError,
    UUIDError as UUIDError,
)

__all__ = [
    "Aleo",
    "HTTPProvider",
    "AccountModule",
    "NetworkModule",
    "AleoError",
    "TransactionNotFound",
    "ProgramNotFound",
    "ExecutionError",
    "TransactionConfirmationTimeout",
    "AleoNetworkError",
    "AleoProvingError",
    "RecordScannerRequestError",
    "DecryptionNotEnabledError",
    "ViewKeyNotStoredError",
    "RecordNotFoundError",
    "UUIDError",
]
