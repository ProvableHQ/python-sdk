from __future__ import annotations

from . import mainnet as mainnet
from . import testnet as testnet
from .encryptor import *
from .network_client import AleoNetworkClient as AleoNetworkClient
from .async_network_client import AsyncAleoNetworkClient as AsyncAleoNetworkClient
from ._client_common import AleoNetworkError as AleoNetworkError
from ._client_common import AleoProvingError as AleoProvingError
from .record_scanner import RecordScanner as RecordScanner
from .async_record_scanner import AsyncRecordScanner as AsyncRecordScanner
from ._scanner_common import (
    RecordScannerRequestError as RecordScannerRequestError,
    DecryptionNotEnabledError as DecryptionNotEnabledError,
    ViewKeyNotStoredError as ViewKeyNotStoredError,
    RecordNotFoundError as RecordNotFoundError,
    UUIDError as UUIDError,
)

# Facade exports (F1)
from .facade import Aleo as Aleo
from .facade import HTTPProvider as HTTPProvider
from .facade.errors import (
    AleoError as AleoError,
    TransactionNotFound as TransactionNotFound,
    ProgramNotFound as ProgramNotFound,
    ExecutionError as ExecutionError,
    TransactionConfirmationTimeout as TransactionConfirmationTimeout,
)


def __getattr__(name: str) -> object:
    if name == "abi":
        import importlib
        mod = importlib.import_module(".abi", __name__)
        # Cache it so subsequent accesses don't call __getattr__ again
        globals()["abi"] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
