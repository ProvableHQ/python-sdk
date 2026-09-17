"""aleo_bridge — move assets between Aleo, Ethereum and Solana (Hyperlane warp routes, Circle xReserve).

Web3.py idioms: bind an ``aleo.Aleo`` facade to :class:`Bridge`; reads return values, Aleo writes
return an :class:`AleoCall` with ``simulate()`` / ``prove()`` / ``transact()`` / ``delegate()``.
Exports grow in Tasks 4–11 of plan 1; keep this list sorted.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .errors import (  # noqa: E402
    AmbiguousRouteError, AttestationError, BridgeError, ChainMismatchError, CheckpointInvalidError,
    ConfigurationError, DeliveryUnknownError, InsufficientBalanceError, InvalidAmountError,
    InvalidRecipientError, MissingExtraError, NotResumableError, PollingTimeoutError,
    RegistryVersionMismatchError, RouteNotFoundError, RouteUnavailableError, UnsupportedRouteError,
)
from .registry import DEFAULT_REGISTRY, Asset, Chain, Locator, Privacy, Registry, Route, validate_registry  # noqa: E402
from .types import (  # noqa: E402
    CALLER_BOUNDARIES, TERMINAL, AleoHyperlaneQuote, AleoXReserveQuote, Attestation, BridgeStatus, BurnReceipt,
    ChainStatus, DepositReceipt, DispatchReceipt, EvmHyperlaneQuote, EvmXReserveQuote, Fee, GasQuote,
    MintReceipt, Plan, PreparedTx, PrivacyReceipt, Progress, Quote, Receipt, SolanaHyperlaneQuote, Status, Step,
    to_progress,
)
from ._calls import AleoCall, EvmCall, SolCall  # noqa: E402
from .checkpoint import Checkpoint, CheckpointStore, FileCheckpointStore, create_checkpoint  # noqa: E402
from .circle import CircleClient  # noqa: E402
from .client import Bridge  # noqa: E402
from .eth import Ethereum, EthModule  # noqa: E402
from .freezelist import EMPTY_MERKLE_PROOF_PAIR, FreezeList  # noqa: E402
from .hyperlane import HyperlaneModule  # noqa: E402
from .privacy import PrivacyModule  # noqa: E402
from .profile import DEFAULT_ENDPOINT, Profile  # noqa: E402
from .sol import DEFAULT_SOLANA_RPC_URL, Solana, SolModule  # noqa: E402
from .xreserve import XReserveModule  # noqa: E402

__all__ = [
    "__version__", "AmbiguousRouteError", "AttestationError", "BridgeError", "ChainMismatchError",
    "CheckpointInvalidError", "ConfigurationError", "DeliveryUnknownError", "InsufficientBalanceError",
    "InvalidAmountError", "InvalidRecipientError", "MissingExtraError", "NotResumableError",
    "PollingTimeoutError", "RegistryVersionMismatchError", "RouteNotFoundError", "RouteUnavailableError",
    "UnsupportedRouteError",
    "Asset", "Chain", "DEFAULT_REGISTRY", "Locator", "Privacy", "Registry", "Route", "validate_registry",
    "CALLER_BOUNDARIES", "TERMINAL", "AleoHyperlaneQuote", "AleoXReserveQuote", "Attestation", "BridgeStatus",
    "BurnReceipt", "ChainStatus", "DepositReceipt", "DispatchReceipt", "EvmHyperlaneQuote", "EvmXReserveQuote",
    "Fee", "GasQuote", "MintReceipt", "Plan", "PreparedTx", "PrivacyReceipt", "Progress", "Quote", "Receipt",
    "SolanaHyperlaneQuote", "Status", "Step", "to_progress",
    "AleoCall", "Bridge", "CircleClient", "DEFAULT_ENDPOINT", "EMPTY_MERKLE_PROOF_PAIR", "FreezeList",
    "HyperlaneModule", "PrivacyModule", "Profile", "XReserveModule",
    "Checkpoint", "CheckpointStore", "FileCheckpointStore", "create_checkpoint",
    "EthModule", "Ethereum", "EvmCall",
    "DEFAULT_SOLANA_RPC_URL", "Solana", "SolCall", "SolModule",
]
