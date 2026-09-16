"""Error taxonomy — every failure raised by aleo_bridge is a BridgeError whose message states the remedy."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .types import Progress, Status


class BridgeError(Exception):
    """Base class for every error raised by aleo_bridge."""


class ConfigurationError(BridgeError):
    """The client, environment, or registry is configured inconsistently."""


class MissingExtraError(BridgeError):
    """An optional dependency group is required for this feature."""

    def __init__(self, extra: str, feature: str) -> None:
        self.extra = extra
        self.feature = feature
        super().__init__(f"{feature} requires the '{extra}' extra: pip install 'aleo-bridge-sdk[{extra}]'")


class RouteNotFoundError(BridgeError):
    """No registry entry matches the lookup (route, asset, or chain)."""


class AmbiguousRouteError(BridgeError):
    """More than one route matches; pass protocol= to disambiguate."""


class RouteUnavailableError(BridgeError):
    """The route exists but is metadata-required, disabled, or carries placeholder configuration."""


class RegistryVersionMismatchError(BridgeError):
    """A plan or checkpoint was prepared against a different registry version."""


class UnsupportedRouteError(BridgeError):
    """No implementation exists for this (protocol, chain family) combination or asset capability."""


class InvalidAmountError(BridgeError):
    """The amount is malformed, too precise, zero, or below the route minimum."""


class InvalidRecipientError(BridgeError):
    """The recipient does not match the destination chain's address format."""


class InsufficientBalanceError(BridgeError):
    """The account cannot cover the amount (public balance or no covering record)."""


class ChainMismatchError(BridgeError):
    """The connected EVM chain id / Solana genesis does not match the route."""


class NotResumableError(BridgeError):
    """The progress is not in a resumable or completable state."""


class CheckpointInvalidError(BridgeError):
    """A checkpoint fails the version-1 allowlist or does not match its plan."""


class AttestationError(BridgeError):
    """Circle's response is malformed, does not hash to the requested message, or the secret does not open the commitment."""


class DeliveryUnknownError(BridgeError):
    """Delivery could not be determined from the destination chain."""


class PollingTimeoutError(BridgeError):
    """``wait`` gave up; carries the last observed status and progress (timeout is not failure)."""

    def __init__(self, message: str, *, status: "Status | str", progress: "Progress | None" = None) -> None:
        self.status: Any = status
        self.progress = progress
        super().__init__(message)


__all__ = [
    "BridgeError", "ConfigurationError", "MissingExtraError", "RouteNotFoundError", "AmbiguousRouteError",
    "RouteUnavailableError", "RegistryVersionMismatchError", "UnsupportedRouteError", "InvalidAmountError",
    "InvalidRecipientError", "InsufficientBalanceError", "ChainMismatchError", "NotResumableError",
    "CheckpointInvalidError", "AttestationError", "DeliveryUnknownError", "PollingTimeoutError",
]
