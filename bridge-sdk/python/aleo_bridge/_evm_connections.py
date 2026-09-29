"""Normalize explicitly named EVM providers without reading chains or writing keys."""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from .errors import ConfigurationError
from .eth import Ethereum
from .registry import Registry

RPC_VARIABLES = {"arc": "ARC_RPC_URL", "base": "BASE_RPC_URL", "arbitrum": "ARBITRUM_RPC_URL"}


def coerce(value: Any) -> Ethereum:
    if isinstance(value, Ethereum):
        return value
    if hasattr(value, "eth") and hasattr(value, "provider"):
        return Ethereum(w3=value)
    raise ConfigurationError("EVM connections must be Ethereum(...) or Web3 instances")


def normalize(registry: Registry, environment: str, connections: Mapping[str, Any] | None,
              ethereum: Ethereum | None) -> dict[str, Ethereum]:
    if connections is not None and not isinstance(connections, Mapping):
        raise ConfigurationError("evm= must map registry chain IDs to EVM connections")
    out: dict[str, Ethereum] = {}
    for key, value in (connections or {}).items():
        chain = registry.chain(key)
        if chain.family != "evm" or chain.environment != environment:
            raise ConfigurationError(f"EVM connection {key!r} must belong to {environment} and the EVM family")
        conn = coerce(value)
        if chain.id in out and out[chain.id] is not conn:
            raise ConfigurationError(f"Conflicting EVM connections for {chain.id}")
        out[chain.id] = conn
    if ethereum is not None:
        key = "ethereum" if environment == "mainnet" else "sepolia"
        if key in out and out[key] is not ethereum:
            raise ConfigurationError(f"Conflicting ethereum= and evm= connections for {key}")
        out[key] = ethereum
    return out


def from_env(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    out = {str(k).lower(): v for k, v in (overrides or {}).items()}
    key = os.environ.get("EVM_PRIVATE_KEY") or os.environ.get("BRIDGE_EVM_PRIVATE_KEY")
    floor = os.environ.get("BRIDGE_MIN_PRIORITY_FEE_WEI")
    kwargs: dict[str, Any] = {}
    if floor:
        if not floor.isdigit():
            raise ConfigurationError("BRIDGE_MIN_PRIORITY_FEE_WEI must be a whole number of wei")
        kwargs["min_priority_fee_wei"] = int(floor)
    for chain, variable in RPC_VARIABLES.items():
        url = os.environ.get(variable)
        if chain not in out and url:
            out[chain] = Ethereum(url, private_key=key or None, **kwargs)
    return out


def needs_legacy_environment(connections: Mapping[str, Any]) -> bool:
    """Retain legacy pair validation unless another explicitly named EVM is configured."""
    return not connections or bool(os.environ.get("ETHEREUM_RPC_URL") or os.environ.get("BRIDGE_LIVE_ETHEREUM_RPC_URL"))
