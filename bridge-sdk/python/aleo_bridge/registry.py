"""Typed, validated views over the pinned deployment registry (``_registry_data.py``).

Discovery is local: chains, assets and routes come from a reviewed snapshot, never from live lookups.
Assets are addressed as ``"chain/key"`` or ``(chain, key)``; symbols and chain ids compare
case-insensitively. Route ``metadata`` keeps veil's camelCase keys.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Iterable, Mapping

from . import _registry_data
from .errors import AmbiguousRouteError, ConfigurationError, RouteNotFoundError

AssetRef = "str | tuple[str, str] | Asset"
FAMILIES = ("aleo", "evm", "solana")
AVAILABILITIES = ("active", "metadata-required", "disabled")
PRIVACY_KINDS = ("arc20", "arc22")
_SOLANA_REQUIRED_METADATA = (
    "warpProgramAddress", "tokenPda", "nativeCollateralPda", "dispatchAuthorityPda", "mailboxProgramAddress",
    "mailboxOutboxPda", "igpProgramAddress", "igpProgramDataPda", "igpAccount", "splNoopProgramAddress",
    "destinationDomain", "destinationGasAmount", "registryCommit", "solanaReviewedAt", "solanaConfigSource",
)


@dataclass(frozen=True)
class Chain:
    id: str
    display_name: str
    family: str                       # "aleo" | "evm" | "solana"
    environment: str                  # "mainnet" | "testnet"
    native_symbol: str
    protocol_domains: Mapping[str, int] = field(default_factory=dict)   # {"xreserve": 0, "hyperlane": 1}


@dataclass(frozen=True)
class Locator:
    kind: str                         # "aleo-program" | "evm-contract" | "solana-mint" | "native"
    value: str
    token_id: str | None = None


@dataclass(frozen=True)
class Privacy:
    kind: str                         # "arc20" | "arc22"
    program: str


@dataclass(frozen=True)
class Asset:
    id: str                           # "chain/key"
    key: str
    chain_id: str
    symbol: str
    name: str
    decimals: int
    kind: str                         # "native" | "token"
    locator: Locator | None = None
    address_regex: str | None = None
    privacy: Privacy | None = None

    def matches_address(self, value: str) -> bool:
        """Whether *value* matches this asset's chain address format (False when no regex is declared)."""
        return bool(self.address_regex) and isinstance(value, str) and re.fullmatch(self.address_regex, value) is not None


@dataclass(frozen=True)
class Route:
    id: str                           # "protocol:source->destination"
    protocol: str                     # "xreserve" | "hyperlane"
    environment: str
    source_asset_id: str
    destination_asset_id: str
    availability: str                 # "active" | "metadata-required" | "disabled"
    deployment_id: str | None = None
    source: str | None = None
    metadata: Mapping[str, "str | int | bool"] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.availability == "active"

    def meta_str(self, key: str) -> str:
        value = self.metadata.get(key)
        if not isinstance(value, str) or not value:
            raise ConfigurationError(f"Route metadata {key} is missing: {self.id}")
        return value

    def meta_int(self, key: str, default: int | None = None) -> int:
        value = self.metadata.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigurationError(f"Route metadata {key} is invalid: {self.id}")
        return value


def _parse_asset_ref(ref: Any) -> tuple[str, str]:
    if isinstance(ref, tuple) and len(ref) == 2:
        return str(ref[0]).lower(), str(ref[1]).lower()
    if isinstance(ref, str) and ref.count("/") == 1:
        chain, key = ref.split("/")
        return chain.lower(), key.lower()
    raise RouteNotFoundError(f"Asset reference must be 'chain/key' or (chain, key), got {ref!r}")


class Registry:
    """Immutable snapshot with case-insensitive lookups. Build with :func:`build_registry` or directly."""

    def __init__(self, version: str, chains: Iterable[Chain], assets: Iterable[Asset], routes: Iterable[Route]) -> None:
        self.version = version
        self._chains = tuple(chains)
        self._assets = tuple(assets)
        self._routes = tuple(routes)
        self._chain_by_id = {c.id: c for c in self._chains}
        self._asset_by_id = {a.id: a for a in self._assets}
        self._route_by_id = {r.id: r for r in self._routes}

    def __repr__(self) -> str:
        return f"Registry(version={self.version!r}, chains={len(self._chains)}, assets={len(self._assets)}, routes={len(self._routes)})"

    # ── chains ──
    def chains(self, environment: str | None = None) -> list[Chain]:
        return [c for c in self._chains if environment is None or c.environment == environment]

    def chain(self, chain_id: str) -> Chain:
        for c in self._chains:
            if c.id.lower() == str(chain_id).lower():
                return c
        raise RouteNotFoundError(f"Unknown bridge chain: {chain_id}")

    # ── assets ──
    def assets(self, chain: str | None = None, symbol: str | None = None, environment: str | None = None) -> list[Asset]:
        out = []
        for a in self._assets:
            if chain is not None and a.chain_id.lower() != chain.lower():
                continue
            if symbol is not None and a.symbol.lower() != symbol.lower():
                continue
            if environment is not None and self._chain_by_id[a.chain_id].environment != environment:
                continue
            out.append(a)
        return out

    def asset(self, ref: Any) -> Asset:
        if isinstance(ref, Asset):
            return ref
        chain, key = _parse_asset_ref(ref)
        for a in self._assets:
            if a.chain_id.lower() == chain and a.key.lower() == key:
                return a
        raise RouteNotFoundError(f"Unknown bridge asset: {chain}/{key}")

    # ── routes ──
    def _endpoint_matches(self, asset_id: str, selector: str | None) -> bool:
        if selector is None:
            return True
        asset = self._asset_by_id[asset_id]
        if "/" in selector:
            return asset.id.lower() == selector.lower()
        return asset.chain_id.lower() == selector.lower()

    def routes(self, source: str | None = None, destination: str | None = None, protocol: str | None = None,
               symbol: str | None = None, include_unavailable: bool = False, environment: str | None = None) -> list[Route]:
        """Filter routes; *source*/*destination* accept a chain id or an ``"chain/key"`` asset ref.
        Disabled routes are hidden unless *include_unavailable*; metadata-required routes are always listed."""
        out = []
        for r in self._routes:
            if not include_unavailable and r.availability == "disabled":
                continue
            if environment is not None and r.environment != environment:
                continue
            if protocol is not None and r.protocol != protocol:
                continue
            if not self._endpoint_matches(r.source_asset_id, source) or not self._endpoint_matches(r.destination_asset_id, destination):
                continue
            if symbol is not None:
                symbols = {self._asset_by_id[r.source_asset_id].symbol.lower(), self._asset_by_id[r.destination_asset_id].symbol.lower()}
                if symbol.lower() not in symbols:
                    continue
            out.append(r)
        return out

    def route(self, route_id: str) -> Route:
        try:
            return self._route_by_id[route_id]
        except KeyError:
            raise RouteNotFoundError(f"Unknown bridge route: {route_id}") from None

    def find_route(self, source: Any, destination: Any, protocol: str | None = None) -> Route:
        """prepare()'s lookup: the single non-disabled route for an exact asset pair (metadata-required included)."""
        src, dst = self.asset(source), self.asset(destination)
        matches = [r for r in self._routes
                   if r.source_asset_id == src.id and r.destination_asset_id == dst.id
                   and r.availability != "disabled" and (protocol is None or r.protocol == protocol)]
        if not matches:
            raise RouteNotFoundError(
                f"No bridge route from {src.id} to {dst.id}" + (f" over {protocol}" if protocol else "")
                + "; list candidates with registry.routes(source=..., destination=...)")
        if len(matches) > 1:
            raise AmbiguousRouteError(
                f"{len(matches)} routes from {src.id} to {dst.id}: {[r.id for r in matches]} — pass protocol=")
        return matches[0]


def validate_registry(registry: Registry) -> Registry:
    """Port of veil ``validateBridgeRegistry`` (+ the Solana metadata gate). Returns *registry* unchanged."""
    if not registry.version.strip():
        raise ConfigurationError("Bridge registry version must not be empty")
    chain_ids: set[str] = set()
    for chain in registry._chains:
        if chain.id in chain_ids:
            raise ConfigurationError(f"Duplicate bridge chain id: {chain.id}")
        if chain.family not in FAMILIES:
            raise ConfigurationError(f"Bridge chain {chain.id} has unsupported family {chain.family!r}")
        chain_ids.add(chain.id)
    asset_ids: set[str] = set()
    asset_keys: set[str] = set()
    for asset in registry._assets:
        if asset.id in asset_ids:
            raise ConfigurationError(f"Duplicate bridge asset id: {asset.id}")
        if asset.chain_id not in chain_ids:
            raise ConfigurationError(f"Bridge asset {asset.id} references unknown chain {asset.chain_id}")
        if not asset.key.strip():
            raise ConfigurationError(f"Bridge asset {asset.id} has an empty key")
        scoped = f"{asset.chain_id}/{asset.key}"
        if scoped in asset_keys:
            raise ConfigurationError(f"Duplicate bridge asset key: {scoped}")
        if isinstance(asset.decimals, bool) or not isinstance(asset.decimals, int) or asset.decimals < 0:
            raise ConfigurationError(f"Bridge asset {asset.id} has invalid decimals {asset.decimals}")
        if asset.address_regex:
            try:
                re.compile(asset.address_regex)
            except re.error as exc:
                raise ConfigurationError(f"Bridge asset {asset.id} has an invalid address validation regex") from exc
        if asset.privacy is not None:
            if registry._chain_by_id[asset.chain_id].family != "aleo":
                raise ConfigurationError(f"Bridge asset {asset.id} declares a privacy capability on a non-Aleo chain")
            if not asset.privacy.program.strip():
                raise ConfigurationError(f"Bridge asset {asset.id} has an empty privacy program")
            if asset.privacy.kind not in PRIVACY_KINDS:
                raise ConfigurationError(f"Bridge asset {asset.id} has an unsupported privacy capability kind")
        asset_ids.add(asset.id)
        asset_keys.add(scoped)
    route_ids: set[str] = set()
    for route in registry._routes:
        if route.id in route_ids:
            raise ConfigurationError(f"Duplicate bridge route id: {route.id}")
        if route.source_asset_id not in asset_ids:
            raise ConfigurationError(f"Bridge route {route.id} references unknown source asset {route.source_asset_id}")
        if route.destination_asset_id not in asset_ids:
            raise ConfigurationError(f"Bridge route {route.id} references unknown destination asset {route.destination_asset_id}")
        if route.availability not in AVAILABILITIES:
            raise ConfigurationError(f"Bridge route {route.id} has unsupported availability {route.availability!r}")
        source_chain = registry._chain_by_id[registry._asset_by_id[route.source_asset_id].chain_id]
        destination_chain = registry._chain_by_id[registry._asset_by_id[route.destination_asset_id].chain_id]
        if source_chain.environment != route.environment or destination_chain.environment != route.environment:
            raise ConfigurationError(f"Bridge route {route.id} crosses registry environments")
        if route.protocol == "hyperlane" and route.availability == "active" and source_chain.family == "solana":
            for key in _SOLANA_REQUIRED_METADATA:
                value = route.metadata.get(key)
                ok = isinstance(value, int) and not isinstance(value, bool) if key == "destinationDomain" \
                    else isinstance(value, str) and bool(value)
                if not ok:
                    raise ConfigurationError(f"Bridge route {route.id} is active but missing required Solana Hyperlane metadata")
        route_ids.add(route.id)
    return registry


def build_registry(data: ModuleType = _registry_data) -> Registry:
    """Turn the plain-dict literals of a data module into a validated :class:`Registry`."""
    chains = [Chain(c["id"], c["displayName"], c["family"], c["environment"], c["nativeCurrencySymbol"],
                    dict(c.get("protocolDomains", {}))) for c in data.CHAINS]
    assets = []
    for a in data.ASSETS:
        loc = a.get("locator")
        priv = a.get("privacy")
        assets.append(Asset(a["id"], a["key"], a["chainId"], a["symbol"], a["name"], a["decimals"], a["kind"],
                            Locator(loc["kind"], loc["value"], loc.get("tokenId")) if loc else None,
                            a.get("addressValidationRegex"),
                            Privacy(priv["kind"], priv["program"]) if priv else None))
    routes = [Route(r["id"], r["protocol"], r["environment"], r["sourceAssetId"], r["destinationAssetId"],
                    r["availability"], r.get("deploymentId"), r.get("source"), dict(r.get("metadata", {})))
              for r in data.ROUTES]
    return validate_registry(Registry(data.REGISTRY_VERSION, chains, assets, routes))


DEFAULT_REGISTRY: Registry = build_registry()

__all__ = ["Asset", "Chain", "DEFAULT_REGISTRY", "Locator", "Privacy", "Registry", "Route", "build_registry", "validate_registry"]
