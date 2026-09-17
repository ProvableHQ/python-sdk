"""Bridge — the web3.py-style client for moving assets to and from Aleo.

    bridge = Bridge(aleo)                    # Aleo legs: hyperlane / xreserve / privacy / freezelist
    bridge = Bridge.from_env()               # BRIDGE_PRIVATE_KEY, ALEO_ENDPOINT, ALEO_NETWORK, …
    bridge = Bridge.from_profile()           # $ALEO_BRIDGE_HOME or ~/.aleo-bridge

Reads return values; Aleo writes return an AleoCall — nothing touches the network until a verb runs.
Ethereum/Solana connections (plans 2/3), lifecycle verbs and checkpoints (plan 4) plug into the
constructor parameters reserved here.
"""
from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any, Callable

from ._calls import AleoCall
from .errors import ConfigurationError, MissingExtraError
from .freezelist import FreezeList
from .hyperlane import HyperlaneModule
from .privacy import PrivacyModule
from .profile import DEFAULT_ENDPOINT, Profile
from .registry import DEFAULT_REGISTRY, Asset, Chain, Registry, validate_registry
from .types import BridgeStatus, ChainStatus, PrivacyReceipt
from .units import format_decimal_amount, parse_decimal_amount
from .xreserve import XReserveModule

if TYPE_CHECKING:  # pragma: no cover
    from .types import Progress

NETWORKS = ("mainnet", "testnet")
BALANCE_MAPPING = "balances"
_UINT_LITERAL_RE = re.compile(r"^(\d+)u\d+$")


def parse_uint_literal(value: str) -> int:
    """``"2392443u64"`` → 2392443 (any ``uN`` suffix)."""
    match = _UINT_LITERAL_RE.match(value.strip())
    if not match:
        raise ConfigurationError(f"Expected an unsigned Aleo integer literal, got {value!r}")
    return int(match.group(1))


def balance_program(asset: Asset) -> str | None:
    """Program whose ``balances`` mapping holds *asset*'s public balance (None for ALEO credits / no locator)."""
    if asset.locator is None or asset.locator.kind != "aleo-program" or asset.locator.value == "credits.aleo":
        return None
    return asset.privacy.program if asset.privacy is not None else asset.locator.value


def build_aleo(endpoint: str, network: str, private_key: str, *, api_key: str | None = None,
               consumer_id: str | None = None) -> Any:
    """An ``aleo.Aleo`` facade bound to *endpoint*/*network* with *private_key* as the default account (local only)."""
    from aleo import Aleo, HTTPProvider
    kwargs: dict[str, Any] = {"network": network}
    if api_key:
        kwargs["api_key"] = api_key
    if consumer_id:
        kwargs["consumer_id"] = consumer_id
    aleo = Aleo(HTTPProvider(endpoint, **kwargs))
    aleo.default_account = aleo.account.from_private_key(private_key)
    return aleo


def ethereum_from_env() -> Any:
    """``Ethereum(ETHEREUM_RPC_URL, private_key=EVM_PRIVATE_KEY)`` or None; both variables or neither."""
    rpc, key = os.environ.get("ETHEREUM_RPC_URL"), os.environ.get("EVM_PRIVATE_KEY")
    if not rpc and not key:
        return None
    if not (rpc and key):
        raise ConfigurationError("Set EVM_PRIVATE_KEY and ETHEREUM_RPC_URL together (both or neither)")
    try:
        from .eth import Ethereum  # plan 2
    except ImportError as exc:
        raise MissingExtraError("evm", "An Ethereum connection from EVM_PRIVATE_KEY/ETHEREUM_RPC_URL") from exc
    return Ethereum(rpc, private_key=key)


def solana_from_env() -> Any:
    """``Solana(SOLANA_RPC_URL, private_key=SOLANA_PRIVATE_KEY)`` or None (RPC optional)."""
    key = os.environ.get("SOLANA_PRIVATE_KEY")
    if not key:
        return None
    try:
        from .sol import Solana  # plan 3
    except ImportError as exc:
        raise MissingExtraError("solana", "A Solana connection from SOLANA_PRIVATE_KEY") from exc
    return Solana(os.environ.get("SOLANA_RPC_URL"), private_key=key)


def checkpoints_from_env() -> Any:
    """``FileCheckpointStore(BRIDGE_CHECKPOINT_DIR)`` or None."""
    directory = os.environ.get("BRIDGE_CHECKPOINT_DIR")
    if not directory:
        return None
    try:
        from .checkpoint import FileCheckpointStore  # plan 4
    except ImportError as exc:
        raise ConfigurationError("BRIDGE_CHECKPOINT_DIR needs the checkpoint store that arrives with plan 4; unset it for now") from exc
    return FileCheckpointStore(directory)


def _checkpoints_for_profile(profile: Profile) -> Any:
    try:
        from .checkpoint import FileCheckpointStore  # plan 4
    except ImportError:
        return None
    return FileCheckpointStore(profile.checkpoint_dir)


class Bridge:
    """Typed bridge client over the Aleo facade (and, from plans 2/3, Ethereum/Solana connections)."""

    def __init__(self, aleo: Any, *, ethereum: Any = None, solana: Any = None, environment: str | None = None,
                 registry: Registry | None = None, checkpoints: Any = None) -> None:
        network = getattr(aleo, "network_name", None)
        if network not in NETWORKS:
            raise ConfigurationError(f"The aleo facade must report network_name mainnet or testnet, got {network!r}")
        environment = environment if environment is not None else network
        if environment not in NETWORKS:
            raise ConfigurationError(f"environment must be mainnet or testnet, got {environment!r}")
        if environment != network:
            raise ConfigurationError(f"Bridge environment {environment!r} does not match the facade network {network!r}")
        self.aleo = aleo
        self.environment: str = environment
        self.network: str = network
        self.registry: Registry = validate_registry(registry if registry is not None else DEFAULT_REGISTRY)
        if not self.registry.chains(environment=environment):
            raise ConfigurationError(f"Registry {self.registry.version} has no chains for {environment}")
        self.checkpoints = checkpoints
        self.ethereum = ethereum
        self.solana = solana
        self.profile: Profile | None = None
        self._eth_module: Any = None
        self._sol_module: Any = None
        self._programs: dict[str, Any] = {}
        self.hyperlane = HyperlaneModule(self)
        self.xreserve = XReserveModule(self)
        self.freezelist = FreezeList(self)
        self.privacy = PrivacyModule(self)

    def __repr__(self) -> str:
        return f"Bridge(environment={self.environment!r}, registry={self.registry.version!r})"

    # ── side-chain namespaces (plans 2/3 supply the modules) ──
    @property
    def eth(self) -> Any:
        if self.ethereum is None:
            raise ConfigurationError("Ethereum is not configured: Bridge(aleo, ethereum=Ethereum(...)) or set EVM_PRIVATE_KEY + ETHEREUM_RPC_URL")
        if self._eth_module is None:
            try:
                from .eth import Ethereum, EthModule  # plan 2
            except ImportError as exc:
                raise MissingExtraError("evm", "Ethereum-origin bridging") from exc
            connection = self.ethereum if isinstance(self.ethereum, Ethereum) else Ethereum(w3=self.ethereum)
            self._eth_module = EthModule(self, connection)
        return self._eth_module

    @property
    def sol(self) -> Any:
        if self.solana is None:
            raise ConfigurationError("Solana is not configured: Bridge(aleo, solana=Solana(...)) or set SOLANA_PRIVATE_KEY")
        if self._sol_module is None:
            try:
                from .sol import Solana, SolModule  # plan 3
            except ImportError as exc:
                raise MissingExtraError("solana", "Solana-origin bridging") from exc
            connection = self.solana if isinstance(self.solana, Solana) else Solana(client=self.solana)
            self._sol_module = SolModule(self, connection)
        return self._sol_module

    # ── identity / registry helpers ──
    def aleo_chain(self) -> Chain:
        chains = [c for c in self.registry.chains(environment=self.environment) if c.family == "aleo"]
        if len(chains) != 1:
            raise ConfigurationError(f"Registry must define exactly one Aleo chain for {self.environment}")
        return chains[0]

    def aleo_address(self) -> str:
        account = getattr(self.aleo, "default_account", None)
        if not account:
            raise ConfigurationError("aleo.default_account is not set; assign aleo.account.from_private_key(...) first")
        return str(account.address)

    def to_atomic(self, amount: Any, asset: Any) -> int:
        return parse_decimal_amount(amount, self.registry.asset(asset).decimals)

    def from_atomic(self, atomic: int, asset: Any) -> str:
        return format_decimal_amount(atomic, self.registry.asset(asset).decimals)

    # ── facade seams used by every module ──
    def program(self, program_id: str) -> Any:
        """Facade ``Program`` for *program_id*, fetched once per client."""
        if program_id not in self._programs:
            self._programs[program_id] = self.aleo.programs.get(program_id)
        return self._programs[program_id]

    def mapping_value(self, program_id: str, mapping: str, key: str) -> str | None:
        """Mapping value as a string, or None when the key is absent/null."""
        value = self.program(program_id).mapping(mapping).get(key)
        if value is None:
            return None
        text = str(value).strip().strip('"')
        return None if text in ("", "null", "None") else text

    def _import_sources(self, program_id: str) -> dict[str, str]:
        """``{program_id: source}`` for every transitive import (dependencies first) and the root last."""
        ordered: dict[str, str] = {}

        def visit(pid: str) -> None:
            if pid in ordered:
                return
            program = self.program(pid)
            for dep in program.imports:
                visit(str(dep))
            ordered[pid] = str(program.source)

        visit(program_id)
        return ordered

    def _call(self, program_id: str, function: str, inputs: list[str], build_result: Callable[[str, list[str]], Any]) -> AleoCall:
        bound = self.program(program_id).functions[function](*inputs)
        return AleoCall(self.aleo, bound, build_result, imports=self._import_sources(program_id))

    # ── privacy shortcuts ──
    def shield(self, asset: Any, *, amount: Any = None, amount_atomic: int | None = None,
               recipient: str | None = None) -> AleoCall[PrivacyReceipt]:
        return self.privacy.shield(asset, amount=amount, amount_atomic=amount_atomic, recipient=recipient)

    def unshield(self, asset: Any, *, amount: Any = None, amount_atomic: int | None = None, record: str | None = None,
                 merkle_proof: str | None = None, recipient: str | None = None) -> AleoCall[PrivacyReceipt]:
        return self.privacy.unshield(asset, amount=amount, amount_atomic=amount_atomic, record=record,
                                     merkle_proof=merkle_proof, recipient=recipient)

    # ── status ──
    def _public_balance(self, asset: Asset, address: str) -> int:
        if asset.locator is not None and asset.locator.value == "credits.aleo":
            value = self.mapping_value("credits.aleo", "account", address)
        else:
            program = balance_program(asset)
            if program is None:
                return 0
            value = self.mapping_value(program, BALANCE_MAPPING, address)
        return parse_uint_literal(value) if value is not None else 0

    def status(self) -> BridgeStatus:
        """Read-only re-orientation: addresses and public balances of every registry asset per configured chain.
        Plans 2/3 append EVM/Solana ChainStatus entries; plan 4 fills ``pending`` from the checkpoint store."""
        chain = self.aleo_chain()
        account = getattr(self.aleo, "default_account", None)
        address = str(account.address) if account else None
        balances = {asset.id: (self._public_balance(asset, address) if address else 0)
                    for asset in self.registry.assets(chain=chain.id)}
        pending: list["Progress"] = []
        return BridgeStatus(environment=self.environment, registry_version=self.registry.version,
                            chains=[ChainStatus(chain.id, address, address is not None, balances)], pending=pending)

    # ── constructors ──
    @classmethod
    def from_env(cls, **overrides: Any) -> "Bridge":
        """Everything from the environment (spec §3.3); writes nothing to disk. Overrides: ethereum, solana, registry, checkpoints."""
        unexpected = set(overrides) - {"ethereum", "solana", "registry", "checkpoints"}
        if unexpected:
            raise TypeError(f"Bridge.from_env() got unexpected overrides: {sorted(unexpected)}")
        private_key = os.environ.get("BRIDGE_PRIVATE_KEY")
        if not private_key:
            raise ConfigurationError("BRIDGE_PRIVATE_KEY is required (an APrivateKey1... string)")
        aleo = build_aleo(os.environ.get("ALEO_ENDPOINT", DEFAULT_ENDPOINT), os.environ.get("ALEO_NETWORK", "mainnet"),
                          private_key, api_key=os.environ.get("ALEO_API_KEY"), consumer_id=os.environ.get("ALEO_CONSUMER_ID"))
        ethereum = overrides["ethereum"] if "ethereum" in overrides else ethereum_from_env()
        solana = overrides["solana"] if "solana" in overrides else solana_from_env()
        checkpoints = overrides["checkpoints"] if "checkpoints" in overrides else checkpoints_from_env()
        return cls(aleo, ethereum=ethereum, solana=solana, registry=overrides.get("registry"), checkpoints=checkpoints)

    @classmethod
    def from_profile(cls, home: Any = None, *, network: str | None = None, endpoint: str | None = None,
                     ethereum: Any = None, solana: Any = None) -> "Bridge":
        """The client for the local profile (spec §3.4), created on first use. *network*/*endpoint* apply only when
        creating. Side-chain connections come from the arguments or the same env variables as ``from_env``."""
        kwargs = {k: v for k, v in (("network", network), ("endpoint", endpoint)) if v is not None}
        profile = Profile.load_or_create(home, **kwargs)
        aleo = build_aleo(profile.endpoint, profile.network, profile.private_key,
                          api_key=os.environ.get("ALEO_API_KEY"), consumer_id=os.environ.get("ALEO_CONSUMER_ID"))
        bridge = cls(aleo, ethereum=ethereum if ethereum is not None else ethereum_from_env(),
                     solana=solana if solana is not None else solana_from_env(),
                     checkpoints=_checkpoints_for_profile(profile))
        bridge.profile = profile
        return bridge


__all__ = ["BALANCE_MAPPING", "Bridge", "balance_program", "build_aleo", "checkpoints_from_env", "ethereum_from_env",
           "parse_uint_literal", "solana_from_env"]
