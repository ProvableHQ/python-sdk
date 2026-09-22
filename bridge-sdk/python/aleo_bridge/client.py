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

from . import lifecycle as _lifecycle
from ._calls import AleoCall
from .errors import BridgeError, ConfigurationError
from .eth import Ethereum, EthModule
from .freezelist import FreezeList
from .hyperlane import HyperlaneModule
from .privacy import PrivacyModule
from .profile import DEFAULT_ENDPOINT, Profile
from .registry import DEFAULT_REGISTRY, Asset, Chain, Registry, validate_registry
from .sol import Solana, SolModule
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
    # Verified on chain 2026-09-17: the warp/xreserve programs mint/burn via arc20_<sym>.aleo's
    # mint_public/burn_public, so this IS the ledger transfer_remote (and xreserve burns) spend.
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
    return Ethereum.from_env()


def _coerce_ethereum(value: Any) -> Ethereum | None:
    """Accept an ``Ethereum`` or a bare ``web3.Web3`` (wrapped; signs only via ``eth.default_account``)."""
    if value is None or isinstance(value, Ethereum):
        return value
    if hasattr(value, "eth") and hasattr(value, "provider"):
        return Ethereum(w3=value)
    raise ConfigurationError("ethereum= must be an aleo_bridge.Ethereum connection or a web3.Web3 instance")


def _coerce_solana(value: Any) -> Solana | None:
    """Accept a ``Solana``, an RPC URL string, or a bare solana-py ``Client`` (wrapped read-only).

    Mirrors :func:`_coerce_ethereum`: anything else is a configuration mistake caught here rather
    than as an ``AttributeError`` from the first RPC read.
    """
    if value is None or isinstance(value, Solana):
        return value
    if isinstance(value, str):
        return Solana(rpc_url=value)
    if callable(getattr(value, "get_latest_blockhash", None)) and callable(getattr(value, "get_account_info", None)):
        return Solana(client=value)                   # spec §3: a bare client is a read-only connection
    raise ConfigurationError(
        "solana= must be an aleo_bridge.Solana connection, an RPC URL string, or a solana-py Client "
        "(an object with get_latest_blockhash and get_account_info)")


def solana_from_env() -> Any:
    """``Solana.from_env()``: SOLANA_PRIVATE_KEY/BRIDGE_SOLANA_PRIVATE_KEY (+ SOLANA_RPC_URL/
    BRIDGE_LIVE_SOLANA_RPC_URL) or None; a key alone signs, a URL alone is read-only, neither → None."""
    return Solana.from_env()


def checkpoints_from_env() -> Any:
    """``FileCheckpointStore(BRIDGE_CHECKPOINT_DIR)`` or None."""
    directory = os.environ.get("BRIDGE_CHECKPOINT_DIR")
    if not directory:
        return None
    from .checkpoint import FileCheckpointStore
    return FileCheckpointStore(directory)


def _checkpoints_for_profile(profile: Profile) -> Any:
    from .checkpoint import FileCheckpointStore
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
        self.ethereum: Ethereum | None = _coerce_ethereum(ethereum)
        self.solana: Solana | None = _coerce_solana(solana)
        solana = self.solana
        self._sol: SolModule | None = SolModule(self, solana) if solana is not None else None
        self.profile: Profile | None = None
        self._eth: EthModule | None = None
        self._programs: dict[str, Any] = {}
        self.hyperlane = HyperlaneModule(self)
        self.xreserve = XReserveModule(self)
        self.freezelist = FreezeList(self)
        self.privacy = PrivacyModule(self)

    def __repr__(self) -> str:
        return f"Bridge(environment={self.environment!r}, registry={self.registry.version!r})"

    # ── side-chain namespaces (plan 3 supplies the Solana module) ──
    @property
    def eth(self) -> EthModule:
        """Ethereum-origin actions (Hyperlane transferRemote, xReserve deposit, status, recovery)."""
        if self.ethereum is None:
            raise ConfigurationError(
                "Pass ethereum=Ethereum(...) to Bridge(...) or set EVM_PRIVATE_KEY + ETHEREUM_RPC_URL")
        if self._eth is None:
            self._eth = EthModule(self, self.ethereum)
        return self._eth

    @property
    def sol(self) -> SolModule:
        """Solana-origin module (spec §6). Requires a Solana connection."""
        if self._sol is None:
            raise ConfigurationError(
                "Solana is not configured: pass solana=Solana(rpc_url, private_key=...) or a solana-py Client to Bridge(), "
                "or set SOLANA_PRIVATE_KEY (and optionally SOLANA_RPC_URL) for Bridge.from_env()")
        return self._sol

    # ── identity / registry helpers ──
    def aleo_chain(self) -> Chain:
        chains = [c for c in self.registry.chains(environment=self.environment) if c.family == "aleo"]
        if len(chains) != 1:
            raise ConfigurationError(f"Registry must define exactly one Aleo chain for {self.environment}")
        return chains[0]

    def solana_chain(self) -> Chain | None:
        """The environment's Solana chain, or None — only mainnet has one."""
        chains = [c for c in self.registry.chains(environment=self.environment) if c.family == "solana"]
        return chains[0] if chains else None

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
        """Mapping value as a string, or None when the key is absent/null, or *program_id* does not exist.

        A missing program is reported through the return value here, not an exception, so status()/freezelist
        reads over a program that may not be deployed (yet) degrade to "no data" instead of raising. Callers
        that want the error can still get it from ``program(program_id)`` directly.
        """
        from aleo.facade.errors import ProgramNotFound
        try:
            value = self.program(program_id).mapping(mapping).get(key)
        except ProgramNotFound:
            return None
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

    def _aleo_chain_status(self) -> ChainStatus:
        chain = self.aleo_chain()
        account = getattr(self.aleo, "default_account", None)
        address = str(account.address) if account else None
        balances = {asset.id: (self._public_balance(asset, address) if address else 0)
                    for asset in self.registry.assets(chain=chain.id)}
        return ChainStatus(chain.id, address, address is not None, balances)

    def status(self) -> BridgeStatus:
        """Read-only re-orientation: addresses and public balances of every registry asset per configured chain.

        ``pending`` is every in-flight transfer of the bound checkpoint store, reconstructed offline by
        :meth:`pending` — no chain is read for it, and a record that cannot be interpreted comes back as a
        ``Progress`` with ``next == "failed"`` instead of hiding the others. It is empty when no store is
        bound. Finish any entry with ``recover`` → ``wait`` / ``resume`` / ``complete``, never by starting
        a new transfer.
        """
        chains = [self._aleo_chain_status()]
        if self.ethereum is not None:
            chains.append(self.eth.chain_status())
        solana_chain = self.solana_chain()
        if self.solana is not None and solana_chain is not None:
            # Chain id and asset id come from the registry, not literals: a testnet client (no Solana
            # chain at all) reports no Solana row rather than one naming a chain this environment lacks.
            native = next((a for a in self.registry.assets(chain=solana_chain.id) if a.kind == "native"), None)
            balances = ({native.id: self.sol.balance()}
                        if native is not None and self.solana.address is not None else {})
            chains.append(ChainStatus(chain_id=solana_chain.id, address=self.solana.address,
                                      can_sign=self.solana.can_sign, balances=balances))
        pending: list["Progress"] = self.pending()
        return BridgeStatus(environment=self.environment, registry_version=self.registry.version,
                            chains=chains, pending=pending)

    # ── Tier 1: the lifecycle ──────────────────────────────────────────────

    def quote(self, source, destination, *, amount=None, amount_atomic=None, recipient: str,
              sender: str | None = None, protocol: str | None = None, mint_mode: str = "public",
              secret_nonce: str = "0scalar"):
        """Price a transfer and get the plan that ``execute`` takes. Nothing is signed.

        ``source`` / ``destination`` are ``"chain/key"`` strings or ``(chain, key)``
        tuples (``"ethereum/usdc"``, ``"aleo/usdcx"``); give exactly one of
        ``amount`` (human units, str) or ``amount_atomic`` (int).  ``recipient`` is
        the destination-chain address.  ``mint_mode`` (xReserve into Aleo only):
        ``"public"`` balance, ``"record"`` minted by the relayer, or ``"private"``
        — you finish it yourself with ``complete`` and must keep ``secret_nonce``.
        Returns a kind-specific ``Quote`` (``quote.kind`` in evm-hyperlane /
        solana-hyperlane / aleo-hyperlane / evm-xreserve / aleo-xreserve) with
        ``fees`` and ``amount_out`` in human units and ``quote.plan``.  Show the
        user fees + amount before ``execute``.
        """
        return _lifecycle.quote(self, source=source, destination=destination, amount=amount,
                                amount_atomic=amount_atomic, recipient=recipient, sender=sender,
                                protocol=protocol, mint_mode=mint_mode, secret_nonce=secret_nonce)

    def execute(self, plan, *, on_checkpoint=None, proving: str = "delegate", mode: str | None = None,
                record: str | None = None, merkle_proof: str | None = None,
                gas_payment_microcredits: int | None = None, secret_nonce: str | None = None,
                poll_seconds: float = 1.0, timeout_seconds: float = 120.0):
        """Commit funds on the source chain for ``quote.plan``; returns ``Progress``.

        Runs approval(s) → deposit / dispatch / burn, emitting a ``Checkpoint`` to
        ``on_checkpoint`` (and the bound store) at every boundary — including
        AFTER proving and BEFORE broadcast for Aleo legs, so a crash there is
        resumable without proving twice.  ``proving`` is ``"delegate"`` (DPS) or
        ``"local"``; ``mode`` is ``"caller"|"signer"`` (Aleo Hyperlane) or
        ``"private"|"public"|"public-as-signer"`` (Aleo xReserve burn, default
        private; ``record``/``merkle_proof`` optional — the SDK selects a record
        and computes the exclusion proof).  The Hyperlane hook payment is
        re-quoted right before proving unless ``gas_payment_microcredits`` is
        pinned.  Irreversible once the source step is broadcast: afterwards use
        ``wait`` / ``recover``, never ``execute`` again.
        """
        return _lifecycle.execute(self, plan, on_checkpoint=on_checkpoint, proving=proving, mode=mode,
                                  record=record, merkle_proof=merkle_proof,
                                  gas_payment_microcredits=gas_payment_microcredits, secret_nonce=secret_nonce,
                                  poll_seconds=poll_seconds, timeout_seconds=timeout_seconds)

    def get_status(self, plan, receipt):
        """One status refresh (no polling, no signing); returns the same receipt when nothing changed."""
        return _lifecycle.get_status(self, plan, receipt)

    def wait(self, progress, *, until=None, poll_seconds: float = 15.0, timeout_seconds: float = 1200.0,
             on_update=None, on_error=None, max_consecutive_errors: int = 5):
        """Poll until the transfer finishes or needs you: stops at ``progress.next``
        in resume / complete / done / failed, or at any status in ``until``.

        A ``PollingTimeoutError`` is NOT a failure — the transfer is still in
        flight; call ``wait`` again or ``recover`` later.  ``on_update`` receives
        each changed ``Progress``.  A transient error (flaky RPC/HTTP transport)
        is retried up to ``max_consecutive_errors`` times, calling ``on_error``
        on each tolerated retry; a non-transient error propagates immediately.
        """
        return _lifecycle.wait(self, progress, until=until, poll_seconds=poll_seconds,
                               timeout_seconds=timeout_seconds, on_update=on_update, on_error=on_error,
                               max_consecutive_errors=max_consecutive_errors)

    def recover(self, checkpoint):
        """Rebuild ``Progress`` from a saved checkpoint (``Checkpoint``, dict or JSON) — reads only.

        Re-resolves the route from the live registry and reads chain state once;
        ``progress.next`` then says what to do: ``wait``, ``resume``, ``complete``,
        ``done`` or ``failed``.
        """
        return _lifecycle.recover(self, checkpoint)

    def resume(self, progress, *, on_checkpoint=None, secret_nonce: str | None = None,
               poll_seconds: float = 1.0, timeout_seconds: float = 120.0, proving: str = "delegate"):
        """Finish an interrupted source submission (``progress.next == "resume"``).

        Rebroadcasts the identical proved Aleo transaction (a duplicate response is
        success) or, on EVM, re-scans history and only then authorizes the single
        missing deposit/dispatch.  Never repeats a confirmed step.
        """
        return _lifecycle.resume(self, progress, on_checkpoint=on_checkpoint, secret_nonce=secret_nonce,
                                 poll_seconds=poll_seconds, timeout_seconds=timeout_seconds, proving=proving)

    def complete(self, progress, *, secret_nonce: str, on_checkpoint=None, proving: str = "delegate"):
        """Submit the private USDCx mint (``progress.next == "complete"``).

        Requires the same ``secret_nonce`` given to ``execute``; the SDK never
        stored it.  Submits exactly one ``private_mint`` and returns
        ``DESTINATION_CONFIRMING`` progress to ``wait`` on.
        """
        return _lifecycle.complete(self, progress, secret_nonce=secret_nonce, on_checkpoint=on_checkpoint,
                                   proving=proving)

    def pending(self) -> list:
        """The in-flight transfers of this profile — every checkpoint in the bound store,
        reconstructed offline (:func:`lifecycle.progress_from_checkpoint`): no network read, so one
        unreachable chain can never hide the others. A malformed checkpoint yields a ``Progress``
        with ``next == "failed"`` and ``error`` set instead of raising; call ``wait()``/``recover()``
        on any entry to refresh it against live chain state.

        Nothing is ever dropped silently. A record this client cannot interpret at all — a route
        that no longer exists, a registry version this build did not write — and a file the store
        could not even read back come back as ``{"next": "failed", "error", "error_type"}`` entries
        (naming the ``checkpoint_id`` or the ``path``) alongside the healthy ``Progress`` objects,
        so a stale or corrupt file can never make a transfer that is still on the wire invisible.
        """
        store = self.checkpoints
        if store is None:
            return []
        lister = getattr(store, "list_with_problems", None)
        checkpoints, problems = lister() if callable(lister) else (store.list(), [])
        out: list = []
        for cp in checkpoints:
            try:
                out.append(_lifecycle.progress_from_checkpoint(self.registry, cp))
            except BridgeError as exc:
                # No Plan could be rebuilt at all (bad format/version/route): report it, never drop
                # it — an unreadable record may still be a transfer holding somebody's funds.
                out.append({"next": "failed", "error": str(exc), "error_type": type(exc).__name__,
                            "checkpoint_id": cp.id})
        out.extend({"next": "failed", **problem.to_dict()} for problem in problems)
        return out

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
