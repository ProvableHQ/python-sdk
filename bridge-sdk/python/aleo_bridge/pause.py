"""Operator pause switches, read live on both ends of a route before any funds move.

Every bridge deployment has a kill switch somewhere: the ``pause`` mapping on the Aleo warp and
USDCx programs, ``paused()`` on Circle's xReserve and CCTP transmitters, a ``PausableHook`` /
``PausableIsm`` inside the Ethereum Hyperlane Mailbox's hook and ISM trees, and the enrolled
remote-router table of the Solana warp program (Hyperlane's Sealevel token program has no pause
flag of its own; un-enrolling the router is how an operator stops it). ``quote``, ``execute``,
``resume`` and ``complete`` read every switch governing the route and refuse when one is set.

Fail closed: a switch that cannot be read after the retry budget below is ``unknown`` and blocks
exactly like ``paused``. Nothing here is cached — a pause can flip between quote and execute.

Assumption stated once: the deployed Solana warp programs match Hyperlane's published
``hyperlane-sealevel-token`` source at hyperlane-monorepo ``45c0988`` (the commit the rest of
``_sealevel.py`` derives from); they are BPF-upgradeable, so this cannot be checked on chain.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Sequence, TypeVar

from .errors import ConfigurationError
from .eth import Ethereum
from .registry import Asset, Chain, Route
from .sol import DEFAULT_SOLANA_RPC_URL, Solana
from .types import PauseStatus, PauseSwitch

T = TypeVar("T")

IMMEDIATE_RETRIES = 2                       # retried at once, for a blip
BACKOFF_SECONDS = (0.5, 1.0, 2.0, 4.0)      # then exponential backoff; 7 attempts in all


class ReadFailed(Exception):
    """Internal: a switch read exhausted its retries. ``attempts`` and the ``last`` transient error."""

    def __init__(self, attempts: int, last: Exception) -> None:
        self.attempts, self.last = attempts, last
        super().__init__(f"{type(last).__name__}: {last} after {attempts} attempts")


def read_with_retry(fn: Callable[[int], T], *, sleep: Callable[[float], None] = time.sleep) -> T:
    """Call ``fn(attempt)`` until it returns: transient errors (``lifecycle._is_transient_error``) are
    retried ``IMMEDIATE_RETRIES`` times with no delay, then once per ``BACKOFF_SECONDS`` entry; anything
    else propagates at once. Raises :class:`ReadFailed` when the budget is spent."""
    from .lifecycle import _is_transient_error          # lifecycle imports this module; keep the cycle lazy

    delays = [0.0] * IMMEDIATE_RETRIES + list(BACKOFF_SECONDS)
    for attempt in range(len(delays) + 1):
        try:
            return fn(attempt)
        except Exception as exc:                        # noqa: BLE001 — classified right below
            if not _is_transient_error(exc):
                raise
            if attempt == len(delays):
                raise ReadFailed(attempt + 1, exc) from exc
            sleep(delays[attempt])
    raise AssertionError("unreachable")


def rotate(urls: Sequence[str], attempt: int) -> str:
    """The endpoint for *attempt*: walks a list of public RPC URLs so retries land on a different host."""
    return urls[attempt % len(urls)]


# ── Aleo ─────────────────────────────────────────────────────────────────────

def _aleo_mapping_value(bridge: Any, program: str, mapping: str, key: str, sleep: Callable[[float], None]) -> str | None:
    """The mapping value, after confirming the program declares *mapping*.

    The node answers ``null`` for a missing key, for a mapping the program never declared and for a
    program that does not exist, so a typo in a pinned mapping name would read as "open" forever.
    Checking the declared list first turns that into a ``ConfigurationError``.
    """
    from aleo.facade.errors import ProgramNotFound

    def declared(_attempt: int) -> list[str]:
        try:
            return list(bridge.program(program).mappings())
        except ProgramNotFound:
            return []
    if mapping not in read_with_retry(declared, sleep=sleep):
        raise ConfigurationError(
            f"{program} does not declare mapping {mapping!r}; the registry pins a program this node does not serve")
    return read_with_retry(lambda _attempt: bridge.mapping_value(program, mapping, key), sleep=sleep)


def read_aleo_flag(bridge: Any, *, chain_id: str, side: str, program: str, mapping: str, key: str,
                   sleep: Callable[[float], None] = time.sleep) -> PauseSwitch:
    """A boolean pause mapping (``pause[true]``, ``paused[true]``, the ARC-20 ``pause__[false]``): ``true`` is paused."""
    switch = f"{mapping}[{key}]"
    try:
        value = _aleo_mapping_value(bridge, program, mapping, key, sleep)
    except ReadFailed as exc:
        return PauseSwitch(chain_id, side, program, switch, "unknown", str(exc))
    if value not in (None, "true", "false"):
        raise ConfigurationError(f"{program}.{switch} expected a boolean literal, got {value!r}")
    return PauseSwitch(chain_id, side, program, switch, "paused" if value == "true" else "open", value)


def read_aleo_remote_router(bridge: Any, *, chain_id: str, side: str, program: str, domain: int,
                            sleep: Callable[[float], None] = time.sleep) -> PauseSwitch:
    """The warp program's enrolled router for *domain*; a missing entry means the operator un-enrolled the route."""
    switch = f"remote_routers[{domain}u32]"
    try:
        value = _aleo_mapping_value(bridge, program, "remote_routers", f"{domain}u32", sleep)
    except ReadFailed as exc:
        return PauseSwitch(chain_id, side, program, switch, "unknown", str(exc))
    return PauseSwitch(chain_id, side, program, switch, "open" if value is not None else "paused", value)


# ── EVM ──────────────────────────────────────────────────────────────────────

HOOK_ROUTING, HOOK_AGGREGATION, HOOK_FALLBACK_ROUTING, HOOK_PAUSABLE = 1, 2, 5, 7     # Hyperlane IPostDispatchHook.Types
ISM_ROUTING, ISM_AGGREGATION, ISM_NULL = 1, 2, 6                                      # IInterchainSecurityModule.Types
TREE_DEPTH_LIMIT = 4


class _TreeTooDeep(Exception):
    pass


def _evm_call(w3: Any, address: str, fn: str, *args: Any, sleep: Callable[[float], None]) -> Any:
    """One view call through the retry loop. *w3* is a ``Web3`` or an ``attempt -> Web3`` callable (public
    endpoints rotate per attempt). A revert is a ``ConfigurationError`` (the registry names a contract that
    does not answer this getter), never retried."""
    from .eth import _web3
    from ._evm_abi import PAUSE_ABI
    web3 = _web3()
    checksummed = web3.Web3.to_checksum_address(address)

    def call(attempt: int) -> Any:
        client = w3(attempt) if callable(w3) else w3
        return getattr(client.eth.contract(address=checksummed, abi=PAUSE_ABI).functions, fn)(*args).call()
    try:
        return read_with_retry(call, sleep=sleep)
    except web3.exceptions.ContractLogicError as exc:
        raise ConfigurationError(f"{fn}() on {address} reverted: {exc}") from exc


def read_evm_paused(w3: Any, *, chain_id: str, side: str, address: str,
                    sleep: Callable[[float], None] = time.sleep) -> PauseSwitch:
    """OpenZeppelin ``paused()`` on an xReserve or CCTP MessageTransmitter contract."""
    try:
        paused = bool(_evm_call(w3, address, "paused", sleep=sleep))
    except ReadFailed as exc:
        return PauseSwitch(chain_id, side, address, "paused()", "unknown", str(exc))
    return PauseSwitch(chain_id, side, address, "paused()", "paused" if paused else "open", str(paused).lower())


def _walk(w3: Any, address: str, *, far_domain: int, kind: str, depth: int, sleep: Callable[[float], None],
          out: list[tuple[str, bool]]) -> None:
    """Depth-first over a Hyperlane hook (``kind="hook"``) or ISM tree, collecting (address, paused) for every
    pausable member. Routing nodes resolve *far_domain*; aggregation nodes expand their members."""
    if depth > TREE_DEPTH_LIMIT:
        raise _TreeTooDeep(address)
    if kind == "hook":
        node_type = _evm_call(w3, address, "hookType", sleep=sleep)
        if node_type in (HOOK_ROUTING, HOOK_FALLBACK_ROUTING):
            _walk(w3, _evm_call(w3, address, "hooks", far_domain, sleep=sleep), far_domain=far_domain, kind=kind,
                  depth=depth + 1, sleep=sleep, out=out)
        elif node_type == HOOK_AGGREGATION:
            for member in _evm_call(w3, address, "hooks", b"", sleep=sleep):
                _walk(w3, member, far_domain=far_domain, kind=kind, depth=depth + 1, sleep=sleep, out=out)
        elif node_type == HOOK_PAUSABLE:
            out.append((address, bool(_evm_call(w3, address, "paused", sleep=sleep))))
        return
    node_type = _evm_call(w3, address, "moduleType", sleep=sleep)
    if node_type == ISM_ROUTING:
        _walk(w3, _evm_call(w3, address, "module", far_domain, sleep=sleep), far_domain=far_domain, kind=kind,
              depth=depth + 1, sleep=sleep, out=out)
    elif node_type == ISM_AGGREGATION:
        for member in _evm_call(w3, address, "modulesAndThreshold", b"", sleep=sleep)[0]:
            _walk(w3, member, far_domain=far_domain, kind=kind, depth=depth + 1, sleep=sleep, out=out)
    elif node_type == ISM_NULL:
        try:                                    # a PausableIsm reports NULL; a plain NullIsm/TrustedRelayerIsm has no paused()
            out.append((address, bool(_evm_call(w3, address, "paused", sleep=sleep))))
        except ConfigurationError:
            pass


def _tree_switches(w3: Any, *, chain_id: str, side: str, mailbox: str, far_domain: int, kind: str,
                   roots: tuple[str, ...], switch: str, sleep: Callable[[float], None]) -> list[PauseSwitch]:
    found: list[tuple[str, bool]] = []
    try:
        for getter in roots:
            _walk(w3, _evm_call(w3, mailbox, getter, sleep=sleep), far_domain=far_domain, kind=kind, depth=1,
                  sleep=sleep, out=found)
    except ReadFailed as exc:
        return [PauseSwitch(chain_id, side, mailbox, switch, "unknown", str(exc))]
    except _TreeTooDeep as exc:
        return [PauseSwitch(chain_id, side, mailbox, switch, "unknown",
                            f"{kind} tree deeper than depth {TREE_DEPTH_LIMIT} at {exc}")]
    return [PauseSwitch(chain_id, side, address, switch, "paused" if paused else "open", str(paused).lower())
            for address, paused in found]


def read_mailbox_hook_pauses(w3: Any, *, chain_id: str, side: str, mailbox: str, far_domain: int,
                             sleep: Callable[[float], None] = time.sleep) -> list[PauseSwitch]:
    """Every ``PausableHook`` under the Mailbox's default and required hooks for *far_domain* (outbound dispatch)."""
    return _tree_switches(w3, chain_id=chain_id, side=side, mailbox=mailbox, far_domain=far_domain, kind="hook",
                          roots=("defaultHook", "requiredHook"), switch="PausableHook.paused()", sleep=sleep)


def read_mailbox_ism_pauses(w3: Any, *, chain_id: str, side: str, mailbox: str, far_domain: int,
                            sleep: Callable[[float], None] = time.sleep) -> list[PauseSwitch]:
    """Every ``PausableIsm`` under the Mailbox's default ISM for messages from *far_domain* (inbound delivery)."""
    return _tree_switches(w3, chain_id=chain_id, side=side, mailbox=mailbox, far_domain=far_domain, kind="ism",
                          roots=("defaultIsm",), switch="PausableIsm.paused()", sleep=sleep)


# ── Solana ───────────────────────────────────────────────────────────────────

def read_solana_remote_router(solana: Any, *, chain_id: str, side: str, token_pda: str, domain: int,
                              sleep: Callable[[float], None] = time.sleep) -> PauseSwitch:
    """The warp program's enrolled router for *domain* in its token PDA; a missing account or entry is paused."""
    from ._sealevel import decode_hyperlane_token_remote_routers
    from .sol import SolModule
    switch = f"remote_routers[{domain}]"
    reader = SolModule(None, solana)
    try:
        data = read_with_retry(lambda _attempt: reader._account_data(token_pda), sleep=sleep)
    except ReadFailed as exc:
        return PauseSwitch(chain_id, side, token_pda, switch, "unknown", str(exc))
    router = None if data is None else decode_hyperlane_token_remote_routers(data).get(domain)
    return PauseSwitch(chain_id, side, token_pda, switch, "open" if router else "paused",
                       None if router is None else "0x" + router.hex())


# ── Per-route assembly ───────────────────────────────────────────────────────

SIDES = ("source", "destination")


class PauseReader:
    """Reads every switch governing a route, on the requested sides, in one pass. Nothing is cached."""

    def __init__(self, bridge: Any, *, sleep: Callable[[float], None] = time.sleep) -> None:
        self._bridge = bridge
        self._sleep = sleep

    # -- connections --
    def evm_w3(self, chain_id: str) -> Any:
        """A ``Web3`` for *chain_id*, or an ``attempt -> Web3`` callable over the registry's public read
        endpoints: configured connection (constructor, env or profile) first, public endpoints last."""
        bridge, registry = self._bridge, self._bridge.registry
        conn = bridge._evm_connections.get(chain_id)
        if conn is not None:
            return conn.w3
        urls = registry.public_read_rpc_urls(chain_id)
        if not urls:
            raise ConfigurationError(
                f"No RPC connection for {chain_id} to read its pause switches: pass ethereum=/evm= to Bridge(...), "
                f"save one with profile.set_rpc_url({chain_id!r}, url), or set {chain_id.upper()}_RPC_URL")
        return lambda attempt: Ethereum(rotate(urls, attempt)).w3

    def _solana(self) -> Any:
        return self._bridge.solana if self._bridge.solana is not None else Solana(DEFAULT_SOLANA_RPC_URL)

    # -- switches per chain family --
    def _aleo(self, side: str, route: Route, asset: Asset, far_chain: Chain) -> list[PauseSwitch]:
        bridge, out = self._bridge, []
        if route.protocol == "hyperlane":
            program = asset.locator.value if asset.locator is not None else route.meta_str("aleoRouterProgram")
            out.append(read_aleo_flag(bridge, chain_id=asset.chain_id, side=side, program=program, mapping="pause",
                                      key="true", sleep=self._sleep))
            out.append(read_aleo_remote_router(bridge, chain_id=asset.chain_id, side=side, program=program,
                                               domain=far_chain.protocol_domains["hyperlane"], sleep=self._sleep))
            if asset.privacy is not None and asset.privacy.kind == "arc20":
                out.append(read_aleo_flag(bridge, chain_id=asset.chain_id, side=side, program=asset.privacy.program,
                                          mapping="pause__", key="false", sleep=self._sleep))
        elif route.protocol == "xreserve":
            out.append(read_aleo_flag(bridge, chain_id=asset.chain_id, side=side, program=route.meta_str("bridgeProgram"),
                                      mapping="paused", key="true", sleep=self._sleep))
            out.append(read_aleo_flag(bridge, chain_id=asset.chain_id, side=side, program=route.meta_str("remoteToken"),
                                      mapping="pause", key="true", sleep=self._sleep))
        return out

    def _evm(self, side: str, route: Route, chain: Chain, far_chain: Chain) -> list[PauseSwitch]:
        w3 = self.evm_w3(chain.id)
        if route.protocol == "hyperlane":
            reader = read_mailbox_hook_pauses if side == "source" else read_mailbox_ism_pauses
            return reader(w3, chain_id=chain.id, side=side, mailbox=route.meta_str("mailboxAddress"),
                          far_domain=far_chain.protocol_domains["hyperlane"], sleep=self._sleep)
        key = "xReserveContract" if route.protocol == "xreserve" else "messageTransmitter"
        return [read_evm_paused(w3, chain_id=chain.id, side=side, address=route.meta_str(key), sleep=self._sleep)]

    def _solana_switches(self, side: str, route: Route, asset: Asset, far_asset: Asset) -> list[PauseSwitch]:
        registry = self._bridge.registry
        if side == "source":
            token_pda = route.meta_str("tokenPda")
        else:                                       # the paired Solana -> Aleo route carries the deployment metadata
            paired = [r for r in registry.routes(bridge_protocol="hyperlane", include_unavailable=True,
                                                 environment=self._bridge.environment)
                      if r.source_asset_id == asset.id and r.destination_asset_id == far_asset.id]
            if not paired:
                raise ConfigurationError(f"No Solana-origin route pairs with {route.id} to locate its token PDA")
            token_pda = paired[0].meta_str("tokenPda")
        aleo_domain = registry.chain(far_asset.chain_id).protocol_domains["hyperlane"]
        return [read_solana_remote_router(self._solana(), chain_id=asset.chain_id, side=side, token_pda=token_pda,
                                          domain=aleo_domain, sleep=self._sleep)]

    def status(self, route: Route, *, sides: Sequence[str] = SIDES) -> PauseStatus:
        registry = self._bridge.registry
        assets = {"source": registry.asset(route.source_asset_id), "destination": registry.asset(route.destination_asset_id)}
        chains = {side: registry.chain(a.chain_id) for side, a in assets.items()}
        switches: list[PauseSwitch] = []
        for side in SIDES:
            if side not in sides:
                continue
            far = "destination" if side == "source" else "source"
            family = chains[side].family
            if family == "aleo":
                switches += self._aleo(side, route, assets[side], chains[far])
            elif family == "evm":
                switches += self._evm(side, route, chains[side], chains[far])
            elif family == "solana":
                switches += self._solana_switches(side, route, assets[side], assets[far])
        return PauseStatus(route.id, tuple(switches))


__all__ = ["BACKOFF_SECONDS", "IMMEDIATE_RETRIES", "PauseReader", "ReadFailed", "SIDES", "read_aleo_flag",
           "read_aleo_remote_router", "read_evm_paused", "read_mailbox_hook_pauses", "read_mailbox_ism_pauses",
           "read_solana_remote_router", "read_with_retry", "rotate"]
