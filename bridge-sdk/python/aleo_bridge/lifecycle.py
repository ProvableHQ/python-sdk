"""Tier 1 lifecycle verbs — veil's ``quote → execute → wait`` with ``recover`` /
``resume`` / ``complete`` driven by ``Progress.next``.

Every function takes the registry (and, from later tasks on, the
:class:`~aleo_bridge.client.Bridge`) and touches only its public surface, so the
unit suite can run them against plain fakes without a network. ``Bridge`` binds
thin methods of the same names.

Order of the file follows the caller journey: planning (``prepare``), pricing
(``quote``), committing funds (``execute``), observing (``get_status``,
``wait``), and recovery (``recover``, ``resume``, ``complete``) — this task
only adds ``prepare`` and the ``resolve_route`` helper every later verb shares.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Callable

from ._plan import build_plan
from .checkpoint import Checkpoint, create_checkpoint
from .errors import (
    CheckpointInvalidError,
    ConfigurationError,
    InvalidAmountError,
    InvalidRecipientError,
    RegistryVersionMismatchError,
    RouteUnavailableError,
    UnsupportedRouteError,
)
from .registry import Asset, Chain, Registry, Route
from .types import (TERMINAL, AleoHyperlaneQuote, AleoXReserveQuote, Fee, Plan, Progress, Quote,
                    Receipt, Status, to_progress)
from .units import format_decimal_amount, parse_decimal_amount, resolve_amount

MINT_MODES = ("public", "record", "private")


# ── Route resolution (invariant 1) ────────────────────────────────────────────

@dataclass(frozen=True)
class ResolvedRoute:
    """The live registry entries behind one plan — re-resolved on every verb."""

    route: Route
    source_asset: Asset
    destination_asset: Asset
    source_chain: Chain
    destination_chain: Chain


def resolve_route(registry: Registry, plan: Plan) -> ResolvedRoute:
    """Re-resolve *plan* against the live registry; refuse stale or altered plans.

    Raises :class:`RegistryVersionMismatchError` when the plan was built from a
    different registry version (re-quote to fix) and
    :class:`CheckpointInvalidError` when its route topology no longer matches.
    """
    if plan.registry_version != registry.version:
        raise RegistryVersionMismatchError(
            f"Plan uses registry {plan.registry_version}; this client has "
            f"{registry.version}. Re-run quote() to rebuild the plan.")
    route = registry.route(plan.route_id)
    if (route.protocol != plan.protocol
            or route.source_asset_id != plan.source_asset_id
            or route.destination_asset_id != plan.destination_asset_id):
        raise CheckpointInvalidError(
            f"Plan route {plan.route_id} does not match the configured registry "
            "(protocol or asset pair differs). Re-run quote().")
    source = registry.asset(route.source_asset_id)
    destination = registry.asset(route.destination_asset_id)
    return ResolvedRoute(route, source, destination,
                         registry.chain(source.chain_id), registry.chain(destination.chain_id))


def _require_active(route: Route) -> None:
    if route.availability != "active":
        raise RouteUnavailableError(
            f"Route {route.id} is '{route.availability}': it is listed by the registry "
            "but cannot move funds until its deployment is reviewed. Pick an active route "
            "(bridge.registry.routes()).")


# ── prepare ───────────────────────────────────────────────────────────────────

def prepare(registry: Registry, *, source, destination, amount=None, amount_atomic=None,
            recipient: str, sender: str | None = None, protocol: str | None = None,
            mint_mode: str = "public") -> Plan:
    """Describe how *amount* of *source* moves to *destination* — pure, no network.

    Resolves the single non-disabled route for the asset pair (``protocol``
    disambiguates), validates the mint mode (non-public only for xReserve into
    Aleo), parses the amount with the source decimals AND re-parses it with the
    destination decimals so no precision is silently lost, regex-checks the
    recipient against the destination chain, then hands off to
    :func:`aleo_bridge._plan.build_plan` for the step list — the same builder
    ``bridge.eth.*`` / ``bridge.sol.*`` use, so a caller-supplied plan and a
    ``prepare()``-built one are always identical for the same route and amount.
    Nothing is signed and no chain is contacted; ``quote`` adds live prices on
    top of this.
    """
    src = registry.asset(source)
    dst = registry.asset(destination)
    route = registry.find_route(src.id, dst.id, protocol)
    dst_chain = registry.chain(dst.chain_id)

    if mint_mode not in MINT_MODES:
        raise ConfigurationError(f"mint_mode must be one of {MINT_MODES}, got {mint_mode!r}")
    if mint_mode != "public" and dst_chain.family != "aleo":
        raise ConfigurationError(
            "Aleo mint mode is only valid when the destination chain is Aleo")
    if route.protocol != "xreserve" and mint_mode != "public":
        raise ConfigurationError(
            "record and private mint modes are only supported by xReserve routes")

    atomic = resolve_amount(amount=amount, amount_atomic=amount_atomic, decimals=src.decimals)
    if atomic <= 0:
        raise InvalidAmountError("Bridge transfer amount must be greater than zero")
    parse_decimal_amount(format_decimal_amount(atomic, src.decimals), dst.decimals)  # destination precision check

    if dst.address_regex and not re.fullmatch(dst.address_regex, recipient):
        raise InvalidRecipientError(
            f"Recipient {recipient!r} does not match the {dst.chain_id} address format "
            f"({dst.address_regex})")

    return build_plan(registry, route, amount_atomic=atomic, recipient=recipient,
                      sender=sender, mint_mode=mint_mode)


# ── Connection helpers ────────────────────────────────────────────────────────

def _module(bridge, name: str):
    """``bridge.eth`` / ``bridge.sol`` or a ConfigurationError that says how to fix it.

    Checks the ``ethereum``/``solana`` connection attribute FIRST: on the real
    ``Bridge``, ``eth``/``sol`` are properties that themselves raise
    ``ConfigurationError`` when unconfigured, so ``getattr(bridge, name, None)``
    would never see the ``None`` default — it would let that raise propagate
    with the property's own (less specific) message instead of this one.
    """
    conn = getattr(bridge, "ethereum" if name == "eth" else "solana", None)
    if conn is None:
        chain, extra, env = (("Ethereum", "evm", "ETHEREUM_RPC_URL / EVM_PRIVATE_KEY") if name == "eth"
                             else ("Solana", "solana", "SOLANA_RPC_URL / SOLANA_PRIVATE_KEY"))
        raise ConfigurationError(
            f"This transfer needs a configured {chain} connection: pass "
            f"{'ethereum' if name == 'eth' else 'solana'}= to Bridge(...) (pip install "
            f"'aleo-bridge-sdk[{extra}]') or set {env} for Bridge.from_env().")
    return getattr(bridge, name)


def _credits_asset_id(chain: Chain) -> str:
    return f"{chain.id}/aleo"


# ── quote ─────────────────────────────────────────────────────────────────────

def quote(bridge, *, source, destination, amount=None, amount_atomic=None, recipient: str,
          sender: str | None = None, protocol: str | None = None, mint_mode: str = "public",
          secret_nonce: str = "0scalar") -> Quote:
    """Price a transfer: ``prepare`` + the source-side live read for the route kind.

    Returns one of ``EvmHyperlaneQuote`` / ``SolanaHyperlaneQuote`` /
    ``AleoHyperlaneQuote`` / ``EvmXReserveQuote`` / ``AleoXReserveQuote``
    (``quote.kind``), each carrying the canonical ``plan`` that ``execute`` takes.
    Aleo-origin xReserve quotes make no network call (fixed withdrawal fee).
    Nothing is signed.

    Dispatches to ``bridge.eth``/``bridge.sol`` with ``plan=`` (never a re-derived
    ``asset``/``recipient``/``amount_atomic`` form): those modules re-resolve the
    route from the plan and validate it against the live registry themselves. The
    quote they return is then re-stamped with this function's own ``plan`` (via
    ``replace``) so the plan on the result is always exactly what ``prepare()``
    built, regardless of what the module attached internally.
    """
    plan = prepare(bridge.registry, source=source, destination=destination, amount=amount,
                   amount_atomic=amount_atomic, recipient=recipient, sender=sender,
                   protocol=protocol, mint_mode=mint_mode)
    resolved = resolve_route(bridge.registry, plan)
    _require_active(resolved.route)
    family = resolved.source_chain.family

    if plan.protocol == "hyperlane" and family == "evm":
        q = _module(bridge, "eth").quote_transfer_remote(plan=plan)
        return replace(q, plan=plan)
    if plan.protocol == "hyperlane" and family == "solana":
        q = _module(bridge, "sol").quote_transfer_remote(plan=plan)
        return replace(q, plan=plan)
    if plan.protocol == "hyperlane" and family == "aleo":
        gas = bridge.hyperlane.quote_gas_payment(plan.source_asset_id)
        fee = Fee(kind="protocol", chain_id=resolved.source_chain.id,
                  asset_id=_credits_asset_id(resolved.source_chain),
                  amount=format_decimal_amount(gas.payment_microcredits, 6), estimated=True)
        return AleoHyperlaneQuote(kind="aleo-hyperlane", plan=plan, fees=(fee,), amount_out=plan.amount,
                                  gas_limit=gas.gas_limit, gas_overhead=gas.gas_overhead,
                                  gas_price=gas.gas_price, exchange_rate=gas.exchange_rate,
                                  payment_microcredits=gas.payment_microcredits)
    if plan.protocol == "xreserve" and family == "evm":
        q = _module(bridge, "eth").quote_deposit_usdc(plan=plan, secret_nonce=secret_nonce)
        return replace(q, plan=plan)
    if plan.protocol == "xreserve" and family == "aleo":
        raw = resolved.route.metadata.get("withdrawalFeeAtomic")
        if not isinstance(raw, str) or not raw.isdigit():
            raise RouteUnavailableError(f"xReserve withdrawal fee is missing or invalid: {plan.route_id}")
        fee_atomic = int(raw)
        decimals = resolved.source_asset.decimals
        fee_human = format_decimal_amount(fee_atomic, decimals)
        if plan.amount_atomic <= fee_atomic:
            raise InvalidAmountError(
                f"xReserve burn amount must exceed the {fee_human} {resolved.source_asset.symbol} "
                f"withdrawal fee (got {plan.amount})")
        return AleoXReserveQuote(
            kind="aleo-xreserve", plan=plan,
            fees=(Fee(kind="protocol", chain_id=resolved.source_chain.id, asset_id=resolved.source_asset.id,
                      amount=fee_human, estimated=False),),
            amount_out=format_decimal_amount(plan.amount_atomic - fee_atomic, decimals),
            withdrawal_fee_atomic=fee_atomic)
    raise UnsupportedRouteError(
        f"Unsupported {plan.protocol} source chain family: {family} ({plan.route_id})")


# ── Checkpoint emission ───────────────────────────────────────────────────────

def _persist(bridge, checkpoint: Checkpoint, receipt: Receipt | None, *, previous_id: str | None = None) -> None:
    """Mirror *checkpoint* into the bound store: supersede the previous id, drop terminal ones.

    ``receipt`` is ``None`` for a boundary the protocol module reduced and saved itself: the
    supersede still runs (a module only ever saves — it never deletes the checkpoint its own
    next boundary replaces), the save does not.
    """
    store = getattr(bridge, "checkpoints", None)
    if store is None:
        return
    if previous_id is not None and previous_id != checkpoint.id:
        store.delete(previous_id)
    if receipt is None:
        return
    if receipt.status in TERMINAL:
        store.delete(checkpoint.id)
    else:
        store.save(checkpoint)


class _Emitter:
    """Turns receipts into checkpoints: caller callback first, then the bound store.

    Two channels feed it — a protocol module's own ``on_checkpoint`` (which hands over a
    ``Checkpoint`` it has already reduced and saved) and ``execute``'s own emission once the
    send returns. A boundary that arrives through both is handed to the caller once: the two
    reductions compare equal, being the same receipt reduced against the same plan.
    """

    def __init__(self, bridge, plan: Plan, on_checkpoint: Callable | None) -> None:
        self._bridge, self._plan, self._cb = bridge, plan, on_checkpoint
        self._last_id: str | None = None
        self._last: Checkpoint | None = None

    def __call__(self, receipt) -> Checkpoint:
        module_emitted = isinstance(receipt, Checkpoint)
        checkpoint = receipt if module_emitted else create_checkpoint(self._plan, receipt, self._bridge.registry)
        if checkpoint != self._last:
            if hasattr(self._bridge, "events"):
                # test hook: FakeBridge records the ordering of proving/checkpoint/broadcast
                label = receipt.status.value if isinstance(receipt, Receipt) else "module"
                self._bridge.events.append((f"checkpoint:{label}", checkpoint.id))
            if self._cb is not None:
                self._cb(checkpoint)                     # the caller's own callback: errors are theirs
        _persist(self._bridge, checkpoint, None if module_emitted else receipt, previous_id=self._last_id)
        self._last_id, self._last = checkpoint.id, checkpoint
        return checkpoint


# ── Execution helpers ─────────────────────────────────────────────────────────

def _assert_sender(plan: Plan, address: str | None, *, family: str) -> None:
    """Refuse a plan prepared for a different account than the one that would sign it.

    EVM addresses are hex and their checksum casing carries no identity, so they compare
    case-insensitively; Solana addresses are base58, where case IS part of the address.
    """
    if not plan.sender or not address:
        return
    same = plan.sender.lower() == address.lower() if family == "evm" else plan.sender == address
    if not same:
        raise ConfigurationError(
            f"Plan sender {plan.sender} does not match the connected account {address}. "
            "Re-quote with sender=None or the connection's own address.")


def _read_destination_balance(bridge, plan: Plan, resolved: ResolvedRoute) -> int | None:
    """The recipient's destination balance, or None when we cannot read it.

    Only read when the destination connection IS the recipient (there is no per-address balance
    read in the module contracts); otherwise omit the delivery-verification pair rather than
    baseline the wrong account.
    """
    chain, asset = resolved.destination_chain, resolved.destination_asset
    if chain.family == "evm":
        conn = getattr(bridge, "ethereum", None)
        if (conn is None or not conn.address
                or conn.address.lower() != plan.recipient.lower()
                or asset.locator is None or asset.locator.kind not in ("native", "evm-contract")):
            return None
        return int(bridge.eth.balance(asset.id))
    if chain.family == "solana":
        conn = getattr(bridge, "solana", None)
        if (conn is None or conn.address != plan.recipient
                or asset.locator is None or asset.locator.kind != "native"):
            return None
        return int(bridge.sol.balance())
    return None            # Aleo private records / token mappings: protocol signal instead


def _delivery_verification(bridge, plan: Plan, resolved: ResolvedRoute) -> dict[str, str]:
    before = _read_destination_balance(bridge, plan, resolved)
    if before is None:
        return {}
    expected = parse_decimal_amount(plan.amount, resolved.destination_asset.decimals)
    return {"destinationBalanceBeforeAtomic": str(before),
            "expectedDestinationIncreaseAtomic": str(expected)}


def _prepare_aleo(call, proving: str):
    if proving == "delegate":
        return call.delegate_prepared()
    if proving == "local":
        return call.prove()
    raise ConfigurationError(f"proving must be 'delegate' (DPS) or 'local', got {proving!r}")


def _run_aleo_leg(bridge, plan: Plan, call, *, proving: str, emit: _Emitter,
                  extra_state: dict[str, Any]) -> Receipt:
    """Invariant 3: prove → checkpoint the exact transaction → broadcast → checkpoint the id."""
    prepared = _prepare_aleo(call, proving)
    emit(Receipt(id=prepared.transaction_id, protocol=plan.protocol,
                 status=Status.SOURCE_SUBMISSION_PENDING,
                 protocol_state={"routeId": plan.route_id, "preparedTransaction": prepared.serialized,
                                 **extra_state}))
    result = call.submit_prepared(prepared, wait=False)      # polling is wait()'s job, not execute()'s
    receipt: Receipt = result.receipt
    receipt = receipt.replace(id=prepared.transaction_id, status=Status.SOURCE_CONFIRMING,
                              source_tx_id=prepared.transaction_id,
                              protocol_state={**receipt.protocol_state, "routeId": plan.route_id, **extra_state})
    emit(receipt)
    return receipt


def _send_call(call, emit: _Emitter, poll_seconds: float, timeout_seconds: float) -> Receipt:
    result = call.send(wait=True, timeout_seconds=timeout_seconds, poll_seconds=poll_seconds,
                       on_checkpoint=emit)
    receipt: Receipt = result.receipt
    emit(receipt)
    return receipt


def _aleo_hyperlane_mode(mode: str | None) -> bool:
    if mode is None or mode == "caller":
        return False
    if mode == "signer":
        return True
    raise ConfigurationError(f"Aleo Hyperlane mode must be 'caller' or 'signer', got {mode!r}")


def _xreserve_burn_mode(mode: str | None) -> str:
    if mode is None:
        return "private"
    if mode in ("private", "public", "public-as-signer"):
        return mode
    raise ConfigurationError(
        f"Aleo xReserve mode must be 'private', 'public' or 'public-as-signer', got {mode!r}")


def _mint_secret(plan: Plan, secret_nonce: str | None) -> str:
    """The secret the EVM deposit commits to. A private mint must supply its own.

    ``secret_nonce`` is not a ``Plan`` field and the SDK never stores it: a private deposit that
    quietly fell back to the ``"0scalar"`` default would commit to a hook nobody can reproduce,
    and ``complete`` needs the same value again to mint the record.
    """
    if plan.mint_mode == "private":
        if not secret_nonce:
            raise ConfigurationError(
                "a secret_nonce is required for a private xReserve mint: the deposit commits to "
                "(recipient, secret_nonce) and complete() needs the same value again — keep it, "
                "the SDK never stores it")
        return secret_nonce
    return secret_nonce or "0scalar"


# ── execute ───────────────────────────────────────────────────────────────────

def execute(bridge, plan: Plan, *, on_checkpoint: Callable | None = None, proving: str = "delegate",
            mode: str | None = None, record: str | None = None, merkle_proof: str | None = None,
            gas_payment_microcredits: int | None = None, secret_nonce: str | None = None,
            poll_seconds: float = 1.0, timeout_seconds: float = 120.0) -> Progress:
    """Commit funds on the source chain and return the transfer's ``Progress``.

    Runs every source-chain leg for ``plan`` — approval(s) → deposit / dispatch / burn — emitting
    a ``Checkpoint`` at each boundary (after each approval hash; after proving and BEFORE
    broadcast for Aleo legs; after broadcast).  Aleo legs prove with ``proving="delegate"`` (DPS)
    or ``"local"``; ``mode`` is ``"caller"|"signer"`` for Aleo Hyperlane and ``"private"|"public"|
    "public-as-signer"`` for Aleo xReserve burns (``record`` / ``merkle_proof`` feed a private
    burn).  The Hyperlane hook payment is re-quoted right before proving unless
    ``gas_payment_microcredits`` pins it.  ``secret_nonce`` is the private-mint commitment secret
    for an EVM xReserve deposit — required when ``plan.mint_mode == "private"``; keep it,
    ``complete`` needs it again and the SDK never stores it.

    EVM and Solana legs are dispatched through the module's own ``plan=`` surface, so the route,
    registry version, sender and every plan field are re-validated by the module that builds the
    transaction.  Returns after broadcast; call ``wait`` to observe acceptance and delivery.  Once
    the irreversible step is broadcast, recover from the checkpoint — never re-run ``execute``.
    """
    resolved = resolve_route(bridge.registry, plan)
    _require_active(resolved.route)
    family = resolved.source_chain.family
    emit = _Emitter(bridge, plan, on_checkpoint)

    if plan.protocol == "hyperlane" and family == "evm":
        eth = _module(bridge, "eth")
        _assert_sender(plan, bridge.ethereum.address, family=family)
        call = eth.transfer_remote(plan=plan)
        return to_progress(plan, _send_call(call, emit, poll_seconds, timeout_seconds))

    if plan.protocol == "hyperlane" and family == "solana":
        sol = _module(bridge, "sol")
        _assert_sender(plan, bridge.solana.address, family=family)
        call = sol.transfer_remote(plan=plan)
        return to_progress(plan, _send_call(call, emit, poll_seconds, timeout_seconds))

    if plan.protocol == "hyperlane" and family == "aleo":
        as_signer = _aleo_hyperlane_mode(mode)
        verification = _delivery_verification(bridge, plan, resolved)
        gas = gas_payment_microcredits
        if gas is None:
            gas = bridge.hyperlane.quote_gas_payment(plan.source_asset_id).payment_microcredits
        call = bridge.hyperlane.transfer_remote(plan.source_asset_id, plan.recipient,
                                                amount_atomic=plan.amount_atomic, as_signer=as_signer,
                                                gas_payment_microcredits=gas)
        return to_progress(plan, _run_aleo_leg(bridge, plan, call, proving=proving, emit=emit,
                                               extra_state=verification))

    if plan.protocol == "xreserve" and family == "evm":
        eth = _module(bridge, "eth")
        _assert_sender(plan, bridge.ethereum.address, family=family)
        nonce = _mint_secret(plan, secret_nonce)
        call = eth.deposit_usdc(plan=plan, secret_nonce=nonce)
        return to_progress(plan, _send_call(call, emit, poll_seconds, timeout_seconds))

    if plan.protocol == "xreserve" and family == "aleo":
        burn_mode = _xreserve_burn_mode(mode)
        call = bridge.xreserve.burn(plan.recipient, amount_atomic=plan.amount_atomic, mode=burn_mode,
                                    record=record, merkle_proof=merkle_proof)
        return to_progress(plan, _run_aleo_leg(bridge, plan, call, proving=proving, emit=emit, extra_state={}))

    raise UnsupportedRouteError(f"Unsupported {plan.protocol} source chain family: {family} ({plan.route_id})")


__all__ = ["MINT_MODES", "ResolvedRoute", "execute", "prepare", "quote", "resolve_route"]
