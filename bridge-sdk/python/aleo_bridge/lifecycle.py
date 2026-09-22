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

import json
import re
import time
from dataclasses import dataclass, replace
from typing import Any, Callable

from . import _sealevel
from ._calls import is_duplicate_submission
from .encoding import HOOK_DATA_BYTES
from ._plan import build_plan
from .checkpoint import Checkpoint, create_checkpoint
from .errors import (
    AttestationError,
    BridgeError,
    CheckpointInvalidError,
    ConfigurationError,
    DeliveryUnknownError,
    InvalidAmountError,
    InvalidRecipientError,
    NotResumableError,
    PollingTimeoutError,
    RegistryVersionMismatchError,
    RouteUnavailableError,
    UnsupportedRouteError,
)
from .registry import Asset, Chain, Registry, Route
from .types import (CALLER_BOUNDARIES, TERMINAL, AleoHyperlaneQuote, AleoXReserveQuote, Attestation,
                    Fee, Plan, Progress, Quote, Receipt, Status, to_progress)
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
    Aleo-origin xReserve quotes make no network call: the withdrawal fee comes
    from the registry literal, which is an ASSUMPTION and not a read of what the
    withdrawal will charge (the fee is flagged ``estimated`` and ``amount_out``
    is therefore a lower bound — the live testnet fee on 2026-09-18 was ≈1.0035
    USDC against the 2 USDCx literal). Nothing is signed.

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
        fee_atomic = _xreserve_withdrawal_fee_atomic(resolved)
        if fee_atomic is None:
            raise RouteUnavailableError(f"xReserve withdrawal fee is missing or invalid: {plan.route_id}")
        decimals = resolved.source_asset.decimals
        fee_human = format_decimal_amount(fee_atomic, decimals)
        if plan.amount_atomic <= fee_atomic:
            raise InvalidAmountError(
                f"xReserve burn amount must exceed the {fee_human} {resolved.source_asset.symbol} "
                f"withdrawal fee (got {plan.amount})")
        return AleoXReserveQuote(
            kind="aleo-xreserve", plan=plan,
            # estimated: the registry literal is the quote ASSUMPTION, not a read of the fee the
            # withdrawal will actually charge (the live testnet fee on 2026-09-18 was ~1.0035 USDC
            # against this 2 USDCx literal), so amount_out below is a lower bound. The literal
            # still drives the burn-minimum guard above — that guard must stay conservative.
            fees=(Fee(kind="protocol", chain_id=resolved.source_chain.id, asset_id=resolved.source_asset.id,
                      amount=fee_human, estimated=True),),
            amount_out=format_decimal_amount(plan.amount_atomic - fee_atomic, decimals),
            withdrawal_fee_atomic=fee_atomic)
    raise UnsupportedRouteError(
        f"Unsupported {plan.protocol} source chain family: {family} ({plan.route_id})")


# ── Checkpoint emission ───────────────────────────────────────────────────────

def _persist(bridge, checkpoint: Checkpoint, receipt: Receipt, *, previous_id: str | None = None) -> None:
    """Mirror *checkpoint* into the bound store: save (or drop, if terminal) BEFORE superseding
    the previous id — never the reverse, so a crash between the two steps still leaves a valid
    record for the transfer rather than a moment where the store holds neither.
    """
    store = getattr(bridge, "checkpoints", None)
    if store is None:
        return
    if receipt.status in TERMINAL:
        store.delete(checkpoint.id)
    else:
        store.save(checkpoint)
    if previous_id is not None and previous_id != checkpoint.id:
        store.delete(previous_id)


class _Emitter:
    """Turns receipts into checkpoints: caller callback first, then the bound store.

    Two channels feed it — a protocol module's own ``on_checkpoint`` (which hands over a
    ``Checkpoint`` it has already reduced, and saves ITSELF only after this call returns) and
    ``execute``'s own emission once the send returns. A boundary that arrives through both is
    handed to the caller once: the two reductions compare equal, being the same receipt reduced
    against the same plan.

    A module-emitted checkpoint supersedes the previous id before the module has actually saved
    the new one — deleting the old id here (save-then-delete, brief §review item 6) would leave a
    window where the store holds neither if it crashed. So that delete is parked as
    ``_pending_supersede`` and only carried out once we know the module's save has landed: at the
    start of the next emission (module-emitted or not — the loop that owns the module has already
    returned from its ``store.save`` by then) or, failing that, when ``execute`` calls
    :meth:`finalize` after its own last receipt is persisted.
    """

    def __init__(self, bridge, plan: Plan, on_checkpoint: Callable | None) -> None:
        self._bridge, self._plan, self._cb = bridge, plan, on_checkpoint
        self._last_id: str | None = None
        self._last: Checkpoint | None = None
        self._pending_supersede: str | None = None

    def __call__(self, receipt) -> Checkpoint:
        self._flush_pending()
        module_emitted = isinstance(receipt, Checkpoint)
        checkpoint = receipt if module_emitted else create_checkpoint(self._plan, receipt, self._bridge.registry)
        if checkpoint != self._last:
            if hasattr(self._bridge, "events"):
                # test hook: FakeBridge records the ordering of proving/checkpoint/broadcast
                label = receipt.status.value if isinstance(receipt, Receipt) else "module"
                self._bridge.events.append((f"checkpoint:{label}", checkpoint.id))
            if self._cb is not None:
                self._cb(checkpoint)                     # the caller's own callback: errors are theirs
            if module_emitted:
                self._pending_supersede = self._last_id   # module saves this one itself, after we return
            else:
                _persist(self._bridge, checkpoint, receipt, previous_id=self._last_id)
            self._last_id, self._last = checkpoint.id, checkpoint
        return checkpoint

    def _flush_pending(self) -> None:
        if self._pending_supersede is None:
            return
        pending, self._pending_supersede = self._pending_supersede, None
        if pending == self._last_id:
            return
        store = getattr(self._bridge, "checkpoints", None)
        if store is not None:
            store.delete(pending)

    def finalize(self) -> None:
        """Drop any still-pending supersede. Call once execute()'s final receipt is persisted."""
        self._flush_pending()


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


def _connected_aleo_address(bridge) -> str | None:
    """``bridge.aleo_address()``, or None when no account is configured to sign with.

    A read-only facade cannot sign an Aleo leg anyway, so a missing account is not this check's
    problem to raise on — it just means there is nothing to compare the plan's sender against.
    """
    try:
        return bridge.aleo_address()
    except (ConfigurationError, AttributeError):
        return None


def _read_destination_balance(bridge, plan: Plan, resolved: ResolvedRoute) -> int | None:
    """The recipient's destination balance, or None when there is no reader for it.

    Only read when the destination connection IS the recipient (there is no per-address balance
    read in the module contracts); otherwise return None rather than baseline the wrong account.

    A transport failure is NOT swallowed here (Task 6 review item 8): ``get_status`` branch 6 uses
    this balance as the delivery SIGNAL, and a swallowed RPC error would read as "not delivered
    yet" forever instead of being retried by ``wait``'s transient classifier. The one caller that
    genuinely cannot afford to raise — ``execute``'s advisory pre-broadcast baseline — does the
    swallowing itself, in :func:`_delivery_verification`.
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


def _xreserve_withdrawal_fee_atomic(resolved: ResolvedRoute) -> int | None:
    """The route's ``withdrawalFeeAtomic`` literal, or None when it is missing or malformed."""
    raw = resolved.route.metadata.get("withdrawalFeeAtomic")
    return int(raw) if isinstance(raw, str) and raw.isdigit() else None


def _delivery_verification(bridge, plan: Plan, resolved: ResolvedRoute, *,
                           expected_atomic: int | None = None) -> dict[str, str]:
    """``execute``'s advisory delivery baseline — an unreadable balance is simply omitted.

    The best-effort swallow lives at THIS call site and not inside ``_read_destination_balance``
    (Task 6 review item 8): here the balance is a nice-to-have baseline written into a checkpoint
    before broadcast, so a flaky RPC must never block funds movement; in ``get_status`` branch 6
    the same read is the delivery signal and must raise.

    ``expected_atomic`` overrides how much the recipient should GAIN. It defaults to the full
    transferred amount (Hyperlane delivers the amount itself and charges its fees elsewhere); an
    xReserve burn passes the quote's ``amount_out`` instead, because the withdrawal fee is paid out
    of the burned amount. Getting that wrong in either direction is a correctness bug, not a
    rounding one: too high and the real delivery never satisfies the predicate, too low and
    unrelated inflow to the same address can satisfy it.
    """
    try:
        before = _read_destination_balance(bridge, plan, resolved)
    except Exception:                                              # noqa: BLE001 — advisory read only
        return {}
    if before is None:
        return {}
    expected = (expected_atomic if expected_atomic is not None
                else parse_decimal_amount(plan.amount, resolved.destination_asset.decimals))
    if expected <= 0:
        return {}                          # nothing to observe: no predicate could distinguish it
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
        receipt = _send_call(call, emit, poll_seconds, timeout_seconds)
        emit.finalize()
        return to_progress(plan, receipt)

    if plan.protocol == "hyperlane" and family == "solana":
        sol = _module(bridge, "sol")
        _assert_sender(plan, bridge.solana.address, family=family)
        call = sol.transfer_remote(plan=plan)
        receipt = _send_call(call, emit, poll_seconds, timeout_seconds)
        emit.finalize()
        return to_progress(plan, receipt)

    if plan.protocol == "hyperlane" and family == "aleo":
        _assert_sender(plan, _connected_aleo_address(bridge), family="aleo")
        as_signer = _aleo_hyperlane_mode(mode)
        verification = _delivery_verification(bridge, plan, resolved)
        gas = gas_payment_microcredits
        if gas is None:
            gas = bridge.hyperlane.quote_gas_payment(plan.source_asset_id).payment_microcredits
        call = bridge.hyperlane.transfer_remote(plan.source_asset_id, plan.recipient,
                                                amount_atomic=plan.amount_atomic, as_signer=as_signer,
                                                gas_payment_microcredits=gas)
        receipt = _run_aleo_leg(bridge, plan, call, proving=proving, emit=emit, extra_state=verification)
        emit.finalize()
        return to_progress(plan, receipt)

    if plan.protocol == "xreserve" and family == "evm":
        eth = _module(bridge, "eth")
        _assert_sender(plan, bridge.ethereum.address, family=family)
        nonce = _mint_secret(plan, secret_nonce)
        call = eth.deposit_usdc(plan=plan, secret_nonce=nonce)
        receipt = _send_call(call, emit, poll_seconds, timeout_seconds)
        emit.finalize()
        return to_progress(plan, receipt)

    if plan.protocol == "xreserve" and family == "aleo":
        _assert_sender(plan, _connected_aleo_address(bridge), family="aleo")
        burn_mode = _xreserve_burn_mode(mode)
        # Circle publishes no delivery query for this direction (see get_status branch 8), so this
        # advisory baseline is the only delivery signal the transfer will ever have — record it
        # before the burn is built, exactly as the Aleo Hyperlane leg does. What lands is the
        # quote's amount_out: the withdrawal fee comes OUT OF the burned amount, so a baseline on
        # the full amount would wait for a delivery that can never arrive. With no readable fee
        # there is no honest expectation to record, and the branch degrades to veil's passthrough.
        fee_atomic = _xreserve_withdrawal_fee_atomic(resolved)
        verification = ({} if fee_atomic is None else
                        _delivery_verification(bridge, plan, resolved,
                                               expected_atomic=plan.amount_atomic - fee_atomic))
        call = bridge.xreserve.burn(plan.recipient, amount_atomic=plan.amount_atomic, mode=burn_mode,
                                    record=record, merkle_proof=merkle_proof)
        receipt = _run_aleo_leg(bridge, plan, call, proving=proving, emit=emit, extra_state=verification)
        emit.finalize()
        return to_progress(plan, receipt)

    raise UnsupportedRouteError(f"Unsupported {plan.protocol} source chain family: {family} ({plan.route_id})")


# ── Aleo transaction status ───────────────────────────────────────────────────

def aleo_transaction_status(bridge, tx_id: str) -> tuple[str, str | None]:
    """``("accepted" | "rejected" | "pending", error)`` from the confirmed-transaction envelope.

    Reads ``GET /transaction/confirmed/{id}`` through the facade.  A 404
    (``TransactionNotFound``) means not confirmed yet → ``pending``.  The
    envelope's top-level ``status`` is the node's verdict; it carries no reason,
    so the error text is generic.
    """
    from aleo.facade.errors import TransactionNotFound
    try:
        confirmed = bridge.aleo.network.get_confirmed_transaction(tx_id)
    except TransactionNotFound:
        return "pending", None
    status = confirmed.get("status") if isinstance(confirmed, dict) else getattr(confirmed, "status", None)
    if status == "accepted":
        return "accepted", None
    if status == "rejected":
        return "rejected", f"Aleo transaction {tx_id} was rejected by the network"
    return "pending", None


_HEX = re.compile(r"^0x[0-9a-fA-F]*$")
_MESSAGE_ID_RE = re.compile(r"^0x[0-9a-fA-F]{64}$")


def _hex_bytes(value: Any, *, length: int | None = None) -> bytes | None:
    """Strict ``0x`` hex → bytes, or None when malformed / wrong length."""
    if not isinstance(value, str) or not _HEX.match(value) or len(value) % 2:
        return None
    data = bytes.fromhex(value[2:])
    if length is not None and len(data) != length:
        return None
    return data


def _check_receipt(plan: Plan, receipt: Receipt) -> None:
    if receipt.protocol != plan.protocol or receipt.protocol_state.get("routeId") != plan.route_id:
        raise CheckpointInvalidError("Bridge receipt does not match the prepared route")


def _clear_action(receipt: Receipt, **changes) -> Receipt:
    return receipt.replace(next_action=None, **changes)


def _message_id(receipt: Receipt) -> str | None:
    """``protocol_state["messageId"]`` first; else ``receipt.id`` when it is itself a message id.

    Solana and EVM Hyperlane receipts carry the id in ``protocol_state["messageId"]`` once
    known; a receipt that instead carries the message id AS its own ``id`` (some Aleo-origin
    shapes) falls back to that, but only when it is an exact 32-byte ``0x`` hex string — never a
    signature or an unrelated transaction hash of a different width. Never falls back when
    ``receipt.id == receipt.source_tx_id``: an EVM Hyperlane receipt whose DispatchId log was
    unreadable carries the source transaction hash as its id, which is a same-shaped ``0x`` hex
    string but is NOT a message id.
    """
    state = receipt.protocol_state
    message_id = state.get("messageId")
    if isinstance(message_id, str) and message_id:
        return message_id
    if (isinstance(receipt.id, str) and receipt.id != receipt.source_tx_id
            and _MESSAGE_ID_RE.fullmatch(receipt.id)):
        return receipt.id
    return None


# ── get_status ────────────────────────────────────────────────────────────────

def get_status(bridge, plan: Plan, receipt: Receipt) -> Receipt:
    """One refresh of the transfer's state — no polling, no signing.

    Ports veil's branch table in order: terminal receipts (``COMPLETED``/``FAILED``/``EXPIRED``)
    return untouched; EVM approvals and source confirmations delegate to the chain module (never
    called for any other status — both ``EthModule.source_status`` and ``SolModule.source_status``
    raise otherwise); Aleo source acceptance moves to ``DELIVERY_PENDING``; Hyperlane delivery is
    read from the destination Mailbox by message id (filling a missing Solana message id from the
    source transaction's logs first, never handing a signature to ``is_delivered``), or from the
    destination balance baseline for Aleo-origin routes; inbound xReserve reads the destination
    nullifier FIRST (invariant 6), then Circle's attestation (private mode stops at
    ``DESTINATION_ACTION_REQUIRED`` with ``next_action`` = ``{"kind": "xreserve-private-mint",
    "chainId": ...}``), then the private mint's acceptance.  Returns the SAME object when nothing
    changed.

    One branch deliberately goes BEYOND veil: an Aleo→EVM xReserve burn has no canonical delivery
    query at all (Circle publishes none for that direction), so veil leaves it ``DELIVERY_PENDING``
    forever and ``wait`` could never terminate it. Here, when ``execute`` was able to baseline the
    recipient's destination balance, the same arithmetic the Aleo-origin Hyperlane branch uses ends
    the transfer — against the amount that actually lands, which for xReserve is the quote's
    ``amount_out`` (the withdrawal fee is paid out of the burned amount), not the burned amount
    itself. Without that baseline — or without a connection that can read it — the branch
    degrades to veil's passthrough rather than raising: the baseline is optional, so its absence is
    "not observable from here", not an error.

    A Solana transport failure inside ``SolModule.source_status`` (or the message-id log read) is
    not swallowed here: a single refresh may raise on a flaky public RPC, and retrying with backoff
    is ``wait``'s job, not this function's.
    """
    resolved = resolve_route(bridge.registry, plan)
    _check_receipt(plan, receipt)
    if receipt.status in TERMINAL:
        return receipt
    route, src, dst = resolved.route, resolved.source_chain, resolved.destination_chain
    state = receipt.protocol_state

    # 1. EVM approval → wallet boundary
    if receipt.status is Status.SOURCE_APPROVAL_PENDING and src.family == "evm":
        return _module(bridge, "eth").source_status(plan, receipt)

    # 2. Aleo source acceptance is the irreversible boundary
    if receipt.status is Status.SOURCE_CONFIRMING and src.family == "aleo":
        if not receipt.source_tx_id:
            raise CheckpointInvalidError("Bridge receipt is missing its Aleo source transaction id")
        verdict, error = aleo_transaction_status(bridge, receipt.source_tx_id)
        if verdict == "accepted":
            return _clear_action(receipt, status=Status.DELIVERY_PENDING)
        if verdict == "rejected":
            return _clear_action(receipt, status=Status.FAILED,
                                 protocol_state={**state, "sourceError": error})
        return receipt

    # 3/4. Hyperlane source confirmation on EVM / Solana (extracts messageId)
    if receipt.status is Status.SOURCE_CONFIRMING and route.protocol == "hyperlane":
        if src.family == "evm":
            return _module(bridge, "eth").source_status(plan, receipt)
        if src.family == "solana":
            return _module(bridge, "sol").source_status(plan, receipt)

    # 5. Hyperlane delivery: the destination Mailbox is canonical
    message_id = _message_id(receipt)
    if (message_id is None and receipt.status is Status.DELIVERY_PENDING and route.protocol == "hyperlane"
            and src.family == "solana" and state.get("messageIdUnavailable") and receipt.source_tx_id):
        sol = _module(bridge, "sol")                                  # missing connection must raise, not be swallowed
        try:
            logs = sol._transaction_logs(receipt.source_tx_id)
        except Exception:                                             # noqa: BLE001 — advisory fill-in only
            logs = None
        filled = None if logs is None else _sealevel.extract_hyperlane_message_id(logs)
        if filled is not None:
            new_state = {k: v for k, v in state.items() if k != "messageIdUnavailable"}
            new_state["messageId"] = filled
            receipt = receipt.replace(id=filled, protocol_state=new_state)
            state, message_id = new_state, filled
        # else: still unavailable — fall through unchanged; never hand the signature to is_delivered

    if (receipt.status is Status.DELIVERY_PENDING and route.protocol == "hyperlane"
            and message_id is not None and dst.family in ("aleo", "evm")):
        delivered = (bridge.hyperlane.is_delivered(message_id) if dst.family == "aleo"
                     else _module(bridge, "eth").is_delivered(message_id))
        return _clear_action(receipt, status=Status.COMPLETED) if delivered else receipt

    # 6. Aleo-origin Hyperlane without a message id: destination balance baseline
    if receipt.status is Status.DELIVERY_PENDING and route.protocol == "hyperlane" and src.family == "aleo":
        before, expected = state.get("destinationBalanceBeforeAtomic"), state.get("expectedDestinationIncreaseAtomic")
        if not (isinstance(before, str) and before.isdigit() and isinstance(expected, str) and expected.isdigit()):
            return receipt
        current = _read_destination_balance(bridge, plan, resolved)
        if current is None:
            raise DeliveryUnknownError(
                f"No destination balance reader is configured for {dst.id}: bind the {dst.id} connection "
                "whose address is the recipient, or confirm delivery out of band")
        if current < int(before) + int(expected):
            return receipt
        return _clear_action(receipt, status=Status.COMPLETED)

    # 7. Other Hyperlane states are observed elsewhere
    if route.protocol == "hyperlane":
        return receipt

    # 8. xReserve Aleo→EVM: Circle exposes no canonical delivery query, so fall back to the baseline
    #    execute recorded. This goes DELIBERATELY BEYOND veil, which has no delivery query for this
    #    direction at all and therefore leaves the transfer DELIVERY_PENDING forever: without a
    #    signal here `wait` can never terminate and the checkpoint is never freed. Unlike branch 6
    #    a missing reader is NOT an error — the baseline is optional, so we degrade to veil's
    #    passthrough (the caller confirms delivery out of band) instead of raising.
    if (receipt.status is Status.DELIVERY_PENDING and route.protocol == "xreserve"
            and src.family == "aleo" and dst.family == "evm"):
        before, expected = state.get("destinationBalanceBeforeAtomic"), state.get("expectedDestinationIncreaseAtomic")
        if not (isinstance(before, str) and before.isdigit() and isinstance(expected, str) and expected.isdigit()):
            return receipt
        current = _read_destination_balance(bridge, plan, resolved)
        if current is None or current < int(before) + int(expected):
            return receipt
        return _clear_action(receipt, status=Status.COMPLETED)

    # 9. Everything else that is not inbound xReserve
    if route.protocol != "xreserve" or src.family != "evm" or dst.family != "aleo":
        raise UnsupportedRouteError("Status refresh is not implemented for this bridge route")

    # 10. xReserve EVM→Aleo — destination nullifier first (invariant 6)
    if receipt.status in (Status.ATTESTATION_PENDING, Status.DELIVERY_PENDING, Status.DESTINATION_ACTION_REQUIRED):
        nonce = state.get("nonce")
        if not isinstance(nonce, str):
            payload = _hex_bytes(state.get("payload"))
            if payload is not None:
                from .encoding import xreserve_nonce_from_payload
                nonce = "0x" + xreserve_nonce_from_payload(payload).hex()
        if isinstance(nonce, str) and nonce and bridge.xreserve.is_delivered(nonce, route=route):
            return _clear_action(receipt, status=Status.COMPLETED)

    if receipt.status is Status.SOURCE_CONFIRMING:
        return _module(bridge, "eth").source_status(plan, receipt)

    if receipt.status is Status.ATTESTATION_PENDING:
        message_hash = state.get("messageHash")
        if _hex_bytes(message_hash, length=32) is None:
            raise CheckpointInvalidError("xReserve receipt is missing its Circle message hash")
        attestation = bridge.xreserve.get_attestation(message_hash, route=route)
        if attestation is None:
            return receipt
        with_att = {**state, "attestation": "0x" + attestation.attestation.hex()}
        if plan.mint_mode != "private":
            return receipt.replace(status=Status.DELIVERY_PENDING, protocol_state=with_att)
        return receipt.replace(status=Status.DESTINATION_ACTION_REQUIRED, protocol_state=with_att,
                               next_action={"kind": "xreserve-private-mint", "chainId": dst.id})

    if receipt.status is Status.DESTINATION_ACTION_REQUIRED:
        return receipt

    if receipt.status is Status.DESTINATION_CONFIRMING:
        if not receipt.destination_tx_id:
            raise CheckpointInvalidError("xReserve receipt is missing its Aleo destination transaction id")
        verdict, error = aleo_transaction_status(bridge, receipt.destination_tx_id)
        if verdict == "accepted":
            return _clear_action(receipt, status=Status.COMPLETED)
        if verdict == "rejected":
            return _clear_action(receipt, status=Status.FAILED,
                                 protocol_state={**state, "destinationError": error})
        return receipt

    return receipt


# ── wait ──────────────────────────────────────────────────────────────────────

def _track(bridge, plan: Plan, previous: Receipt, current: Receipt) -> None:
    """Mirror an observed status change into the bound store (no caller callback)."""
    if getattr(bridge, "checkpoints", None) is None:
        return
    _persist(bridge, create_checkpoint(plan, current, bridge.registry), current, previous_id=previous.id)


def _status_set(values) -> set[Status]:
    try:
        return {v if isinstance(v, Status) else Status(v) for v in values}
    except ValueError as exc:
        raise ConfigurationError(f"wait(until=...) contains an unknown status: {exc}") from exc


_TRANSIENT_BRIDGE_ERROR_RE = re.compile(r"HTTP status (429|5\d\d)|request failed:")


def _is_transient_error(exc: Exception) -> bool:
    """``wait``'s retry classifier: a network/RPC hiccup vs. a real problem that must propagate.

    Transient: ``requests.RequestException`` (any ``requests``-based transport), ``aleo.facade.
    errors.AleoNetworkError`` (the Aleo facade), a ``BridgeError`` whose message matches an HTTP
    429/5xx or a wrapped "request failed:" transport error (``SolanaRpcClient``), or a web3
    provider/connection error. Everything else — including programming errors, ``BridgeError``s
    about an actual on-chain failure, ``CheckpointInvalidError``, ``RouteUnavailableError`` —
    propagates immediately.
    """
    try:
        import requests
        if isinstance(exc, requests.RequestException):
            return True
    except ImportError:
        pass
    try:
        from aleo.facade.errors import AleoNetworkError
        if isinstance(exc, AleoNetworkError):
            return True
    except ImportError:
        pass
    try:
        from web3.exceptions import ProviderConnectionError
        if isinstance(exc, ProviderConnectionError):
            return True
    except ImportError:
        if type(exc).__name__ == "ProviderConnectionError":     # web3 extra not installed here
            return True
    if isinstance(exc, BridgeError) and _TRANSIENT_BRIDGE_ERROR_RE.search(str(exc)):
        return True
    return False


def wait(bridge, progress: Progress, *, until=None, poll_seconds: float = 15.0,
         timeout_seconds: float = 1200.0, on_update: Callable[[Progress], Any] | None = None,
         on_error: Callable[[Exception], Any] | None = None, max_consecutive_errors: int = 5) -> Progress:
    """Poll ``get_status`` until the transfer needs the caller or finishes.

    Always stops at the caller boundaries — ``SOURCE_SUBMISSION_PENDING`` (→ ``resume``),
    ``DESTINATION_ACTION_REQUIRED`` (→ ``complete``), ``COMPLETED``, ``FAILED``, ``EXPIRED`` — plus
    any statuses in ``until`` (a ``Status`` or its name; ``until=[]`` is a ``ConfigurationError``,
    an unknown name too). Returns immediately when ``progress.next != "wait"`` or the receipt is
    already at a stop. ``on_update`` fires only when the receipt changed, never on a retry.
    ``poll_seconds`` is floored at 0.1 unless exactly 0; negative ``poll_seconds``/``timeout_seconds``
    is a ``ConfigurationError``.

    A ``get_status`` call that raises a transient error (flaky RPC/HTTP transport — see
    :func:`_is_transient_error`) is retried with the normal poll interval, up to
    ``max_consecutive_errors`` (default 5) consecutive failures before the last one is re-raised;
    ``on_error`` fires on each tolerated retry so callers can log them. A non-transient error
    propagates immediately, on the first attempt.

    Hitting ``timeout_seconds`` raises ``PollingTimeoutError`` carrying the last ``status`` and
    ``progress`` — a timeout is NOT a failure (invariant 5): the transfer is still in flight; call
    ``wait`` again or ``recover`` later.
    """
    if until is not None and len(until) == 0:
        raise ConfigurationError("wait(until=[]) has nothing to stop at: pass at least one Status or omit until")
    plan, receipt = progress.plan, progress.receipt
    resolve_route(bridge.registry, plan)
    _check_receipt(plan, receipt)
    stops = set(CALLER_BOUNDARIES) | _status_set(until or ())
    current = to_progress(plan, receipt)
    if current.next != "wait" or receipt.status in stops:
        return current
    if poll_seconds < 0 or timeout_seconds < 0:
        raise ConfigurationError("poll_seconds and timeout_seconds must be non-negative")
    interval = 0.0 if poll_seconds == 0 else max(0.1, float(poll_seconds))
    deadline = time.monotonic() + timeout_seconds
    updated = receipt
    consecutive_errors = 0
    while True:
        try:
            nxt = get_status(bridge, plan, updated)
        except Exception as exc:
            if not _is_transient_error(exc):
                raise
            consecutive_errors += 1
            if consecutive_errors > max_consecutive_errors:
                raise
            if on_error is not None:
                on_error(exc)
            if time.monotonic() >= deadline:
                raise PollingTimeoutError(
                    f"Bridge status polling timed out in state {updated.status.value}; the transfer is "
                    "still in flight — call wait() again or recover() from the last checkpoint. This is "
                    "not a failure.", status=updated.status, progress=to_progress(plan, updated)) from exc
            time.sleep(interval)
            continue
        consecutive_errors = 0
        if nxt != updated:
            _track(bridge, plan, updated, nxt)
            if on_update is not None:
                on_update(to_progress(plan, nxt))
        updated = nxt
        if updated.status in stops:
            return to_progress(plan, updated)
        if time.monotonic() >= deadline:
            raise PollingTimeoutError(
                f"Bridge status polling timed out in state {updated.status.value}; the transfer is still "
                "in flight — call wait() again or recover() from the last checkpoint. This is not a failure.",
                status=updated.status, progress=to_progress(plan, updated))
        time.sleep(interval)


__all__ = ["MINT_MODES", "ResolvedRoute", "aleo_transaction_status", "complete", "execute", "get_status",
           "is_duplicate_broadcast_error", "prepare", "progress_from_checkpoint", "quote", "recover",
           "resolve_route", "resume", "submit_serialized", "wait"]


# ── recover ───────────────────────────────────────────────────────────────────

def _coerce_checkpoint(value):
    if isinstance(value, Checkpoint):
        return value
    if isinstance(value, str):
        return Checkpoint.from_json(value)
    if isinstance(value, dict):
        return Checkpoint.from_dict(value)
    raise CheckpointInvalidError(f"recover() takes a Checkpoint, its dict, or its JSON; got {type(value).__name__}")


def _plan_from_intent(registry: Registry, intent: dict[str, Any]) -> Plan:
    """Rebuild the ``Plan`` behind a checkpoint by re-running ``prepare`` on its intent.

    ``prepare`` is only ever a thin validating wrapper around ``_plan.build_plan``
    (proven field-identical for every active route — see
    ``tests/test_prepare.py::test_prepare_equals_build_plan_for_every_active_route``), so a
    recovered plan is exactly what the original ``prepare()``/``quote()`` call produced and
    passes ``EthModule``/``SolModule``'s field-by-field ``plan=`` checks.
    """
    try:
        return prepare(registry,
                       source=(intent["source"]["chain"], intent["source"]["asset"]),
                       destination=(intent["destination"]["chain"], intent["destination"]["asset"]),
                       amount=intent["amount"], recipient=intent["recipient"], sender=intent.get("sender"),
                       protocol=intent.get("bridgeProtocol"), mint_mode=intent.get("mintMode", "public"))
    except (KeyError, TypeError) as exc:
        raise CheckpointInvalidError(f"Bridge checkpoint intent is incomplete: missing {exc}") from exc


def _assert_prepared_id(serialized: Any, expected_id: str, what: str = "prepared Aleo transaction") -> str:
    """The serialized transaction's ``id`` must equal the saved id — never substitute bytes."""
    if not isinstance(serialized, str) or not serialized:
        raise CheckpointInvalidError(f"Bridge checkpoint contains an invalid {what} (empty)")
    try:
        decoded = json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise CheckpointInvalidError(f"Bridge checkpoint contains an invalid {what}: not JSON") from exc
    tx_id = decoded.get("id") if isinstance(decoded, dict) else None
    if not isinstance(tx_id, str) or tx_id != expected_id:
        raise CheckpointInvalidError(f"Bridge checkpoint {what} id does not match its payload "
                                     f"({tx_id!r} != {expected_id!r})")
    return tx_id


def _finish(bridge, plan: Plan, receipt: Receipt, checkpoint_id: str) -> Progress:
    """Reduce *receipt* to ``Progress``, dropping the checkpoint (keyed on ``checkpoint_id`` —
    the checkpoint's OWN id, never ``receipt.id``: for Solana and EVM Hyperlane the checkpoint's
    id is the source transaction id while the receipt's own id flips to the message id once
    confirmed) from the bound store once the transfer reaches a terminal status.
    """
    store = getattr(bridge, "checkpoints", None)
    if store is not None and receipt.status in TERMINAL:
        store.delete(checkpoint_id)
    return to_progress(plan, receipt)


def _verification_from_delivery(dv: dict[str, Any]) -> dict[str, str]:
    """Validate and translate a checkpoint's ``deliveryVerification`` block (Aleo-origin Hyperlane
    only); ``{}`` when absent. Raises :class:`CheckpointInvalidError` on a malformed block (missing
    key or non-digit value) rather than ``KeyError`` — the block came from a stored file, not a
    live read, so it is untrusted input."""
    if not dv:
        return {}
    before, expected = dv.get("balanceBeforeAtomic"), dv.get("expectedIncreaseAtomic")
    if not (isinstance(before, str) and before.isdigit() and isinstance(expected, str) and expected.isdigit()):
        raise CheckpointInvalidError("Bridge checkpoint contains invalid destination balance verification state")
    return {"destinationBalanceBeforeAtomic": before, "expectedDestinationIncreaseAtomic": expected}


def _reconstruct_source_receipt(plan: Plan, resolved: ResolvedRoute, cp: Checkpoint,
                                verification: dict[str, str]) -> Receipt:
    """Build the pre-refresh source ``Receipt`` purely from a checkpoint's stored fields — no
    network, no signing. Shared by ``recover`` (which then reads live chain state to refine
    non-``SOURCE_SUBMISSION_PENDING`` results) and the fully offline ``progress_from_checkpoint``
    (which stops here), so the two can never drift apart on what a checkpoint alone can tell you.

    Mirrors ``create_checkpoint``'s own allowlist: a ``preparedTransaction`` with no
    ``transactionId`` is an unbroadcast Aleo leg (``SOURCE_SUBMISSION_PENDING`` — next: resume);
    a ``transactionId`` alone (any chain family) is a submitted, still-confirming leg
    (``SOURCE_CONFIRMING``); an EVM leg with only approvals and no ``transactionId`` yet is
    waiting on its deposit/dispatch (``SOURCE_SUBMISSION_PENDING`` too — the same "call resume()"
    signal; ``resume()`` re-verifies against live chain history before repeating anything, so an
    offline guess here is never unsafe); anything else has nothing to build a receipt from.
    """
    src = resolved.source_chain
    source = cp.source or {}
    approvals = [a for a in (source.get("approvalTransactionIds") or []) if isinstance(a, str)]

    if src.family == "aleo":
        prepared = source.get("preparedTransaction")
        if prepared and not source.get("transactionId"):
            if cp.destination or approvals:
                raise CheckpointInvalidError(
                    "Bridge checkpoint contains transactions that are invalid for a prepared Aleo source route")
            tx_id = _assert_prepared_id(prepared.get("serializedTransaction"), str(prepared.get("transactionId")))
            return Receipt(id=tx_id, protocol=plan.protocol, status=Status.SOURCE_SUBMISSION_PENDING,
                          protocol_state={"routeId": plan.route_id, "preparedTransaction": prepared["serializedTransaction"],
                                          **verification})
        tx_id = source.get("transactionId")
        if not tx_id:
            raise CheckpointInvalidError("Bridge checkpoint contains no submitted source transaction")
        if cp.destination or approvals:
            raise CheckpointInvalidError(
                "Bridge checkpoint contains transactions that are invalid for an Aleo source route")
        return Receipt(id=tx_id, protocol=plan.protocol, status=Status.SOURCE_CONFIRMING, source_tx_id=tx_id,
                      protocol_state={"routeId": plan.route_id, **verification})

    if src.family == "solana":
        tx_id = source.get("transactionId")
        if not tx_id:
            raise CheckpointInvalidError("Bridge checkpoint contains no submitted source transaction")
        if cp.destination or approvals:
            raise CheckpointInvalidError(
                "Bridge checkpoint contains transactions that are invalid for a Solana source route")
        blockhash, last_valid = source.get("blockhash"), source.get("lastValidBlockHeight")
        if (blockhash is not None or last_valid is not None) and (
                not isinstance(blockhash, str) or not blockhash
                or not isinstance(last_valid, str) or not last_valid.isdigit()):
            raise CheckpointInvalidError("Bridge checkpoint contains an invalid Solana blockhash lifetime")
        state: dict[str, Any] = {"routeId": plan.route_id}
        if isinstance(blockhash, str) and isinstance(last_valid, str):
            state.update(blockhash=blockhash, lastValidBlockHeight=last_valid)
        return Receipt(id=tx_id, protocol=plan.protocol, status=Status.SOURCE_CONFIRMING, source_tx_id=tx_id,
                      protocol_state=state)

    if src.family == "evm":
        tx_id = source.get("transactionId")
        if tx_id:
            state: dict[str, Any] = {"routeId": plan.route_id}
            if approvals:
                state["approvalTxIds"] = approvals
            return Receipt(id=tx_id, protocol=plan.protocol, status=Status.SOURCE_CONFIRMING, source_tx_id=tx_id,
                          protocol_state=state)
        if approvals:
            return Receipt(id=approvals[-1], protocol=plan.protocol, status=Status.SOURCE_SUBMISSION_PENDING,
                          protocol_state={"routeId": plan.route_id, "approvalTxIds": approvals})
        raise CheckpointInvalidError("Bridge checkpoint contains no submitted source transaction")

    raise UnsupportedRouteError(f"Bridge checkpoint recovery is not implemented for source family {src.family!r}")


def _apply_destination_overlay(resolved: ResolvedRoute, cp: Checkpoint, receipt: Receipt) -> Receipt:
    """Offline-only: fold a checkpoint's own destination fields (inbound xReserve) into *receipt*
    without any network read — a submitted destination transaction id becomes
    ``DESTINATION_CONFIRMING``, an unbroadcast prepared one becomes ``DESTINATION_ACTION_REQUIRED``
    (ready for ``complete()``). Never validates either against live chain state the way ``recover``
    does over the wire (that would need the Circle attestation, which a checkpoint never stores);
    ``complete()``/``resume()`` re-verify before acting, so an offline-optimistic guess here is
    never unsafe — only ever a prompt to call the verb that actually checks.
    """
    destination = cp.destination or {}
    if not destination:
        return receipt
    if resolved.route.protocol != "xreserve" or resolved.destination_chain.family != "aleo":
        raise CheckpointInvalidError(
            "Bridge checkpoint contains a destination transaction that is invalid for this route")
    prepared_dest = destination.get("preparedTransaction")
    if prepared_dest and destination.get("transactionId"):
        raise CheckpointInvalidError(
            "Bridge checkpoint cannot contain both prepared and submitted destination transactions")
    if destination.get("transactionId"):
        receipt = receipt.replace(status=Status.DESTINATION_CONFIRMING, destination_tx_id=destination["transactionId"])
    if prepared_dest:
        tx_id = _assert_prepared_id(prepared_dest.get("serializedTransaction"), str(prepared_dest.get("transactionId")),
                                    "prepared Aleo destination transaction")
        receipt = receipt.replace(id=tx_id, status=Status.DESTINATION_ACTION_REQUIRED,
                                  next_action={"kind": "xreserve-private-mint", "chainId": resolved.destination_chain.id},
                                  protocol_state={**receipt.protocol_state,
                                                  "preparedDestinationTransaction": prepared_dest["serializedTransaction"]})
    return receipt


def progress_from_checkpoint(registry: Registry, checkpoint) -> Progress:
    """Pure, fully offline reconstruction of a checkpoint's ``Progress`` — no network, no signing.

    Used by ``Bridge.pending()`` instead of ``recover`` so that listing every in-flight transfer
    never depends on any chain being reachable (one unreachable RPC must never hide every other
    transfer). Rebuilds the ``Plan`` from the checkpoint's own intent and validates it against the
    live registry exactly like ``recover`` (still raises on a bad format, version, or route
    mismatch — those mean the record cannot be interpreted at all). From there, everything is
    built purely from the checkpoint's stored fields (:func:`_reconstruct_source_receipt` /
    :func:`_apply_destination_overlay`) — the same reconstruction ``recover`` performs before its
    own live refresh. A checkpoint whose stored fields cannot be interpreted after that point
    (e.g. no submitted or prepared source transaction at all) folds into a ``Progress`` with
    ``next == "failed"`` and ``error`` set, instead of raising — so one malformed record can never
    hide the others in a ``pending()`` listing.
    """
    cp = _coerce_checkpoint(checkpoint)
    if cp.version != 1 or not cp.intent or not cp.route:
        raise CheckpointInvalidError("Bridge checkpoint format is invalid or unsupported (version 1 required)")
    plan = _plan_from_intent(registry, cp.intent)
    if cp.route.get("registryVersion") != plan.registry_version:
        raise RegistryVersionMismatchError(
            f"Checkpoint was written against registry {cp.route.get('registryVersion')}; this client has "
            f"{plan.registry_version}. Upgrade/downgrade aleo-bridge-sdk to the version that wrote it.")
    if cp.route.get("id") != plan.route_id:
        raise CheckpointInvalidError(
            f"Bridge checkpoint route {cp.route.get('id')} does not match the prepared route {plan.route_id}")
    resolved = resolve_route(registry, plan)
    try:
        verification = _verification_from_delivery(cp.delivery_verification or {})
        receipt = _reconstruct_source_receipt(plan, resolved, cp, verification)
        receipt = _apply_destination_overlay(resolved, cp, receipt)
    except BridgeError as exc:
        receipt = Receipt(id=cp.id, protocol=plan.protocol, status=Status.FAILED,
                          protocol_state={"routeId": plan.route_id, "sourceError": str(exc)})
    return to_progress(plan, receipt)


def recover(bridge, checkpoint) -> Progress:
    """Rebuild a transfer's ``Progress`` from a saved checkpoint — reads only, never signs.

    Accepts a ``Checkpoint``, its dict, or its JSON.  Re-runs ``prepare`` on the
    saved intent against the LIVE registry, then checks the route id and registry
    version (``CheckpointInvalidError`` / ``RegistryVersionMismatchError``).
    Aleo source: a proved-but-unbroadcast transaction yields ``next == "resume"``
    with no network read; a submitted one gets exactly one ``get_status`` from
    ``SOURCE_CONFIRMING``.  Solana: validates the blockhash pair and reads the
    signature status.  EVM: delegates to ``bridge.eth.recover_source`` (log
    scan); inbound xReserve additionally restores a submitted or prepared
    private mint.  The result's ``next`` tells the caller what to do.
    """
    cp = _coerce_checkpoint(checkpoint)
    if cp.version != 1 or not cp.intent or not cp.route:
        raise CheckpointInvalidError("Bridge checkpoint format is invalid or unsupported (version 1 required)")
    plan = _plan_from_intent(bridge.registry, cp.intent)
    if cp.route.get("registryVersion") != plan.registry_version:
        raise RegistryVersionMismatchError(
            f"Checkpoint was written against registry {cp.route.get('registryVersion')}; this client has "
            f"{plan.registry_version}. Upgrade/downgrade aleo-bridge-sdk to the version that wrote it.")
    if cp.route.get("id") != plan.route_id:
        raise CheckpointInvalidError(
            f"Bridge checkpoint route {cp.route.get('id')} does not match the prepared route {plan.route_id}")
    resolved = resolve_route(bridge.registry, plan)
    src, dst = resolved.source_chain, resolved.destination_chain
    verification = _verification_from_delivery(cp.delivery_verification or {})

    if src.family in ("aleo", "solana"):
        receipt = _reconstruct_source_receipt(plan, resolved, cp, verification)
        if receipt.status is Status.SOURCE_SUBMISSION_PENDING:
            return to_progress(plan, receipt)          # Aleo prepared, unbroadcast: no network read
        receipt = get_status(bridge, plan, receipt)
        return _finish(bridge, plan, receipt, cp.id)

    if resolved.route.protocol == "hyperlane" and src.family == "evm":
        if cp.destination:
            raise CheckpointInvalidError(
                "Bridge checkpoint contains a destination transaction that is invalid for this Hyperlane route")
        receipt = _module(bridge, "eth").recover_source(plan, cp, required=False)
        return _finish(bridge, plan, receipt, cp.id)

    if resolved.route.protocol != "xreserve" or src.family != "evm" or dst.family != "aleo":
        raise UnsupportedRouteError("Bridge checkpoint recovery is not implemented for this route")

    receipt = _module(bridge, "eth").recover_source(plan, cp, required=False)
    destination = cp.destination or {}
    prepared_dest = destination.get("preparedTransaction")
    if prepared_dest and destination.get("transactionId"):
        raise CheckpointInvalidError(
            "Bridge checkpoint cannot contain both prepared and submitted destination transactions")
    if prepared_dest:
        _assert_prepared_id(prepared_dest.get("serializedTransaction"), str(prepared_dest.get("transactionId")),
                            "prepared Aleo destination transaction")
    if destination.get("transactionId"):
        receipt = receipt.replace(status=Status.DESTINATION_CONFIRMING, destination_tx_id=destination["transactionId"])
    if receipt.status in (Status.ATTESTATION_PENDING, Status.DESTINATION_CONFIRMING):
        receipt = get_status(bridge, plan, receipt)
    if prepared_dest:
        if receipt.status is not Status.DESTINATION_ACTION_REQUIRED:
            raise CheckpointInvalidError(
                "Prepared destination transaction is no longer valid for the recovered bridge state")
        receipt = receipt.replace(id=str(prepared_dest["transactionId"]),
                                  protocol_state={**receipt.protocol_state,
                                                  "preparedDestinationTransaction": prepared_dest["serializedTransaction"]})
    return _finish(bridge, plan, receipt, cp.id)


# ── Idempotent Aleo rebroadcast (invariant 3) ─────────────────────────────────

#: Plan 1's duplicate-broadcast classifier, re-exported under the lifecycle's own name.
#:
#: It is deliberately NOT reimplemented here: ``AleoCall.submit_prepared`` already applies exactly
#: this rule to every prepared broadcast, and two rules that could ever disagree about "is this a
#: duplicate?" is one rule too many for a funds-critical path. Only the node's *"already exists"*
#: answer (ledger or mempool) means "this exact transaction is already known"; ``duplicate serial
#: number`` / ``duplicate output id`` mean a DIFFERENT transaction collided with this one's records
#: and must surface as failures — and they do, because they never say "already exists".
is_duplicate_broadcast_error = is_duplicate_submission


def submit_serialized(bridge, serialized: str, expected_id: str) -> str:
    """Broadcast an already-proved transaction; a duplicate answer is success, not a failure.

    Returns the transaction id the node acknowledged, which must be *expected_id* — the id of the
    exact bytes that were broadcast. A node that answers with a different id has accepted something
    this transfer never checkpointed, so it is refused rather than recorded as its transaction.
    """
    try:
        submitted = str(bridge.aleo.network.submit_transaction(serialized)).strip().strip('"')
    except Exception as exc:  # noqa: BLE001 — the node's error type varies by transport
        if is_duplicate_broadcast_error(exc):
            return expected_id            # the earlier broadcast won the race: nothing left to do
        raise
    if submitted != expected_id:
        raise CheckpointInvalidError(
            f"Aleo node acknowledged transaction {submitted}; expected {expected_id}. The prepared "
            "bytes and the node's answer disagree — do not resend; inspect both ids first.")
    return submitted


# ── resume ────────────────────────────────────────────────────────────────────

def resume(bridge, progress: Progress, *, on_checkpoint: Callable | None = None,
           secret_nonce: str | None = None, poll_seconds: float = 1.0, timeout_seconds: float = 120.0,
           proving: str = "delegate") -> Progress:
    """Finish the source leg an interruption left unsubmitted — never repeats an irreversible step.

    Requires ``progress.next == "resume"`` (status ``SOURCE_SUBMISSION_PENDING``); anything else is
    a :class:`~aleo_bridge.errors.NotResumableError` pointing at ``wait``/``recover``. In particular
    a ``SOURCE_APPROVAL_PENDING`` receipt is NOT resumable directly: call ``recover`` first, which
    observes the approval and yields the ``SOURCE_SUBMISSION_PENDING`` progress this verb takes.

    Aleo source: rebroadcasts the checkpointed transaction byte-for-byte, after checking that the
    serialized payload's own id matches the saved one — a duplicate-transaction answer means the
    first broadcast won the race and counts as success. The bytes are then dropped from the
    receipt. Nothing is re-proved, so the transfer can only ever exist once on chain.

    EVM source: re-scans source history from the confirmed approval
    (``bridge.eth.recover_source(plan, checkpoint, required=True)``) and, only when that scan proves
    no deposit/dispatch exists yet, re-quotes and authorizes the single remaining transaction
    through the module's own ``plan=`` surface. Two guards ported from veil refuse rather than
    guess: the re-quoted hook data must equal the hook the checkpointed approval committed to (so a
    private mint can never be re-hooked to a commitment its recipient cannot open), and the
    allowance must still cover the deposit (a vanished allowance means something else spent it, and
    re-approving is a second irreversible step ``resume`` does not own). A confirmed approval is
    never repeated; the ids already recorded are carried into the new receipt.

    Solana source: ``SolCall`` has no approval step and no pre-broadcast state to continue, so there
    is nothing to resume — ``recover``/``wait`` observe the signature instead.

    ``secret_nonce`` is mandatory when ``plan.mint_mode == "private"``, and is checked before any
    RPC: the SDK never stored it, and a silent ``"0scalar"`` fallback would commit the deposit to a
    hook nobody can open. ``proving`` is accepted for symmetry with ``execute``/``complete`` and is
    never used — no resume path ever proves anything: an Aleo leg rebroadcasts bytes that were
    already proved, and an EVM leg has no proofs at all.
    """
    plan, receipt = progress.plan, progress.receipt
    if progress.next != "resume" or receipt.status is not Status.SOURCE_SUBMISSION_PENDING:
        raise NotResumableError(
            "Bridge progress has no source submission to resume (next must be 'resume' at "
            "SOURCE_SUBMISSION_PENDING); call wait() or recover() to refresh it")
    resolved = resolve_route(bridge.registry, plan)
    _require_active(resolved.route)
    _check_receipt(plan, receipt)
    emit = _Emitter(bridge, plan, on_checkpoint)
    state = receipt.protocol_state
    family = resolved.source_chain.family

    if family == "aleo":
        serialized = state.get("preparedTransaction")
        if not isinstance(serialized, str) or not serialized:
            raise NotResumableError(
                "Prepared Aleo transfer is missing its serialized transaction: resume() rebroadcasts "
                "the exact proved bytes and never re-proves. Recover from the checkpoint written "
                "between proving and broadcast, or start the transfer over if none exists.")
        tx_id = _assert_prepared_id(serialized, receipt.id)
        submit_serialized(bridge, serialized, tx_id)
        new_state: dict[str, Any] = {"routeId": plan.route_id}
        for key in ("destinationBalanceBeforeAtomic", "expectedDestinationIncreaseAtomic"):
            if isinstance(state.get(key), str):
                new_state[key] = state[key]
        submitted = Receipt(id=tx_id, protocol=plan.protocol, status=Status.SOURCE_CONFIRMING,
                            source_tx_id=tx_id, protocol_state=new_state)
        emit(submitted)
        emit.finalize()
        return to_progress(plan, submitted)

    if family == "solana":
        raise NotResumableError(
            "Solana source legs have no resumable state: the transfer is signed and broadcast in one "
            "step, so nothing is ever left to submit. Call recover() or wait() to observe the "
            "signature instead.")

    if family != "evm":
        raise UnsupportedRouteError(f"Source resumption is not implemented for {resolved.source_chain.id}")

    eth = _module(bridge, "eth")
    _assert_sender(plan, bridge.ethereum.address, family="evm")
    is_xreserve = resolved.route.protocol == "xreserve"
    nonce = _mint_secret(plan, secret_nonce) if is_xreserve else None      # before any RPC
    saved_hook = state.get("hookData")
    if is_xreserve and _hex_bytes(saved_hook, length=HOOK_DATA_BYTES) is None:
        # Without the hook the approval committed to there is nothing to compare the re-quote
        # against, so the guard below would silently pass and the deposit could be re-hooked to a
        # different commitment. Refuse here, before any RPC, rather than resume half-blind.
        raise NotResumableError(
            "This transfer's checkpoint carries no xReserve hook data (a 65-byte 0x hex string); "
            "recover() and re-quote instead of resuming — resume() will not re-derive the hook the "
            "approval committed to")

    recovered = eth.recover_source(plan, create_checkpoint(plan, receipt, bridge.registry), required=True)
    if recovered.status is not Status.SOURCE_SUBMISSION_PENDING:
        emit(recovered)                       # history already holds the irreversible step
        emit.finalize()
        return to_progress(plan, recovered)

    if is_xreserve:
        quoted = eth.quote_deposit_usdc(plan=plan, secret_nonce=nonce)
        if saved_hook.lower() != ("0x" + quoted.hook_data.hex()).lower():     # always runs: validated above
            raise NotResumableError(
                "The re-quoted hook data does not match the hook this transfer's approval committed "
                "to: the secret nonce differs from the one used at execute(). Pass that same "
                "secret_nonce — depositing under another hook mints to a commitment the recipient "
                "can never open.")
    else:
        quoted = eth.quote_transfer_remote(plan=plan)
    if quoted.approval_required:
        raise NotResumableError(
            "The approval recorded for this transfer no longer covers it: its allowance is gone. "
            "Inspect Ethereum source history before starting another transfer — resume() will not "
            "issue a second approval.")

    call = eth.deposit_usdc(plan=plan, secret_nonce=nonce) if is_xreserve else eth.transfer_remote(plan=plan)
    result = call.send(wait=True, timeout_seconds=timeout_seconds, poll_seconds=poll_seconds,
                       on_checkpoint=emit)
    submitted = result.receipt
    prior = [a for a in (state.get("approvalTxIds") or []) if isinstance(a, str)]
    approvals = prior + [a for a in (submitted.protocol_state.get("approvalTxIds") or []) if a not in prior]
    if approvals != list(submitted.protocol_state.get("approvalTxIds") or []):
        submitted = submitted.replace(protocol_state={**submitted.protocol_state, "approvalTxIds": approvals})
    emit(submitted)
    emit.finalize()
    return to_progress(plan, submitted)


# ── complete ──────────────────────────────────────────────────────────────────

def complete(bridge, progress: Progress, *, secret_nonce: str | None = None,
             on_checkpoint: Callable | None = None, proving: str = "delegate") -> Progress:
    """Submit the one user-signed Aleo transaction a private USDCx mint needs.

    Requires ``progress.next == "complete"`` — Circle has attested the deposit and the receipt
    carries ``next_action == {"kind": "xreserve-private-mint", "chainId": <aleo chain>}``. The
    persisted payload (305 bytes), message hash (32 bytes) and attestation hex are re-validated
    first, then one of two paths runs:

    * a ``preparedDestinationTransaction`` left by an earlier interrupted attempt is rebroadcast
      byte-for-byte (a duplicate answer is success, and no ``secret_nonce`` is needed — those bytes
      are already proved), or
    * ``bridge.xreserve.private_mint`` builds the mint — re-verifying on the way that
      ``(recipient, secret_nonce)`` really opens the attested hook-data commitment, so a wrong nonce
      never reaches proving — which is then proved, checkpointed BEFORE broadcast, and broadcast.

    The source deposit is never repeated, and the secret nonce, the attestation and the hook data
    are never written to a receipt, a checkpoint or a ``Progress``. ``secret_nonce`` must be the
    value used at ``execute``; it is required for a private plan on the proving path
    (``ConfigurationError``, raised before any RPC).
    """
    if progress.next != "complete":
        raise NotResumableError(
            "Bridge progress has no destination action to complete (next must be 'complete'); call "
            "wait() to refresh it — the Circle attestation may still be pending")
    plan, receipt = progress.plan, progress.receipt
    # Deliberately no _require_active here (unlike resume): by the time a transfer reaches
    # DESTINATION_ACTION_REQUIRED the USDC is already deposited on Ethereum, and refusing the mint
    # because the registry has since parked the route would strand it. The proving path still hits
    # XReserveModule's own availability check; a rebroadcast of already-proved bytes needs none.
    resolved = resolve_route(bridge.registry, plan)
    _check_receipt(plan, receipt)
    action = receipt.next_action or {}
    if (receipt.status is not Status.DESTINATION_ACTION_REQUIRED
            or action.get("kind") != "xreserve-private-mint"
            or action.get("chainId") != resolved.destination_chain.id):
        raise NotResumableError(
            "Bridge receipt carries no supported destination action: complete() only finishes an "
            "xReserve private mint, on this transfer's own destination chain")
    if (resolved.route.protocol != "xreserve" or resolved.source_chain.family != "evm"
            or resolved.destination_chain.family != "aleo"):
        raise UnsupportedRouteError("Destination completion is not implemented for this bridge route")

    state = receipt.protocol_state
    payload = _hex_bytes(state.get("payload"), length=305)
    message_hash = _hex_bytes(state.get("messageHash"), length=32)
    attestation = _hex_bytes(state.get("attestation"))
    if payload is None or message_hash is None or not attestation:
        raise AttestationError(
            "Ready xReserve receipt is missing its validated Circle attestation (a 305-byte payload, "
            "a 32-byte messageHash and the attestation hex); refresh it with wait()")
    emit = _Emitter(bridge, plan, on_checkpoint)

    prepared_dest = state.get("preparedDestinationTransaction")
    if prepared_dest is not None:
        tx_id = _assert_prepared_id(prepared_dest, receipt.id, "prepared Aleo destination transaction")
        submit_serialized(bridge, prepared_dest, tx_id)
        submitted = receipt.replace(
            status=Status.DESTINATION_CONFIRMING, destination_tx_id=tx_id, next_action=None,
            protocol_state={k: v for k, v in state.items() if k != "preparedDestinationTransaction"})
        emit(submitted)
        emit.finalize()
        return to_progress(plan, submitted)

    nonce = _mint_secret(plan, secret_nonce)                               # before any RPC
    att = Attestation(payload=payload, message_hash=message_hash, attestation=attestation, status="complete")
    call = bridge.xreserve.private_mint(att, plan.recipient, secret_nonce=nonce, route=resolved.route)
    prepared = _prepare_aleo(call, proving)
    # invariant 3: the exact bytes live in a checkpoint before the network can ever see them
    emit(receipt.replace(id=prepared.transaction_id,
                         protocol_state={**state, "preparedDestinationTransaction": prepared.serialized}))
    call.submit_prepared(prepared, wait=False)          # polling is wait()'s job, not complete()'s
    submitted = receipt.replace(
        status=Status.DESTINATION_CONFIRMING, destination_tx_id=prepared.transaction_id, next_action=None,
        protocol_state={**{k: v for k, v in state.items() if k != "preparedDestinationTransaction"},
                        "destinationProgram": call.program_id, "destinationFunction": call.function_name})
    emit(submitted)
    emit.finalize()
    return to_progress(plan, submitted)
