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

from . import _sealevel
from ._plan import build_plan
from .checkpoint import Checkpoint, create_checkpoint
from .errors import (
    CheckpointInvalidError,
    ConfigurationError,
    DeliveryUnknownError,
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
    """The recipient's destination balance, or None when we cannot read it.

    Only read when the destination connection IS the recipient (there is no per-address balance
    read in the module contracts); otherwise omit the delivery-verification pair rather than
    baseline the wrong account. The RPC read itself is best-effort (review item 8): a transient
    failure here is only ever used as an advisory baseline (``execute``'s pre-broadcast checkpoint)
    or re-attempted by ``get_status``/``wait`` — it must never raise and block funds movement.
    """
    chain, asset = resolved.destination_chain, resolved.destination_asset
    if chain.family == "evm":
        conn = getattr(bridge, "ethereum", None)
        if (conn is None or not conn.address
                or conn.address.lower() != plan.recipient.lower()
                or asset.locator is None or asset.locator.kind not in ("native", "evm-contract")):
            return None
        try:
            return int(bridge.eth.balance(asset.id))
        except Exception:                                              # noqa: BLE001 — advisory read only
            return None
    if chain.family == "solana":
        conn = getattr(bridge, "solana", None)
        if (conn is None or conn.address != plan.recipient
                or asset.locator is None or asset.locator.kind != "native"):
            return None
        try:
            return int(bridge.sol.balance())
        except Exception:                                              # noqa: BLE001 — advisory read only
            return None
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
        call = bridge.xreserve.burn(plan.recipient, amount_atomic=plan.amount_atomic, mode=burn_mode,
                                    record=record, merkle_proof=merkle_proof)
        receipt = _run_aleo_leg(bridge, plan, call, proving=proving, emit=emit, extra_state={})
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
    signature or an unrelated transaction hash of a different width.
    """
    state = receipt.protocol_state
    message_id = state.get("messageId")
    if isinstance(message_id, str) and message_id:
        return message_id
    if isinstance(receipt.id, str) and _MESSAGE_ID_RE.fullmatch(receipt.id):
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
        try:
            logs = _module(bridge, "sol")._transaction_logs(receipt.source_tx_id)
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

    # 8. xReserve Aleo→EVM: Circle exposes no canonical delivery query
    if (receipt.status is Status.DELIVERY_PENDING and route.protocol == "xreserve"
            and src.family == "aleo" and dst.family == "evm"):
        return receipt

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


__all__ = ["MINT_MODES", "ResolvedRoute", "aleo_transaction_status", "execute", "get_status", "prepare",
           "quote", "resolve_route"]
