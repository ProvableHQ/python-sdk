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

from ._plan import build_plan
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
from .types import AleoHyperlaneQuote, AleoXReserveQuote, Fee, Plan, Quote
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
        # Unlike EthModule's quote methods, SolModule.quote_transfer_remote's ``recipient`` has
        # no default — it is a required positional argument even though it is fully overwritten
        # from ``plan`` on the real module's plan branch.
        q = _module(bridge, "sol").quote_transfer_remote(plan.recipient, plan=plan)
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


__all__ = ["MINT_MODES", "ResolvedRoute", "prepare", "quote", "resolve_route"]
