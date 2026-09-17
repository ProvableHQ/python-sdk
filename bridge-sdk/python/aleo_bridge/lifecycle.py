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
from dataclasses import dataclass

from ._plan import build_plan
from .errors import (
    CheckpointInvalidError,
    ConfigurationError,
    InvalidAmountError,
    InvalidRecipientError,
    RegistryVersionMismatchError,
    RouteUnavailableError,
)
from .registry import Asset, Chain, Registry, Route
from .types import Plan
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


__all__ = ["MINT_MODES", "ResolvedRoute", "prepare", "resolve_route"]
