"""``build_plan`` — the one Plan builder every origin chain shares (brief §2.1, mirrors veil ``prepare``).

Plan 4's ``lifecycle.prepare`` is the Tier-1 entry point; this helper is the Tier-2 path, so
``bridge.eth.*`` / ``bridge.sol.*`` calls carry a checkpointable plan without importing lifecycle.
It lives outside ``eth.py`` because the step shape is protocol- and chain-driven, not Ethereum-driven:
the wallet steps' executor comes from the source chain's family, so the same builder describes an
Ethereum, Solana or Aleo origin.
"""
from __future__ import annotations

from .errors import BridgeError, UnsupportedRouteError
from .registry import Asset, Registry, Route
from .types import Plan, Step
from .units import format_decimal_amount

WALLET_EXECUTOR_BY_FAMILY = {"evm": "evm-wallet", "solana": "solana-wallet", "aleo": "aleo-wallet"}


def _wallet_executor(registry: Registry, source: Asset) -> str:
    """Who signs the source-chain steps: the wallet of the chain the funds leave from."""
    family = registry.chain(source.chain_id).family
    executor = WALLET_EXECUTOR_BY_FAMILY.get(family)
    if executor is None:
        raise BridgeError(f"No wallet executor for chain family {family!r} ({source.chain_id})")
    return executor


def build_plan(registry: Registry, route: Route, *, amount_atomic: int, recipient: str, sender: str | None,
               mint_mode: str = "public") -> Plan:
    """Build the ``Plan`` for one bridge route (mirrors veil ``prepare`` steps, brief §2.1)."""
    source: Asset = registry.asset(route.source_asset_id)
    destination: Asset = registry.asset(route.destination_asset_id)
    if mint_mode not in ("public", "record", "private"):
        raise BridgeError(f"mint_mode must be public, record or private; got {mint_mode!r}")
    if mint_mode != "public" and route.protocol != "xreserve":
        raise BridgeError("mint_mode other than public applies only to xReserve deposits to Aleo")
    if amount_atomic <= 0:
        raise BridgeError("amount_atomic must be positive")
    wallet = _wallet_executor(registry, source)
    source_family = registry.chain(source.chain_id).family
    destination_family = registry.chain(destination.chain_id).family
    if route.protocol == "xreserve":
        if source_family == "evm" and destination_family == "aleo":
            steps = (Step("source-approval", "approve", wallet, False),
                     Step("source-deposit", "deposit", wallet, True),
                     Step("deposit-attestation", "wait-attestation", "protocol", False),
                     Step("destination-mint", "mint", "aleo-wallet" if mint_mode == "private" else "protocol", False))
        elif source_family == "aleo" and destination_family == "evm":
            steps = (Step("source-burn", "burn", wallet, True),
                     Step("withdrawal-attestation", "wait-attestation", "protocol", False),
                     Step("destination-withdrawal", "withdraw", "protocol", False),
                     Step("destination-confirmation", "confirm-delivery", "protocol", False))
        else:
            raise UnsupportedRouteError(f"Unsupported xReserve route direction: {route.id}")
    else:
        # Aleo ARC-20 tokens need no on-chain approval; only a non-Aleo token source does.
        needs_approval = source.kind == "token" and source_family != "aleo"
        steps = tuple([Step("source-approval", "approve", wallet, False)] if needs_approval else []) + (
            Step("source-dispatch", "dispatch", wallet, True),
            Step("message-delivery", "wait-delivery", "protocol", False),
            Step("destination-confirmation", "confirm-delivery", "protocol", False))
    return Plan(route_id=route.id, registry_version=registry.version, protocol=route.protocol,
                environment=route.environment, source_asset_id=source.id, destination_asset_id=destination.id,
                amount=format_decimal_amount(amount_atomic, source.decimals), amount_atomic=amount_atomic,
                recipient=recipient, sender=sender, mint_mode=mint_mode, steps=steps)


__all__ = ["WALLET_EXECUTOR_BY_FAMILY", "build_plan"]
