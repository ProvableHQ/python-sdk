"""Aleo-origin Hyperlane warp routes (port of veil protocols/hyperlane/aleo.ts and utils/hyperlaneDelivery.ts).

Seven-input ``transfer_remote`` with allowance slot 0 = live IGP payment; IGP quote from
``hyp_hook_manager.aleo/destination_gas_configs``; delivery read from ``hyp_mailbox.aleo/deliveries``.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Callable

from . import encoding as enc
from ._calls import AleoCall
from .errors import (AmbiguousRouteError, ConfigurationError, InvalidAmountError, InvalidRecipientError,
                     RouteNotFoundError, RouteUnavailableError, UnsupportedRouteError)
from .registry import Asset, Chain, Route
from .types import DispatchReceipt, GasQuote, Receipt, Status
from .units import format_decimal_amount, parse_decimal_amount, resolve_amount

if TYPE_CHECKING:  # pragma: no cover
    from .client import Bridge

MAX_U64 = (1 << 64) - 1
GAS_QUOTE_SCALE = 10_000_000_000          # fixed by hyp_hook_manager.aleo post_dispatch
ZERO_GAS_LIMIT_FALLBACK = 50_000
_GAS_FIELDS = ("gas_overhead", "exchange_rate", "gas_price")
_GAS_FIELD_RE = re.compile(r"\b(gas_overhead|exchange_rate|gas_price)\s*:\s*(\d+)u128")


def parse_gas_config(literal: str) -> dict[str, int]:
    """``{ gas_overhead: 159337u128, exchange_rate: 402u128, gas_price: 1000000000u128 }`` → ints."""
    found = {m.group(1): int(m.group(2)) for m in _GAS_FIELD_RE.finditer(literal)}
    missing = [f for f in _GAS_FIELDS if f not in found]
    if missing:
        raise ConfigurationError(f"Hyperlane destination gas configuration is malformed (missing {missing}): {literal!r}")
    return found


def gas_config_key(route: Route) -> str:
    """The ``destination_gas_configs`` key the hook manager reads at finalization."""
    return f"{{ igp: {route.meta_str('aleoMailboxDefaultHook')}, destination: {route.meta_int('aleoDestinationDomain')}u32 }}"


def compute_gas_payment(*, gas_limit: int, gas_overhead: int, gas_price: int, exchange_rate: int) -> int:
    """Exact integer formula asserted on chain: ``(limit + overhead) * price * rate // 10^10`` as a positive u64."""
    payment = ((gas_limit + gas_overhead) * gas_price * exchange_rate) // GAS_QUOTE_SCALE
    if payment <= 0 or payment > MAX_U64:
        raise ConfigurationError(f"Hyperlane hook payment does not fit a positive u64: {payment}")
    return payment


def _allowance(route: Route, index: int, amount: int | None) -> str:
    value = str(amount) if amount is not None else route.meta_str(f"aleoAllowanceAmount{index}")
    return f"{{ spender: {route.meta_str(f'aleoAllowanceSpender{index}')}, amount: {value}u64 }}"


class HyperlaneModule:
    """``bridge.hyperlane`` — Aleo-side Hyperlane reads and the ``transfer_remote`` write."""

    def __init__(self, bridge: "Bridge") -> None:
        self._bridge = bridge

    # ── route resolution ──
    def _aleo_chain(self) -> Chain:
        chains = [c for c in self._bridge.registry.chains(environment=self._bridge.environment) if c.family == "aleo"]
        if len(chains) != 1:
            raise ConfigurationError(f"Registry must define exactly one Aleo chain for {self._bridge.environment}")
        return chains[0]

    def _aleo_asset(self, asset: Any) -> Asset:
        resolved = self._bridge.registry.asset(asset)
        if resolved.chain_id != self._aleo_chain().id:
            raise UnsupportedRouteError(
                f"{resolved.id} is not an Aleo asset on {self._bridge.environment}; Aleo-origin Hyperlane transfers "
                "start from aleo/eth, aleo/wbtc, aleo/usdt or aleo/sol (use bridge.eth / bridge.sol for other origins)")
        return resolved

    def _route_for(self, asset_or_route: Any) -> Route:
        if isinstance(asset_or_route, Route):
            route = asset_or_route
            if route.protocol != "hyperlane":
                raise UnsupportedRouteError(f"Not a Hyperlane route: {route.id}")
            self._aleo_asset(route.source_asset_id)
            return route
        return self.outbound_route(asset_or_route)

    def outbound_route(self, asset: Any) -> Route:
        """The single active, non-placeholder Hyperlane route leaving this Aleo asset."""
        source = self._aleo_asset(asset)
        candidates = [r for r in self._bridge.registry.routes(protocol="hyperlane", include_unavailable=True,
                                                             environment=self._bridge.environment)
                      if r.source_asset_id == source.id]
        if not candidates:
            raise RouteNotFoundError(f"No Hyperlane route leaves {source.id}")
        executable = [r for r in candidates if r.active and r.metadata.get("aleoPlaceholderConfiguration") is not True]
        if not executable:
            detail = ", ".join(f"{r.id} ({r.availability})" for r in candidates)
            raise RouteUnavailableError(f"Hyperlane routes from {source.id} are not executable: {detail}")
        if len(executable) > 1:
            raise AmbiguousRouteError(f"Several active Hyperlane routes leave {source.id}: {[r.id for r in executable]}")
        return executable[0]

    # ── reads ──
    def quote_gas_payment(self, asset: Any) -> GasQuote:
        """Live relayer payment for the route (the exact u64 the hook asserts); quote right before proving."""
        route = self._route_for(asset)
        literal = self._bridge.mapping_value(route.meta_str("aleoHookManagerProgram"), "destination_gas_configs",
                                             gas_config_key(route))
        if literal is None:
            raise ConfigurationError(f"Hyperlane destination gas configuration is missing on chain: {route.id}")
        config = parse_gas_config(literal)
        if config["exchange_rate"] == 0 or config["gas_price"] == 0:
            raise ConfigurationError(f"Hyperlane destination gas configuration is unpriced: {route.id}")
        gas_limit = int(route.meta_str("aleoRemoteRouterGas")) or ZERO_GAS_LIMIT_FALLBACK
        payment = compute_gas_payment(gas_limit=gas_limit, gas_overhead=config["gas_overhead"],
                                      gas_price=config["gas_price"], exchange_rate=config["exchange_rate"])
        return GasQuote(route.id, gas_limit, config["gas_overhead"], config["gas_price"], config["exchange_rate"], payment)

    def _mailbox_program(self) -> str:
        for route in self._bridge.registry.routes(protocol="hyperlane", include_unavailable=True, environment=self._bridge.environment):
            program = route.metadata.get("aleoMailboxProgram")
            if isinstance(program, str) and program:
                return program
        raise ConfigurationError(f"No Aleo Hyperlane mailbox program is configured for {self._bridge.environment}")

    def is_delivered(self, message_id: "str | bytes") -> bool:
        """Whether ``hyp_mailbox.aleo/deliveries`` holds the message (mapping presence is the acceptance signal)."""
        try:
            raw = enc.hex_to_bytes(message_id, 32)
        except ValueError as exc:
            raise ConfigurationError("Hyperlane delivery requires a 32-byte message id") from exc
        return self._bridge.mapping_value(self._mailbox_program(), "deliveries", enc.hyperlane_delivery_key(raw)) is not None

    # ── transfer_remote ──
    def build_transfer_remote_inputs(self, route: Route, *, recipient: str, amount_atomic: int,
                                     gas_payment_microcredits: int, decimals: tuple[int, int]) -> list[str]:
        """The seven ``transfer_remote`` literals (brief §3.3). Pure; works for placeholder routes too (inspection only)."""
        if isinstance(gas_payment_microcredits, bool) or not isinstance(gas_payment_microcredits, int) \
                or not (0 < gas_payment_microcredits <= MAX_U64):
            raise ConfigurationError(f"gas_payment_microcredits must be a positive u64: {gas_payment_microcredits}")
        if amount_atomic <= 0:
            raise InvalidAmountError("Bridge transfer amount must be greater than zero")
        registry = self._bridge.registry
        destination = registry.asset(route.destination_asset_id)
        destination_chain = registry.chain(destination.chain_id)
        if not destination.matches_address(recipient):
            raise InvalidRecipientError(f"Recipient does not match the {destination.chain_id} address format: {recipient}")
        if destination_chain.family == "evm":
            limbs = enc.evm_address_to_hyperlane_recipient(recipient)
        elif destination_chain.family == "solana":
            limbs = enc.solana_address_to_hyperlane_recipient(recipient)
        else:
            raise UnsupportedRouteError(f"Unsupported Hyperlane destination family {destination_chain.family!r}: {route.id}")
        local_decimals, remote_decimals = decimals
        domain = route.meta_int("aleoDestinationDomain")
        app_metadata = (f"{{ token_type: {route.meta_str('aleoTokenType')}u8, token_owner: {route.meta_str('aleoTokenOwner')}, "
                        f"ism: {route.meta_str('aleoIsm')}, hook: {route.meta_str('aleoHook')}, "
                        f"token_id: {route.meta_str('aleoTokenId')}, local_decimals: {local_decimals}u8, "
                        f"remote_decimals: {remote_decimals}u8 }}")
        mailbox_state = (f"{{ default_hook: {route.meta_str('aleoMailboxDefaultHook')}, "
                         f"required_hook: {route.meta_str('aleoMailboxRequiredHook')} }}")
        remote_router = (f"{{ domain: {domain}u32, recipient: {route.meta_str('aleoRemoteRouterRecipient')}, "
                         f"gas: {route.meta_str('aleoRemoteRouterGas')}u128 }}")
        allowances = "[" + ", ".join(_allowance(route, i, gas_payment_microcredits if i == 0 else None) for i in range(4)) + "]"
        return [app_metadata, mailbox_state, remote_router, f"{domain}u32", enc.u128_pair_literal(limbs),
                f"{amount_atomic}u128", allowances]

    def transfer_remote(self, asset: Any, recipient: str, *, amount: Any = None, amount_atomic: int | None = None,
                        as_signer: bool = False, gas_payment_microcredits: int | None = None) -> AleoCall[DispatchReceipt]:
        """Withdraw an Aleo warp asset to Ethereum/Solana. Quotes the IGP payment now unless pinned; the
        lifecycle layer (plan 4) re-quotes at the last responsible moment by calling this again."""
        route = self.outbound_route(asset)
        registry = self._bridge.registry
        source, destination = registry.asset(route.source_asset_id), registry.asset(route.destination_asset_id)
        atomic = resolve_amount(amount=amount, amount_atomic=amount_atomic, decimals=source.decimals)
        parse_decimal_amount(format_decimal_amount(atomic, source.decimals), destination.decimals)  # veil prepare(): representable on both sides
        payment = gas_payment_microcredits if gas_payment_microcredits is not None \
            else self.quote_gas_payment(route).payment_microcredits
        decimals = (route.meta_int("aleoLocalDecimals", source.decimals), route.meta_int("aleoRemoteDecimals", destination.decimals))
        inputs = self.build_transfer_remote_inputs(route, recipient=recipient, amount_atomic=atomic,
                                                   gas_payment_microcredits=payment, decimals=decimals)
        program = route.meta_str("aleoRouterProgram")
        function = "transfer_remote_as_signer" if as_signer else "transfer_remote"

        def build(tx_id: str, _outputs: list[str]) -> DispatchReceipt:
            receipt = Receipt(id=tx_id, protocol="hyperlane", status=Status.SOURCE_CONFIRMING, source_tx_id=tx_id,
                              protocol_state={"routeId": route.id, "sourceProgram": program, "sourceFunction": function,
                                              "amountAtomic": str(atomic), "recipient": recipient,
                                              "gasPaymentMicrocredits": str(payment)})
            return DispatchReceipt(transaction_id=tx_id, route_id=route.id, message_id=None, amount_atomic=atomic, receipt=receipt)

        return self._bridge._call(program, function, inputs, build)


__all__ = ["GAS_QUOTE_SCALE", "MAX_U64", "ZERO_GAS_LIMIT_FALLBACK", "HyperlaneModule", "compute_gas_payment",
           "gas_config_key", "parse_gas_config"]
