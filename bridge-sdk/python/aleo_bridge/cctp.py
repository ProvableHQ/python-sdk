"""Move native USDC between Arc and Ethereum, Base, or Arbitrum through Circle CCTP V2.

_source: ProvableHQ/veil packages/bridge/src/protocols/cctp/evm.ts @
3c3b457bd5f63620657321893a2487e489750d24. Network effects stay behind lifecycle verbs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

import requests

from ._cctp_message import address_bytes
from .errors import AttestationError, ConfigurationError, InvalidAmountError
from .registry import Route
from .types import CctpOptions, EvmCctpQuote, Fee, Plan, normalize_cctp
from .units import format_decimal_amount, parse_decimal_amount


@dataclass(frozen=True)
class Metadata:
    route: Route
    source_chain: str
    destination_chain: str
    source_chain_id: int
    destination_chain_id: int
    source_domain: int
    destination_domain: int
    messenger: str
    transmitter: str
    source_token: str
    destination_token: str
    attestation_url: str


def _uint(value: Any, field: str, maximum: int = 2**256 - 1) -> int:
    if (isinstance(value, bool) or not isinstance(value, (int, str))
            or not re.fullmatch(r"[0-9]+", str(value)) or int(value) > maximum):
        raise ConfigurationError(f"Invalid CCTP {field}")
    return int(value)


def _basis_points(amount: int, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str, float)) or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", str(value)):
        raise AttestationError("Invalid Circle minimumFee")
    whole, _, fraction = str(value).partition(".")
    denominator = 10 ** len(fraction) * 10_000
    return (amount * int(whole + fraction) + denominator - 1) // denominator


class CctpModule:
    """Quote CCTP fees and track delivery through the bridge's shared lifecycle."""

    def __init__(self, bridge: Any) -> None:
        self.bridge = bridge
        self.circle_session: Any = None

    def _metadata(self, plan: Plan) -> Metadata:
        from .lifecycle import resolve_route
        resolved = resolve_route(self.bridge.registry, plan)
        route, src, dst = resolved.route, resolved.source_asset, resolved.destination_asset
        if plan.protocol != "cctp" or not route.active or plan.environment != self.bridge.environment:
            raise ConfigurationError("CCTP plan does not match an active route in this environment")
        for asset, chain in ((src, resolved.source_chain), (dst, resolved.destination_chain)):
            if (chain.family != "evm" or asset.key != "usdc" or asset.decimals != 6
                    or asset.locator is None or asset.locator.kind != "evm-contract"):
                raise ConfigurationError("CCTP requires canonical six-decimal EVM USDC")
            address_bytes(asset.locator.value)
        address_bytes(plan.recipient)
        if plan.sender is not None:
            address_bytes(plan.sender)
        amount = parse_decimal_amount(plan.amount, 6)
        if type(plan.amount_atomic) is not int or amount != plan.amount_atomic or not 0 < amount < 2**256:
            raise InvalidAmountError("CCTP amount must be a matching positive uint256")
        if plan.mint_mode != "public":
            raise ConfigurationError("CCTP does not support Aleo mint modes")
        options = normalize_cctp(plan.cctp)
        if options.max_fee is not None and parse_decimal_amount(options.max_fee, 6) >= amount:
            raise InvalidAmountError("CCTP max_fee must be less than the burn amount")
        for chain, key in ((resolved.source_chain, "sourceDomain"), (resolved.destination_chain, "destinationDomain")):
            domain = chain.protocol_domains.get("cctp")
            if type(domain) is not int or route.meta_int(key) != domain:
                raise ConfigurationError("CCTP route domains must match configured chain domains")
        messenger, transmitter = route.meta_str("tokenMessenger"), route.meta_str("messageTransmitter")
        address_bytes(messenger)
        address_bytes(transmitter)
        url = route.meta_str("attestationBaseUrl")
        if not url.startswith("https://"):
            raise ConfigurationError("CCTP attestation URL must use HTTPS")
        assert src.locator is not None and dst.locator is not None
        return Metadata(route, src.chain_id, dst.chain_id,
                        _uint(route.meta_int("sourceChainId"), "source chain", 2**32 - 1),
                        _uint(route.meta_int("destinationChainId"), "destination chain", 2**32 - 1),
                        route.meta_int("sourceDomain"), route.meta_int("destinationDomain"),
                        messenger, transmitter, src.locator.value, dst.locator.value, url.rstrip("/"))

    def _json(self, url: str) -> Any:
        if self.circle_session is None:
            self.circle_session = requests.Session()
        try:
            response = self.circle_session.get(url, timeout=30)
            if response.status_code == 404:
                return None
            if response.status_code != 200:
                raise AttestationError(f"Circle CCTP API returned HTTP {response.status_code}")
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            raise AttestationError("Circle CCTP request failed") from exc

    def quote(self, plan: Plan) -> EvmCctpQuote:
        """Read current CCTP fees and fix the maximum USDC deduction without signing."""
        m = self._metadata(plan)
        options = normalize_cctp(plan.cctp)
        finality = 1000 if options.speed == "fast" else 2000
        suffix = "?forward=true" if options.forwarding else ""
        response = self._json(f"{m.attestation_url}/v2/burn/USDC/fees/{m.source_domain}/{m.destination_domain}{suffix}")
        if not isinstance(response, list):
            raise AttestationError("Circle returned invalid CCTP fees")
        matches = [f for f in response if isinstance(f, dict) and type(f.get("finalityThreshold")) is int
                   and f["finalityThreshold"] == finality]
        if len(matches) != 1:
            raise AttestationError("Circle does not uniquely quote the requested CCTP finality")
        fee = matches[0]
        protocol = _basis_points(plan.amount_atomic, fee.get("minimumFee"))
        forwarding = 0
        if options.forwarding:
            forward = fee.get("forwardFee")
            if not isinstance(forward, dict):
                raise AttestationError("Invalid Circle forwarding fee")
            forwarding = _uint(forward.get("medium", forward.get("med")), "forwarding fee")
        required = protocol + forwarding
        cap = required if options.max_fee is None else parse_decimal_amount(options.max_fee, 6)
        if required > cap:
            raise InvalidAmountError("Live CCTP fees exceed the approved max_fee; request a new quote")
        if cap >= plan.amount_atomic:
            raise InvalidAmountError("CCTP fees must be less than the burn amount")
        receive = plan.amount_atomic - (cap if options.forwarding else required)
        plan = replace(plan, cctp=CctpOptions(options.speed, options.forwarding, format_decimal_amount(cap, 6)))
        fees = [Fee("protocol", m.destination_chain, plan.destination_asset_id, format_decimal_amount(protocol, 6), True)]
        if options.forwarding:
            fees.append(Fee("relayer", m.destination_chain, plan.destination_asset_id,
                            format_decimal_amount(forwarding, 6), True))
        return EvmCctpQuote("evm-cctp", plan, tuple(fees), format_decimal_amount(receive, 6), plan.amount_atomic,
                           receive, protocol, forwarding, cap, finality, options.forwarding)
