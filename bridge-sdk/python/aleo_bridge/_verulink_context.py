"""Resolve reviewed deployments and read fresh Verulink contract state."""
from __future__ import annotations

import re
from typing import Any

from ._verulink_abi import ABI
from ._verulink_wire import ZERO_ALEO, U64_MAX, address, decimal, from_aleo_bytes, integer, parse, same
from .errors import BridgeError
from .types import Plan
from .units import parse_decimal_amount


class Context:
    def __init__(self, bridge: Any, plan: Plan) -> None:
        from .lifecycle import resolve_route
        r = resolve_route(bridge.registry, plan)
        self.bridge, self.aleo, self.network = bridge, bridge.aleo, bridge.environment
        self.route, self.source, self.destination = r.route, r.source_asset, r.destination_asset
        self.outbound = r.source_chain.family == "aleo"
        aleo_chain, evm_chain = (r.source_chain, r.destination_chain) if self.outbound else (r.destination_chain, r.source_chain)
        if (plan.protocol != "verulink" or not r.route.active or plan.environment != bridge.environment
                or aleo_chain.family != "aleo" or evm_chain.family != "evm"
                or plan.mint_mode != "public" or plan.cctp is not None
                or self.source.decimals != 6 or self.destination.decimals != 6):
            raise BridgeError("Verulink requires an active public ALEO route between Aleo and EVM")
        self.aleo_chain, self.evm_chain = aleo_chain.id, evm_chain.id
        self.evm = bridge.evm(evm_chain.id).conn
        self.evm_chain_id = r.route.meta_int("evmChainId")
        self.confirmations = r.route.meta_int("sourceConfirmations")
        if self.evm_chain_id < 1 or self.confirmations < 1:
            raise BridgeError("Invalid Verulink chain id or confirmation policy")
        self.evm_domain = decimal(r.route.meta_str("evmDomain"), 2**128 - 1)
        self.aleo_domain = decimal(r.route.meta_str("aleoDomain"), 2**128 - 1)
        for chain, domain in ((aleo_chain, self.aleo_domain), (evm_chain, self.evm_domain)):
            if domain == "0" or ("verulink" in chain.protocol_domains and chain.protocol_domains["verulink"] != int(domain)):
                raise BridgeError("Verulink domain does not match chain metadata")
        self.evm_token = address(r.route.meta_str("evmToken"))
        self.evm_service = address(r.route.meta_str("evmTokenService"))
        self.evm_bridge = address(r.route.meta_str("evmBridge"))
        self.aleo_service = r.route.meta_str("aleoTokenService")
        self.aleo_bridge = r.route.meta_str("aleoBridge")
        self.aleo_service_address = r.route.meta_str("aleoServiceAddress")
        self.aleo_token_id = r.route.meta_str("aleoTokenId")
        self.collector_url = r.route.meta_str("collectorUrl")
        if (any(not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*\.aleo", p) for p in (self.aleo_service, self.aleo_bridge))
                or not re.fullmatch(r"aleo1[0-9a-z]{58}", self.aleo_service_address)
                or not re.fullmatch(r"[0-9]+field", self.aleo_token_id)):
            raise BridgeError("Invalid Verulink Aleo deployment")
        aa, ea = (self.source, self.destination) if self.outbound else (self.destination, self.source)
        if (aa.locator is None or aa.locator.value != "credits.aleo" or aa.locator.kind != "aleo-program"
                or ea.locator is None or ea.locator.kind != "evm-contract" or not same(ea.locator.value, self.evm_token)):
            raise BridgeError("Verulink asset does not match the reviewed token")
        if self.outbound:
            address(plan.recipient)
        else:
            self._aleo_address(plan.recipient)
        if plan.recipient == ZERO_ALEO or same(plan.recipient, "0x" + "00" * 20):
            raise BridgeError("Verulink recipient cannot be zero")
        if plan.sender is not None:
            self._aleo_address(plan.sender) if self.outbound else address(plan.sender)
        if (integer(plan.amount_atomic, "amount", U64_MAX) < 1
                or parse_decimal_amount(plan.amount, 6) != plan.amount_atomic):
            raise BridgeError("Verulink amount must be a matching positive u64")
        if plan.amount_out is not None and not 0 < parse_decimal_amount(plan.amount_out, 6) <= plan.amount_atomic:
            raise BridgeError("Invalid Verulink quoted output")
        self.chain_token = f"{{ chain_id: {self.evm_domain}u128, token_id: {self.aleo_token_id} }}"

    @staticmethod
    def _aleo_address(value: str) -> None:
        from aleo import mainnet
        try:
            mainnet.Address.from_string(value)
        except ValueError as exc:
            raise BridgeError("Invalid Verulink Aleo address") from exc

    def check_chain(self) -> None:
        # Check the provider each pass, not Ethereum.chain_id's cached value.
        if self.evm.w3.eth.chain_id != self.evm_chain_id:
            raise BridgeError(f"Verulink requires EVM chain {self.evm_chain_id}")

    def contract(self, to: str) -> Any:
        return self.evm.w3.eth.contract(address=address(to), abi=ABI)

    def read(self, to: str, name: str, args: list[Any] | None = None) -> Any:
        return self.contract(to).get_function_by_name(name)(*(args or [])).call()

    def mapping(self, program: str, name: str, key: str) -> Any:
        value = self.aleo.network.get_program_mapping_value(program, name, key)
        return None if value is None else parse(value)

    def assert_active(self, *, destination_only: bool = False) -> None:
        self.check_chain()
        if (not destination_only or self.outbound) and (
                self.read(self.evm_service, "paused") is not False
                or self.read(self.evm_bridge, "paused") is not False):
            raise BridgeError("Verulink route is paused, unsupported, or unreadable")
        if not destination_only or not self.outbound:
            active = self.mapping(self.aleo_bridge, "bridge_settings", "3u8")
            if (self.mapping(self.aleo_service, "status", self.chain_token) is not False
                    or type(active) is not int or active != 1
                    or self.mapping(self.aleo_bridge, "supported_chains", self.evm_domain + "u128") is not True
                    or self.mapping(self.aleo_bridge, "supported_services", self.aleo_service_address) is not True):
                raise BridgeError("Verulink route is paused, unsupported, or unreadable")

    def configuration(self) -> tuple[int, int, int]:
        self.check_chain()
        own = self.read(self.evm_service, "self")
        token = self.read(self.evm_service, "supportedTokens", [self.evm_token])
        if (not isinstance(own, (list, tuple)) or len(own) != 2
                or own[0] != int(self.evm_domain) or not same(own[1], self.evm_service)
                or self.read(self.evm_service, "destChainId") != int(self.aleo_domain)
                or not same(self.read(self.evm_service, "erc20Bridge"), self.evm_bridge)
                or not isinstance(token, (list, tuple)) or len(token) != 7 or not same(token[0], self.evm_token)
                or token[2] != self.aleo_token_id or token[3] != self.aleo_service_address or token[6] is not True
                or self.read(self.evm_token, "decimals") != 6
                or from_aleo_bytes(self.mapping(self.aleo_service, "other_chain_token_service", self.chain_token)) != self.evm_service
                or from_aleo_bytes(self.mapping(self.aleo_service, "other_chain_token_address", self.chain_token)) != self.evm_token):
            raise BridgeError("Verulink onchain peers do not match the reviewed deployment")
        if self.outbound:
            rate, minimum, maximum = [self.mapping(self.aleo_service, key, self.chain_token)
                                      for key in ("platform_fee", "min_transfers", "max_transfers")]
        else:
            collector = address(self.read(self.evm_service, "feeCollector"))
            rate, minimum, maximum = self.read(collector, "platformFees", [self.evm_token]), token[4], token[5]
        return integer(rate, "fee rate", 100_000), integer(minimum, "minimum"), integer(maximum, "maximum")
