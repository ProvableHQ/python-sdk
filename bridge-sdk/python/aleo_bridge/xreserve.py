"""Aleo side of Circle xReserve (port of veil protocols/xreserve/aleoToEvm.ts, the ``complete`` half of
evmToAleo.ts, and utils/xreserveDelivery.ts): USDCx burns, the user-signed private mint, attestation
lookup and the ``nullifier`` delivery read."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import encoding as enc
from ._calls import AleoCall
from ._keccak import keccak256
from .circle import CircleClient
from .errors import (AttestationError, ConfigurationError, InvalidAmountError, InvalidRecipientError,
                     RouteNotFoundError, RouteUnavailableError, UnsupportedRouteError)
from .registry import Route
from .types import Attestation, BurnReceipt, MintReceipt, Receipt, Status
from .units import format_decimal_amount, resolve_amount

if TYPE_CHECKING:  # pragma: no cover
    from .client import Bridge

ETHEREUM_DESTINATION_DOMAIN = 0
BURN_MODES = ("private", "public", "public-as-signer")


class XReserveModule:
    """``bridge.xreserve`` — every xReserve step that happens on Aleo."""

    def __init__(self, bridge: "Bridge") -> None:
        self._bridge = bridge
        self.circle_session: Any = None   # injectable HTTP session (tests); None → requests.Session()

    # ── routes ──
    def _single(self, direction: str) -> Route:
        registry, aleo = self._bridge.registry, self._bridge.aleo_chain().id
        matches = [r for r in registry.routes(bridge_protocol="xreserve", include_unavailable=True, environment=self._bridge.environment)
                   if registry.asset(r.destination_asset_id if direction == "inbound" else r.source_asset_id).chain_id == aleo]
        if not matches:
            raise RouteNotFoundError(f"No {direction} xReserve route for {self._bridge.environment}")
        if len(matches) > 1:
            raise ConfigurationError(f"Several {direction} xReserve routes for {self._bridge.environment}: {[r.id for r in matches]}")
        if not matches[0].active:
            raise RouteUnavailableError(f"xReserve route is not executable: {matches[0].id}")
        return matches[0]

    def inbound_route(self) -> Route:
        """Ethereum → Aleo (mint side) for this environment."""
        return self._single("inbound")

    def outbound_route(self) -> Route:
        """Aleo → Ethereum (burn side) for this environment."""
        return self._single("outbound")

    def _validated(self, route: Route, *, direction: str) -> Route:
        registry = self._bridge.registry
        if route.protocol != "xreserve":
            raise UnsupportedRouteError(f"Not an xReserve route: {route.id}")
        if not route.active:
            raise RouteUnavailableError(f"xReserve route is not executable: {route.id}")
        source = registry.chain(registry.asset(route.source_asset_id).chain_id).family
        destination = registry.chain(registry.asset(route.destination_asset_id).chain_id).family
        if direction == "burn" and (source, destination) != ("aleo", "evm"):
            raise UnsupportedRouteError(f"USDCx burn requires an Aleo-to-Ethereum route, got {route.id}")
        if direction == "mint" and (source, destination) != ("evm", "aleo"):
            raise UnsupportedRouteError(f"private_mint requires an Ethereum-to-Aleo route, got {route.id}")
        if route.meta_int("ethereumDestinationDomain") != ETHEREUM_DESTINATION_DOMAIN:
            raise ConfigurationError(f"xReserve Ethereum destination domain must be {ETHEREUM_DESTINATION_DOMAIN}: {route.id}")
        return route

    # ── burn ──
    def build_burn_inputs(self, route: Route, *, mode: str, amount_atomic: int, recipient: str,
                          record: str | None, merkle_proof: str | None) -> tuple[str, str, list[str]]:
        """``(program, function, inputs)`` for one USDCx burn (brief §3.4). Pure."""
        if mode not in BURN_MODES:
            raise ConfigurationError(f"Unsupported USDCx burn mode {mode!r}; expected one of {BURN_MODES}")
        self._validated(route, direction="burn")
        source = self._bridge.registry.asset(route.source_asset_id)
        if amount_atomic <= 0:
            raise InvalidAmountError("USDCx burn amount must be greater than zero")
        fee = int(route.meta_str("withdrawalFeeAtomic"))
        if amount_atomic <= fee:
            raise InvalidAmountError(
                f"USDCx burn amount must exceed the {format_decimal_amount(fee, source.decimals)} {source.symbol} withdrawal fee")
        recipient32 = enc.evm_address_to_bytes32(recipient)          # InvalidRecipientError
        amount_lit, domain_lit, recipient_lit = f"{amount_atomic}u128", f"{ETHEREUM_DESTINATION_DOMAIN}u32", enc.u8_array_literal(recipient32)
        if mode == "private":
            if not isinstance(record, str) or not record.strip():
                raise ConfigurationError(f"private_burn requires a USDCx Token record from {route.meta_str('remoteToken')}")
            if not isinstance(merkle_proof, str) or not (merkle_proof.startswith("[") and merkle_proof.endswith("]")):
                raise ConfigurationError("private_burn requires an encoded [MerkleProof; 2] Aleo literal")
            return route.meta_str("wrapperProgram"), "private_burn", [record, amount_lit, domain_lit, recipient_lit, merkle_proof]
        function = "burn_public" if mode == "public" else "burn_public_as_signer"
        return route.meta_str("bridgeProgram"), function, [amount_lit, domain_lit, recipient_lit]

    def burn(self, recipient: str, *, amount: Any = None, amount_atomic: int | None = None, mode: str = "private",
             record: str | None = None, merkle_proof: str | None = None) -> AleoCall[BurnReceipt]:
        """Burn USDCx for USDC on Ethereum. ``private`` (default) spends a Token record via the wrapper and needs a
        freeze-list exclusion proof — both are resolved from chain state when not supplied. Minimum: more than
        the 2 USDCx withdrawal fee. The Aleo burn-attestation service forwards accepted burns to Circle."""
        if mode not in BURN_MODES:
            raise ConfigurationError(f"Unsupported USDCx burn mode {mode!r}; expected one of {BURN_MODES}")
        if mode != "private" and (record is not None or merkle_proof is not None):
            raise ConfigurationError(
                f"mode={mode!r} burns the public balance; record=/merkle_proof= only apply to mode='private'")
        route = self._validated(self.outbound_route(), direction="burn")
        source = self._bridge.registry.asset(route.source_asset_id)
        atomic = resolve_amount(amount=amount, amount_atomic=amount_atomic, decimals=source.decimals)
        if mode == "private":
            token_program = route.meta_str("remoteToken")
            if record is None:
                privacy = getattr(self._bridge, "privacy", None)
                if privacy is None:
                    raise ConfigurationError(
                        "burn(mode='private') needs a record; pass record= explicitly (bridge.privacy is not available yet)")
                record = privacy.select_record(token_program, atomic)
            if merkle_proof is None:
                freezelist = getattr(self._bridge, "freezelist", None)
                if freezelist is None:
                    raise ConfigurationError(
                        "burn(mode='private') needs merkle_proof; pass merkle_proof= explicitly "
                        "(bridge.freezelist is not available yet)")
                merkle_proof = freezelist.exclusion_proof(self._bridge.aleo_address(), token_program)
        program, function, inputs = self.build_burn_inputs(route, mode=mode, amount_atomic=atomic, recipient=recipient,
                                                           record=record, merkle_proof=merkle_proof)
        recipient_hex = enc.to_hex(enc.evm_address_to_bytes32(recipient))

        def build(tx_id: str, _outputs: list[str]) -> BurnReceipt:
            receipt = Receipt(id=tx_id, protocol="xreserve", status=Status.SOURCE_CONFIRMING, source_tx_id=tx_id,
                              protocol_state={"routeId": route.id, "burnMode": mode, "amountAtomic": str(atomic),
                                              "nativeDomain": ETHEREUM_DESTINATION_DOMAIN, "nativeRecipientBytes32": recipient_hex,
                                              "sourceProgram": program, "sourceFunction": function,
                                              "forwardingService": "aleo-burn-attestation"})
            return BurnReceipt(transaction_id=tx_id, route_id=route.id, mode=mode, amount_atomic=atomic, receipt=receipt)

        return self._bridge._call(program, function, inputs, build)

    # ── private mint ──
    def hook_data(self, mode: str, recipient: str, secret_nonce: str = "0scalar") -> bytes:
        return enc.xreserve_hook_data(mode, recipient, self._bridge.environment, secret_nonce)

    def build_private_mint_inputs(self, route: Route, attestation: Attestation, recipient: str, secret_nonce: str) -> list[str]:
        """The five ``private_mint`` literals (brief §3.5) after re-verifying hash and the (recipient, nonce) commitment."""
        self._validated(route, direction="mint")
        enc.validate_scalar(secret_nonce)
        if attestation.status != "complete":
            raise AttestationError("Private mint requires a completed Circle attestation; it is still pending")
        payload, signature, digest = bytes(attestation.payload), bytes(attestation.attestation), bytes(attestation.message_hash)
        if len(payload) != enc.XRESERVE_PAYLOAD_BYTES or len(signature) != enc.HOOK_DATA_BYTES or len(digest) != 32:
            raise AttestationError("Circle attestation has invalid widths (expected 305-byte payload, 65-byte signature, 32-byte hash)")
        if keccak256(payload) != digest:
            raise AttestationError("Circle attestation payload has an invalid message hash")
        expected_hook = self.hook_data("private", recipient, secret_nonce)
        if payload[-enc.HOOK_DATA_BYTES:] != expected_hook:
            raise AttestationError("Private mint secret nonce and recipient do not match the attested hook data")
        return [enc.u8_array_literal(payload), enc.u8_array_literal(signature), enc.u8_array_literal(digest), secret_nonce, recipient]

    def private_mint(self, attestation: Attestation, recipient: str, *, secret_nonce: str = "0scalar",
                     route: Route | None = None) -> AleoCall[MintReceipt]:
        """Finish a private-mode deposit: the only user-signed Aleo step of the inbound flow (``wrapper.private_mint``)."""
        route = self._validated(route if route is not None else self.inbound_route(), direction="mint")
        inputs = self.build_private_mint_inputs(route, attestation, recipient, secret_nonce)
        program = route.meta_str("wrapperProgram")
        message_hash = enc.to_hex(attestation.message_hash)
        nonce = enc.to_hex(enc.xreserve_nonce_from_payload(attestation.payload))

        def build(tx_id: str, _outputs: list[str]) -> MintReceipt:
            receipt = Receipt(id=message_hash, protocol="xreserve", status=Status.DESTINATION_CONFIRMING, destination_tx_id=tx_id,
                              protocol_state={"routeId": route.id, "mintMode": "private", "intendedRecipient": recipient,
                                              "messageHash": message_hash, "nonce": nonce,
                                              "bridgeProgram": route.meta_str("bridgeProgram"), "wrapperProgram": program,
                                              "destinationProgram": program, "destinationFunction": "private_mint"})
            return MintReceipt(transaction_id=tx_id, route_id=route.id, receipt=receipt)

        return self._bridge._call(program, "private_mint", inputs, build)

    # ── reads ──
    def get_attestation(self, message_hash: "str | bytes", *, route: Route | None = None) -> Attestation | None:
        """One Circle request for *message_hash*; ``None`` while pending (404)."""
        route = route if route is not None else self.inbound_route()
        client = CircleClient(route.meta_str("attestationBaseUrl"), session=self.circle_session)
        return client.get_attestation(enc.to_hex(enc.hex_to_bytes(message_hash)))

    def is_delivered(self, nonce: "str | bytes", *, route: Route | None = None) -> bool:
        """``bridgeProgram/nullifier[nonce as [u8; 32]] == true`` — the mint already landed on Aleo."""
        route = route if route is not None else self.inbound_route()
        try:
            raw = enc.hex_to_bytes(nonce, 32)
        except (ValueError, InvalidRecipientError) as exc:
            raise ConfigurationError("xReserve delivery requires a 32-byte deposit nonce") from exc
        value = self._bridge.mapping_value(route.meta_str("bridgeProgram"), "nullifier", enc.u8_array_literal(raw))
        return value is not None and value.strip() == "true"


__all__ = ["BURN_MODES", "ETHEREUM_DESTINATION_DOMAIN", "XReserveModule"]
