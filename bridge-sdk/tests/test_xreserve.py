import pytest

from aleo_bridge import encoding as enc
from aleo_bridge.errors import (AttestationError, ConfigurationError, InvalidAmountError, InvalidRecipientError,
                                UnsupportedRouteError)
from aleo_bridge.registry import DEFAULT_REGISTRY as REG
from aleo_bridge.types import Attestation, BurnReceipt, MintReceipt, Status
from tests.conftest import NULLIFIED_NONCE, USDCX_RECORD

EVM1 = "0x0000000000000000000000000000000000000001"
RECIPIENT = "aleo1kypwp5m7qtk9mwazgcpg0tq8aal23mnrvwfvug65qgcg9xvsrqgspyjm6n"
PROOF = "[{ siblings: [0field], leaf_index: 1u32 }, { siblings: [0field], leaf_index: 1u32 }]"
ONE_LIT = "[" + ",".join(["0u8"] * 31 + ["1u8"]) + "]"
MAIN = REG.route("xreserve:aleo/usdcx->ethereum/usdc")
TESTNET = REG.route("xreserve:aleo-testnet/usdcx->sepolia/usdc")
INBOUND = REG.route("xreserve:ethereum/usdc->aleo/usdcx")


def test_routes(bridge):
    assert bridge.xreserve.outbound_route().id == "xreserve:aleo/usdcx->ethereum/usdc"
    assert bridge.xreserve.inbound_route().id == "xreserve:ethereum/usdc->aleo/usdcx"


def test_private_burn_inputs_match_veil(bridge):
    program, function, inputs = bridge.xreserve.build_burn_inputs(
        MAIN, mode="private", amount_atomic=2_500_000, recipient=EVM1, record=USDCX_RECORD, merkle_proof=PROOF)
    assert (program, function) == ("shielded_usdcx_wrapper.aleo", "private_burn")
    assert inputs == [USDCX_RECORD, "2500000u128", "0u32", ONE_LIT, PROOF]
    program, function, inputs = bridge.xreserve.build_burn_inputs(
        TESTNET, mode="private", amount_atomic=2_500_000, recipient=EVM1, record=USDCX_RECORD, merkle_proof=PROOF)
    assert program == "shielded_usdcx_wrapper.aleo" and inputs[0] == USDCX_RECORD


def test_public_burn_inputs(bridge):
    assert bridge.xreserve.build_burn_inputs(MAIN, mode="public", amount_atomic=2_500_000, recipient=EVM1, record=None, merkle_proof=None) == \
        ("usdcx_bridge_v2.aleo", "burn_public", ["2500000u128", "0u32", ONE_LIT])
    assert bridge.xreserve.build_burn_inputs(MAIN, mode="public-as-signer", amount_atomic=2_500_000, recipient=EVM1, record=None, merkle_proof=None)[1] == "burn_public_as_signer"
    assert bridge.xreserve.build_burn_inputs(TESTNET, mode="public", amount_atomic=2_500_000, recipient=EVM1, record=None, merkle_proof=None)[0] == "test_usdcx_bridge_v2.aleo"


def test_burn_input_validation(bridge):
    kw = dict(amount_atomic=2_500_000, recipient=EVM1, record=USDCX_RECORD, merkle_proof=PROOF)
    with pytest.raises(ConfigurationError, match="Unsupported USDCx burn mode"):
        bridge.xreserve.build_burn_inputs(MAIN, mode="unknown", **kw)
    with pytest.raises(ConfigurationError, match="private_burn requires a USDCx Token record"):
        bridge.xreserve.build_burn_inputs(MAIN, mode="private", **{**kw, "record": None})
    with pytest.raises(ConfigurationError, match=r"\[MerkleProof; 2\]"):
        bridge.xreserve.build_burn_inputs(MAIN, mode="private", **{**kw, "merkle_proof": "not-a-literal"})
    with pytest.raises(InvalidAmountError, match="must exceed the 2 USDCx withdrawal fee"):
        bridge.xreserve.build_burn_inputs(MAIN, mode="public", **{**kw, "amount_atomic": 2_000_000})
    with pytest.raises(InvalidAmountError, match="greater than zero"):
        bridge.xreserve.build_burn_inputs(MAIN, mode="public", **{**kw, "amount_atomic": 0})
    with pytest.raises(InvalidRecipientError):
        bridge.xreserve.build_burn_inputs(MAIN, mode="public", **{**kw, "recipient": "0x1234"})
    with pytest.raises(UnsupportedRouteError, match="Aleo-to-Ethereum"):
        bridge.xreserve.build_burn_inputs(INBOUND, mode="public", **{**kw, "recipient": RECIPIENT})


def test_burn_builds_call_and_receipt(bridge):
    call = bridge.xreserve.burn(EVM1, amount="2.5", mode="private", record=USDCX_RECORD, merkle_proof=PROOF)
    assert (call.program_id, call.function_name) == ("shielded_usdcx_wrapper.aleo", "private_burn")
    assert call.inputs == [USDCX_RECORD, "2500000u128", "0u32", ONE_LIT, PROOF]
    result = call.transact()
    assert isinstance(result, BurnReceipt)
    assert (result.transaction_id, result.route_id, result.mode, result.amount_atomic) == ("at1built", MAIN.id, "private", 2_500_000)
    assert result.receipt.status is Status.SOURCE_CONFIRMING and result.receipt.source_tx_id == "at1built"
    assert result.receipt.protocol_state == {
        "routeId": MAIN.id, "burnMode": "private", "amountAtomic": "2500000", "nativeDomain": 0,
        "nativeRecipientBytes32": "0x" + "00" * 31 + "01", "sourceProgram": "shielded_usdcx_wrapper.aleo",
        "sourceFunction": "private_burn", "forwardingService": "aleo-burn-attestation"}
    public = bridge.xreserve.burn(EVM1, amount_atomic=3_000_000, mode="public-as-signer")
    assert (public.program_id, public.function_name, public.inputs) == ("usdcx_bridge_v2.aleo", "burn_public_as_signer", ["3000000u128", "0u32", ONE_LIT])


def _attested(bridge, nonce="7scalar", recipient=RECIPIENT) -> Attestation:
    hook = bridge.xreserve.hook_data("private", recipient, nonce)
    payload = bytes.fromhex("5a2e0acd00000001") + bytes(228) + bytes.fromhex("00000041") + hook
    return Attestation(payload=payload, message_hash=enc.xreserve_message_hash(payload), attestation=bytes.fromhex("11" * 65), status="complete")


def test_hook_data_uses_bridge_environment(bridge):
    assert bridge.xreserve.hook_data("public", RECIPIENT) == bytes(65)
    assert bridge.xreserve.hook_data("private", RECIPIENT, "7scalar") == enc.xreserve_hook_data("private", RECIPIENT, "mainnet", "7scalar")


def test_private_mint_inputs_match_veil(bridge):
    att = _attested(bridge)
    inputs = bridge.xreserve.build_private_mint_inputs(INBOUND, att, RECIPIENT, "7scalar")
    assert len(inputs) == 5
    assert inputs[0] == enc.u8_array_literal(att.payload) and inputs[0].count("u8") == 305 and " " not in inputs[0]
    assert inputs[1] == "[" + ",".join(["17u8"] * 65) + "]"
    assert inputs[2] == enc.u8_array_literal(att.message_hash) and inputs[2].count("u8") == 32
    assert inputs[3] == "7scalar" and inputs[4] == RECIPIENT


def test_private_mint_rejections(bridge):
    att = _attested(bridge)
    with pytest.raises(AttestationError, match="do not match the attested hook"):
        bridge.xreserve.build_private_mint_inputs(INBOUND, att, RECIPIENT, "8scalar")
    with pytest.raises(AttestationError, match="do not match the attested hook"):
        bridge.xreserve.build_private_mint_inputs(INBOUND, att, "aleo1rhgdu77hgyqd3xjj8ucu3jj9r2krwz6mnzyd80gncr5fxcwlh5rsvzp9px", "7scalar")
    with pytest.raises(AttestationError, match="completed Circle attestation"):
        bridge.xreserve.build_private_mint_inputs(INBOUND, Attestation(att.payload, att.message_hash, att.attestation, "pending"), RECIPIENT, "7scalar")
    with pytest.raises(AttestationError, match="invalid message hash"):
        bridge.xreserve.build_private_mint_inputs(INBOUND, Attestation(att.payload, bytes(32), att.attestation, "complete"), RECIPIENT, "7scalar")
    with pytest.raises(ConfigurationError, match="scalar"):
        bridge.xreserve.build_private_mint_inputs(INBOUND, att, RECIPIENT, "seven")
    with pytest.raises(UnsupportedRouteError, match="Ethereum-to-Aleo"):
        bridge.xreserve.build_private_mint_inputs(MAIN, att, RECIPIENT, "7scalar")


def test_private_mint_call_and_receipt(bridge):
    att = _attested(bridge)
    call = bridge.xreserve.private_mint(att, RECIPIENT, secret_nonce="7scalar")
    assert (call.program_id, call.function_name) == ("shielded_usdcx_wrapper.aleo", "private_mint")
    result = call.delegate(wait=False)
    assert isinstance(result, MintReceipt) and result.transaction_id == "at1delegated" and result.route_id == INBOUND.id
    r = result.receipt
    assert r.id == enc.to_hex(att.message_hash) and r.status is Status.DESTINATION_CONFIRMING and r.destination_tx_id == "at1delegated"
    assert r.protocol_state["routeId"] == INBOUND.id and r.protocol_state["mintMode"] == "private"
    assert r.protocol_state["intendedRecipient"] == RECIPIENT and r.protocol_state["nonce"] == "0x" + "00" * 32
    assert r.protocol_state["destinationProgram"] == "shielded_usdcx_wrapper.aleo" and r.protocol_state["destinationFunction"] == "private_mint"
    assert "secretNonce" not in r.protocol_state          # the secret never travels in receipts


def test_get_attestation_uses_route_base_url(bridge):
    class _Session:
        def __init__(self): self.urls = []
        def get(self, url, timeout=None):
            self.urls.append(url)
            class R: status_code = 404
            return R()
    session = _Session()
    bridge.xreserve.circle_session = session
    assert bridge.xreserve.get_attestation("0x" + "22" * 32) is None
    assert session.urls == ["https://xreserve-api.circle.com/v1/attestations/0x" + "22" * 32]
    bridge.xreserve.get_attestation(bytes.fromhex("33" * 32), route=REG.route("xreserve:sepolia/usdc->aleo-testnet/usdcx"))
    assert session.urls[-1].startswith("https://xreserve-api-testnet.circle.com/v1/attestations/0x33")


def test_is_delivered_reads_bridge_program_nullifier(bridge):
    assert bridge.xreserve.is_delivered(NULLIFIED_NONCE) is True
    assert bridge.xreserve.is_delivered("0x" + NULLIFIED_NONCE.hex()) is True
    assert bridge.xreserve.is_delivered(bytes(32)) is False
    assert "usdcx_bridge_v2.aleo" in bridge.aleo.fetched
    with pytest.raises(ConfigurationError, match="32-byte deposit nonce"):
        bridge.xreserve.is_delivered("0x01")
