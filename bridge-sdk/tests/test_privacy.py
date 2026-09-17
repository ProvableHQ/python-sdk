import pytest

from aleo_bridge.errors import ConfigurationError, InsufficientBalanceError, InvalidAmountError, InvalidRecipientError, UnsupportedRouteError
from aleo_bridge.freezelist import EMPTY_MERKLE_PROOF_PAIR
from aleo_bridge.privacy import record_amount
from aleo_bridge.types import PrivacyReceipt
from tests.conftest import SIGNER, USDCX_RECORD, USDCX_RECORD_SMALL

OTHER = "aleo1kypwp5m7qtk9mwazgcpg0tq8aal23mnrvwfvug65qgcg9xvsrqgspyjm6n"
SOL_RECORD = f"{{ owner: {SIGNER}.private, amount: 300000000u128.private, _nonce: 9group.public }}"
SOL_RECORD_BIG = f"{{ owner: {SIGNER}.private, amount: 900000000u128.private, _nonce: 10group.public }}"


def test_record_amount_parser():
    assert record_amount(USDCX_RECORD) == 5_000_000 and record_amount(USDCX_RECORD_SMALL) == 100
    assert record_amount("{ owner: aleo1x.private, _nonce: 1group.public }") is None
    assert record_amount("") is None


def test_shield_arc20_matches_veil(bridge):
    call = bridge.privacy.shield("aleo/eth", amount="0.000000000000000001")
    assert (call.program_id, call.function_name, call.inputs) == ("arc20_eth.aleo", "shield", ["1u128"])
    result = call.transact()
    assert result == PrivacyReceipt("at1built", "aleo/eth", "0.000000000000000001", 1, "shield")


def test_shield_arc22_names_recipient(bridge):
    call = bridge.privacy.shield(("aleo", "usdcx"), amount="2.5")
    assert (call.program_id, call.function_name, call.inputs) == ("usdcx_stablecoin.aleo", "transfer_public_to_private", [SIGNER, "2500000u128"])
    assert bridge.privacy.shield("aleo/usdcx", amount_atomic=2_500_000, recipient=OTHER).inputs == [OTHER, "2500000u128"]
    with pytest.raises(ConfigurationError, match="ARC-20 shield always credits the caller"):
        bridge.privacy.shield("aleo/eth", amount="1", recipient=OTHER)
    with pytest.raises(InvalidRecipientError):
        bridge.privacy.shield("aleo/usdcx", amount="1", recipient="0x1234")


def test_unshield_arc20_selects_smallest_covering_record(bridge):
    bridge.aleo.record_rows = [{"program": "arc20_sol.aleo", "record_plaintext": SOL_RECORD_BIG},
                               {"program": "arc20_sol.aleo", "record_plaintext": SOL_RECORD},
                               {"program": "usdcx_stablecoin.aleo", "record_plaintext": USDCX_RECORD}]
    call = bridge.privacy.unshield("aleo/sol", amount="0.25")
    assert (call.program_id, call.function_name, call.inputs) == ("arc20_sol.aleo", "unshield", [SOL_RECORD, "250000000u128"])
    assert bridge.aleo.record_queries[-1] == {"program": "arc20_sol.aleo", "record": "Token", "unspent": True}
    result = call.delegate(wait=False)
    assert result == PrivacyReceipt("at1delegated", "aleo/sol", "0.25", 250_000_000, "unshield")
    with pytest.raises(ConfigurationError, match="ARC-20 unshield takes no Merkle proof"):
        bridge.privacy.unshield("aleo/sol", amount="0.25", merkle_proof="[x]")


def test_unshield_arc22_defaults_to_signer_record_and_empty_proof(bridge):
    call = bridge.privacy.unshield("aleo/usdcx", amount="2.5")
    assert (call.program_id, call.function_name) == ("usdcx_stablecoin.aleo", "transfer_private_to_public")
    assert call.inputs[:3] == [SIGNER, "2500000u128", USDCX_RECORD]
    assert call.inputs[3] == EMPTY_MERKLE_PROOF_PAIR and call.inputs[3].count("0field") == 32


def test_unshield_arc22_accepts_explicit_inputs(bridge):
    call = bridge.privacy.unshield("aleo/usdcx", amount="2.5", recipient=OTHER, record=USDCX_RECORD_SMALL, merkle_proof="[custom-proof]")
    assert call.inputs == [OTHER, "2500000u128", USDCX_RECORD_SMALL, "[custom-proof]"]
    assert bridge.aleo.record_queries == []          # explicit record: no scanner query


def test_unsupported_assets_and_zero_amounts(bridge):
    with pytest.raises(UnsupportedRouteError, match="does not support shielding"):
        bridge.privacy.shield("aleo/aleo", amount="1")
    with pytest.raises(UnsupportedRouteError, match="does not support unshielding"):
        bridge.privacy.unshield("aleo/aleo", amount="1")
    with pytest.raises(UnsupportedRouteError, match="does not support shielding"):
        bridge.privacy.shield("ethereum/usdc", amount="1")
    with pytest.raises(InvalidAmountError, match="Unshielding amount must be greater than zero"):
        bridge.privacy.unshield("aleo/sol", amount="0")
    with pytest.raises(InvalidAmountError, match="Shielding amount must be greater than zero"):
        bridge.privacy.shield("aleo/sol", amount_atomic=0)


def test_select_record_reports_largest_available(bridge):
    bridge.aleo.record_rows = [{"program": "usdcx_stablecoin.aleo", "record_plaintext": USDCX_RECORD_SMALL}]
    with pytest.raises(InsufficientBalanceError, match="largest available is 100"):
        bridge.privacy.select_record("usdcx_stablecoin.aleo", 2_500_000)
    bridge.aleo.record_rows = []
    with pytest.raises(InsufficientBalanceError, match="largest available is 0"):
        bridge.privacy.select_record("usdcx_stablecoin.aleo", 1)


def test_private_burn_defaults_resolve_through_privacy_and_freezelist(bridge):
    ONE_LIT = "[" + ",".join(["0u8"] * 31 + ["1u8"]) + "]"
    call = bridge.xreserve.burn("0x0000000000000000000000000000000000000001", amount="2.5")
    assert call.inputs == [USDCX_RECORD, "2500000u128", "0u32", ONE_LIT, EMPTY_MERKLE_PROOF_PAIR]
    assert bridge.aleo.record_queries[-1]["program"] == "usdcx_stablecoin.aleo"
