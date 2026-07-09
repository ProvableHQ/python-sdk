"""Tests for W4d2 wasm-parity additions."""
import re
import pytest
from aleo.mainnet import Proof, DynamicRecord, RecordPlaintext, Metadata, OfflineQuery


def test_proof_from_invalid_string():
    with pytest.raises(Exception):
        Proof.from_string("not_a_proof")


def test_proof_from_invalid_bytes():
    with pytest.raises(Exception):
        Proof.from_bytes(bytes(10))


CREDITS_RECORD_V1 = (
    "{ owner: aleo12a4wll9ax6w5355jph0dr5wt2vla5sss2t4cnch0tc3vzh643v8qcfvc7a.private, "
    "microcredits: 1000000u64.private, "
    "_nonce: 3634848344765318974603121890869676775499130077229666060613233255327643175219group.public, "
    "_version: 1u8.public }"
)
EXPECTED_OWNER = "aleo12a4wll9ax6w5355jph0dr5wt2vla5sss2t4cnch0tc3vzh643v8qcfvc7a"
EXPECTED_ROOT = "3632128850012040781624982531669233558118256132998845003264257146683715587370field"
EXPECTED_NONCE = "3634848344765318974603121890869676775499130077229666060613233255327643175219group"


def test_dynamic_record_from_record_round_trip():
    """KAT: DynamicRecord from RecordPlaintext and back."""
    record = RecordPlaintext.from_string(CREDITS_RECORD_V1)
    dr = DynamicRecord.from_record(record)
    assert str(dr.owner) == EXPECTED_OWNER
    assert str(dr.root) == EXPECTED_ROOT
    assert str(dr.nonce) == EXPECTED_NONCE
    assert dr.is_hiding() is True


def test_dynamic_record_string_round_trip():
    record = RecordPlaintext.from_string(CREDITS_RECORD_V1)
    dr = DynamicRecord.from_record(record)
    dr_str = str(dr)
    dr2 = DynamicRecord.from_string(dr_str)
    assert str(dr2) == dr_str


def test_dynamic_record_bytes_round_trip():
    record = RecordPlaintext.from_string(CREDITS_RECORD_V1)
    dr = DynamicRecord.from_record(record)
    raw = dr.bytes()
    dr2 = DynamicRecord.from_bytes(bytes(raw))
    assert str(dr2.owner) == EXPECTED_OWNER


def test_dynamic_record_to_record():
    record = RecordPlaintext.from_string(CREDITS_RECORD_V1)
    dr = DynamicRecord.from_record(record)
    back = dr.to_record(owner_is_private=True)
    assert back is not None


def test_dynamic_record_to_fields():
    record = RecordPlaintext.from_string(CREDITS_RECORD_V1)
    dr = DynamicRecord.from_record(record)
    fields = dr.to_fields()
    assert len(fields) > 0


def test_dynamic_record_to_bits_le():
    record = RecordPlaintext.from_string(CREDITS_RECORD_V1)
    dr = DynamicRecord.from_record(record)
    bits = dr.to_bits_le()
    assert len(bits) > 0


def test_dynamic_record_invalid():
    with pytest.raises(Exception):
        DynamicRecord.from_string("not a record")


def test_metadata_bond_public():
    m = Metadata.bond_public()
    assert m.name == "bond_public"
    assert m.locator == "credits.aleo/bond_public"
    assert m.verifying_key == "bondPublicVerifier"
    assert "bond_public.prover." in m.prover
    assert "bond_public.verifier." in m.verifier
    assert "parameters.provable.com/mainnet" in m.prover
    assert re.match(r"^bond_public\.verifier\.[0-9a-f]{7}$", m.verifier)


def test_metadata_transfer_public():
    m = Metadata.transfer_public()
    assert m.name == "transfer_public"
    assert m.locator == "credits.aleo/transfer_public"


def test_metadata_inclusion():
    m = Metadata.inclusion()
    assert m.name == "inclusion"
    assert m.locator == "inclusion"


def test_metadata_base_url():
    assert "parameters.provable.com" in Metadata.base_url()


OFFLINE_QUERY_KAT = '{"block_height":0,"state_paths":{},"state_root":"sr1wjueje6hy86yw9j4lhl7jwvhjxwunw34paj4k3cn2wm5h5r2syfqd83yw4"}'
STATE_ROOT = "sr1wjueje6hy86yw9j4lhl7jwvhjxwunw34paj4k3cn2wm5h5r2syfqd83yw4"


def test_offline_query_kat():
    oq = OfflineQuery.new(0, STATE_ROOT)
    assert str(oq) == OFFLINE_QUERY_KAT


def test_offline_query_round_trip():
    oq = OfflineQuery.new(0, STATE_ROOT)
    oq2 = OfflineQuery.from_string(str(oq))
    assert str(oq2) == str(oq)


def test_offline_query_add_block_height():
    oq = OfflineQuery.new(0, STATE_ROOT)
    oq.add_block_height(42)
    assert '"block_height":42' in str(oq)


# ---------------------------------------------------------------------------
# Task 6: Program introspection methods
# ---------------------------------------------------------------------------
from aleo.mainnet import Program, Address  # noqa: E402

# The credits.aleo program address for this snarkvm version.
CREDITS_ADDRESS = "aleo1lqmly7ez2k48ajf5hs92ulphaqr05qm4n8qwzj8v0yprmasgpqgsez59gg"


def test_program_has_function():
    p = Program.credits()
    assert p.has_function("transfer_public") is True
    assert p.has_function("nonexistent_fn") is False


def test_program_get_function_inputs_transfer_public():
    """KAT: exact dict shape for credits.aleo/transfer_public.
    transfer_public takes public inputs (receiver address and amount).
    """
    p = Program.credits()
    inputs = p.get_function_inputs("transfer_public")
    assert len(inputs) == 2
    assert inputs[0]["type"] == "address"
    assert inputs[0]["visibility"] == "public"
    assert inputs[0]["register"] == "r0"
    assert inputs[1]["type"] == "u64"
    assert inputs[1]["visibility"] == "public"
    assert inputs[1]["register"] == "r1"


def test_program_get_mappings():
    p = Program.credits()
    mappings = p.get_mappings()
    names = [m["name"] for m in mappings]
    for expected in ["account", "bonded", "committee"]:
        assert expected in names
    for m in mappings:
        assert "name" in m
        assert "key_type" in m
        assert "value_type" in m

    # Additional assertions for account mapping values
    account_mapping = next((m for m in mappings if m["name"] == "account"), None)
    assert account_mapping is not None
    assert account_mapping["key_type"] == "address"
    assert account_mapping["value_type"] == "u64"


def test_program_get_record_members():
    p = Program.credits()
    rec = p.get_record_members("credits")
    assert rec["type"] == "record"
    assert rec["record"] == "credits"
    members = rec["members"]
    names = [m["name"] for m in members]
    assert "microcredits" in names
    assert "_nonce" in names

    # Additional assertions for member shape
    microcredits_member = next((m for m in members if m["name"] == "microcredits"), None)
    assert microcredits_member is not None
    assert microcredits_member["type"] == "u64"
    assert microcredits_member["visibility"] == "private"

    nonce_member = next((m for m in members if m["name"] == "_nonce"), None)
    assert nonce_member is not None
    assert nonce_member["type"] == "group"
    assert nonce_member["visibility"] == "public"


def test_program_get_struct_members():
    # Use a program that has structs
    src = """program test_struct.aleo;
struct point:
    x as u32;
    y as u32;
function noop:
    input r0 as u32.public;
    output r0 as u32.public;
"""
    p = Program.from_source(src)
    members = p.get_struct_members("point")
    assert isinstance(members, list)
    names = [m["name"] for m in members]
    assert "x" in names
    assert "y" in names

    # Additional assertions for full dict shape
    for member in members:
        assert set(member.keys()) == {"name", "type"}
        if member["name"] == "x":
            assert member["type"] == "u32"
        elif member["name"] == "y":
            assert member["type"] == "u32"


def test_program_get_function_inputs_record():
    """KAT: exact dict shape for credits.aleo/transfer_private.
    transfer_private takes a record input (credits) and public/private field inputs.
    This mirrors the wasm test_get_inputs KAT for transfer_private.
    """
    p = Program.credits()
    inputs = p.get_function_inputs("transfer_private")
    assert len(inputs) == 3

    # Input 0: record type
    assert inputs[0]["type"] == "record"
    assert inputs[0]["record"] == "credits"
    assert inputs[0]["register"] == "r0"
    assert "members" in inputs[0]

    # Check record members
    members = inputs[0]["members"]
    microcredits = next((m for m in members if m["name"] == "microcredits"), None)
    assert microcredits is not None
    assert microcredits["type"] == "u64"
    assert microcredits["visibility"] == "private"

    nonce = next((m for m in members if m["name"] == "_nonce"), None)
    assert nonce is not None
    assert nonce["type"] == "group"
    assert nonce["visibility"] == "public"

    # Input 1: address (private)
    assert inputs[1]["type"] == "address"
    assert inputs[1]["visibility"] == "private"
    assert inputs[1]["register"] == "r1"

    # Input 2: u64 (private)
    assert inputs[2]["type"] == "u64"
    assert inputs[2]["visibility"] == "private"
    assert inputs[2]["register"] == "r2"


def test_program_address():
    p = Program.credits()
    addr = p.address()
    assert str(addr) == CREDITS_ADDRESS


# ---------------------------------------------------------------------------
# Task 7: VerifyingKey additions (checksum, num_constraints, credits statics, is_*_verifier)
# ---------------------------------------------------------------------------
from aleo.mainnet import VerifyingKey  # noqa: E402

# KAT from wasm tests (mainnet) — sha256 of the raw verifier bytes / to_bytes_le() round-trip
TRANSFER_PUBLIC_VK_CHECKSUM = "ea77f42a35b3f891e7753c7333df365f356883550c4602df11f270237bef340d"
TRANSFER_PUBLIC_NUM_CONSTRAINTS = 12326

# All 16 (function_name, getter_name) pairs for credits + inclusion
CREDITS_VERIFIER_GETTERS = [
    ("bond_public", "bond_public_verifier"),
    ("bond_validator", "bond_validator_verifier"),
    ("claim_unbond_public", "claim_unbond_public_verifier"),
    ("fee_private", "fee_private_verifier"),
    ("fee_public", "fee_public_verifier"),
    ("join", "join_verifier"),
    ("set_validator_state", "set_validator_state_verifier"),
    ("split", "split_verifier"),
    ("transfer_private", "transfer_private_verifier"),
    ("transfer_private_to_public", "transfer_private_to_public_verifier"),
    ("transfer_public", "transfer_public_verifier"),
    ("transfer_public_as_signer", "transfer_public_as_signer_verifier"),
    ("transfer_public_to_private", "transfer_public_to_private_verifier"),
    ("unbond_public", "unbond_public_verifier"),
    # inclusion
    (None, "inclusion_verifier"),
]


def test_verifying_key_checksum():
    vk = VerifyingKey.transfer_public_verifier()
    assert vk.checksum() == TRANSFER_PUBLIC_VK_CHECKSUM


def test_verifying_key_num_constraints():
    vk = VerifyingKey.transfer_public_verifier()
    assert vk.num_constraints() == TRANSFER_PUBLIC_NUM_CONSTRAINTS


def test_verifying_key_bond_public_verifier_is_checker():
    bond = VerifyingKey.bond_public_verifier()
    transfer = VerifyingKey.transfer_public_verifier()
    assert bond.is_bond_public_verifier() is True
    assert transfer.is_bond_public_verifier() is False


def test_verifying_key_inclusion_verifier():
    vk = VerifyingKey.inclusion_verifier()
    assert vk.is_inclusion_verifier() is True
    assert vk.is_bond_public_verifier() is False


def test_all_16_verifier_getters_and_is_checkers():
    """Iterate all 16 getters and verify each is_* self-check returns True."""
    IS_CHECKER_MAP = {
        "bond_public_verifier": "is_bond_public_verifier",
        "bond_validator_verifier": "is_bond_validator_verifier",
        "claim_unbond_public_verifier": "is_claim_unbond_public_verifier",
        "fee_private_verifier": "is_fee_private_verifier",
        "fee_public_verifier": "is_fee_public_verifier",
        "join_verifier": "is_join_verifier",
        "set_validator_state_verifier": "is_set_validator_state_verifier",
        "split_verifier": "is_split_verifier",
        "transfer_private_verifier": "is_transfer_private_verifier",
        "transfer_private_to_public_verifier": "is_transfer_private_to_public_verifier",
        "transfer_public_verifier": "is_transfer_public_verifier",
        "transfer_public_as_signer_verifier": "is_transfer_public_as_signer_verifier",
        "transfer_public_to_private_verifier": "is_transfer_public_to_private_verifier",
        "unbond_public_verifier": "is_unbond_public_verifier",
        "inclusion_verifier": "is_inclusion_verifier",
    }
    for _fn_name, getter in CREDITS_VERIFIER_GETTERS:
        vk = getattr(VerifyingKey, getter)()
        is_checker = IS_CHECKER_MAP[getter]
        result = getattr(vk, is_checker)()
        assert result is True, f"{getter}: {is_checker}() returned {result}, expected True"
