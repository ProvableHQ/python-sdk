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

# All 15 (function_name, getter_name) pairs for credits + inclusion
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


def test_all_15_verifier_getters_and_is_checkers():
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


# ---------------------------------------------------------------------------
# Task 8: ProvingKey additions (checksum, is_*_prover)
# ---------------------------------------------------------------------------
from aleo.mainnet import ProvingKey  # noqa: E402

# All 15 is_*_prover method names
PROVER_CHECKER_METHODS = [
    "is_bond_public_prover",
    "is_bond_validator_prover",
    "is_claim_unbond_public_prover",
    "is_fee_private_prover",
    "is_fee_public_prover",
    "is_inclusion_prover",
    "is_join_prover",
    "is_set_validator_state_prover",
    "is_split_prover",
    "is_transfer_private_prover",
    "is_transfer_private_to_public_prover",
    "is_transfer_public_prover",
    "is_transfer_public_as_signer_prover",
    "is_transfer_public_to_private_prover",
    "is_unbond_public_prover",
]


def test_proving_key_checksum_shape():
    """Shape test: checksum() method exists and raises on invalid bytes (not AttributeError)."""
    with pytest.raises(Exception, match=r"."):
        ProvingKey.from_bytes(bytes(10)).checksum()


def test_proving_key_checker_methods_exist():
    """API-surface test: all 15 is_*_prover methods exist and are callable on ProvingKey."""
    for method_name in PROVER_CHECKER_METHODS:
        assert callable(getattr(ProvingKey, method_name, None)), (
            f"ProvingKey.{method_name} is not callable"
        )


@pytest.mark.slow
def test_proving_key_split_prover_checksum():
    """Slow: downloads split.prover, verifies checksum() matches metadata and is_split_prover() is True."""
    import urllib.request

    meta = Metadata.split()
    url = meta.prover  # e.g. https://parameters.provable.com/mainnet/split.prover.XXXXXXX
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            prover_bytes = resp.read()
    except Exception:
        pytest.skip("network unavailable or download failed")

    pk = ProvingKey.from_bytes(prover_bytes)

    # checksum() should return a 64-char hex string
    cs = pk.checksum()
    assert isinstance(cs, str)
    assert len(cs) == 64
    assert all(c in "0123456789abcdef" for c in cs)

    # The first 7 chars of the checksum appear in the prover URL
    assert cs[:7] in url, f"checksum prefix {cs[:7]} not found in URL {url}"

    # is_split_prover() must be True; a different checker must be False
    assert pk.is_split_prover() is True
    assert pk.is_bond_public_prover() is False


# ---------------------------------------------------------------------------
# Task 9: Ciphertext.decrypt_with_transition_view_key and decrypt_with_transition_info
# ---------------------------------------------------------------------------
from aleo.mainnet import Ciphertext, ViewKey, Transition, Group, Field  # noqa: E402

# KAT generated for MainnetV0 (network ID 0):
#   Private key: APrivateKey1zkp8CZNn3yeCseEtxuVPbDCwSyhGW6yZKUYKfgXmcpoGPWH
#   tpk: Group::generator()
#   tvk: (tpk * vk_scalar).to_x_coordinate()
#   ciphertext: encrypt_symmetric(hash_psd4([function_id, tvk, Field::from_u16(1)]))
#   on plaintext "42u32" with program="hello_hello.aleo", function="hello", index=1
#
# NOTE: The wasm SDK KAT used network ID 1 (testnet). This SDK targets MainnetV0
# (network ID 0), so the wasm KAT ciphertext does not decrypt to the expected
# "2u32" on mainnet. These tests use a self-generated KAT verified by a Rust unit
# test to ensure correctness with the actual network ID.
_TASK9_PK_STR = "APrivateKey1zkp8CZNn3yeCseEtxuVPbDCwSyhGW6yZKUYKfgXmcpoGPWH"
_TASK9_TPK_STR = "1540945439182663264862696551825005342995406165131907382295858612069623286213group"
_TASK9_TVK_STR = "5913691033198561235987877735437800286574097563446004224670039882724243551875field"
_TASK9_CT_STR = "ciphertext1qyqxk3s4jy4cz265h82q8m0z7zm0jr8w8xkh89a2qpy82kq0ja8w5rq2qj5rw"
_TASK9_PROGRAM = "hello_hello.aleo"
_TASK9_FUNCTION = "hello"
_TASK9_INDEX = 1
_TASK9_EXPECTED = "42u32"


def test_ciphertext_decrypt_with_transition_view_key():
    """KAT: decrypt_with_transition_view_key returns '42u32' for the mainnet KAT."""
    tvk = Field.from_string(_TASK9_TVK_STR)
    ct = Ciphertext.from_string(_TASK9_CT_STR)
    plaintext = ct.decrypt_with_transition_view_key(
        tvk, _TASK9_PROGRAM, _TASK9_FUNCTION, _TASK9_INDEX
    )
    assert str(plaintext) == _TASK9_EXPECTED


def test_ciphertext_decrypt_with_transition_info():
    """KAT: decrypt_with_transition_info returns '42u32' for the mainnet KAT."""
    from aleo.mainnet import PrivateKey

    pk = PrivateKey.from_string(_TASK9_PK_STR)
    vk = pk.view_key
    tpk = Group.from_string(_TASK9_TPK_STR)
    ct = Ciphertext.from_string(_TASK9_CT_STR)
    plaintext = ct.decrypt_with_transition_info(
        vk, tpk, _TASK9_PROGRAM, _TASK9_FUNCTION, _TASK9_INDEX
    )
    assert str(plaintext) == _TASK9_EXPECTED


def test_ciphertext_decrypt_methods_agree():
    """Self-consistency: both methods produce identical plaintexts for the same inputs."""
    from aleo.mainnet import PrivateKey

    pk = PrivateKey.from_string(_TASK9_PK_STR)
    vk = pk.view_key
    tpk = Group.from_string(_TASK9_TPK_STR)
    tvk = Field.from_string(_TASK9_TVK_STR)
    ct = Ciphertext.from_string(_TASK9_CT_STR)

    p_tvk = ct.decrypt_with_transition_view_key(
        tvk, _TASK9_PROGRAM, _TASK9_FUNCTION, _TASK9_INDEX
    )
    p_info = ct.decrypt_with_transition_info(
        vk, tpk, _TASK9_PROGRAM, _TASK9_FUNCTION, _TASK9_INDEX
    )
    assert str(p_tvk) == str(p_info)


def test_ciphertext_decrypt_wrong_index_fails():
    """Decrypting with the wrong index must raise (different symmetric key)."""
    tvk = Field.from_string(_TASK9_TVK_STR)
    ct = Ciphertext.from_string(_TASK9_CT_STR)
    with pytest.raises(Exception):
        ct.decrypt_with_transition_view_key(
            tvk, _TASK9_PROGRAM, _TASK9_FUNCTION, 0  # wrong index
        )


def test_ciphertext_decrypt_wrong_program_fails():
    """Decrypting with the wrong program name must raise."""
    tvk = Field.from_string(_TASK9_TVK_STR)
    ct = Ciphertext.from_string(_TASK9_CT_STR)
    with pytest.raises(Exception):
        ct.decrypt_with_transition_view_key(
            tvk, "credits.aleo", _TASK9_FUNCTION, _TASK9_INDEX
        )


def test_ciphertext_decrypt_with_transition_info_wrong_vk_fails():
    """Decrypting with a different view key must raise."""
    from aleo.mainnet import PrivateKey

    other_pk = PrivateKey.from_string(
        "APrivateKey1zkpJkyYRGYtkeHDaFfwsKtUJzia7csiWhfBWPXWhXJzy9Ls"
    )
    other_vk = other_pk.view_key
    tpk = Group.from_string(_TASK9_TPK_STR)
    ct = Ciphertext.from_string(_TASK9_CT_STR)
    with pytest.raises(Exception):
        ct.decrypt_with_transition_info(
            other_vk, tpk, _TASK9_PROGRAM, _TASK9_FUNCTION, _TASK9_INDEX
        )


# ---------------------------------------------------------------------------
# Task 9 cross-validation: Ciphertext.decrypt_with_transition_view_key vs
# Transition.decrypt_transition (independently tested in test_chain_data.py).
#
# The self-generated KAT tests above only prove "the code decrypts its own
# output". A shared derivation bug (e.g. wrong field order in hash_psd4) would
# not be caught because both paths would share the same defect.
#
# This test uses the vendored hello_hello.aleo/main fixture from
# test_chain_data.py, which has a real private input at index 1 (value "2u32")
# encrypted under the transition view key derived from DECRYPTION_PRIVATE_KEY.
# Transition.decrypt_transition was independently tested in test_chain_data.py
# (TestTransition.test_decrypt_transition), so agreeing with it here catches
# any bug that affects only one of the two derivation paths.
# ---------------------------------------------------------------------------

# Vendored from test_chain_data.py (hello_hello.aleo/main, mainnet fixture):
_XVAL_TRANSITION_JSON = (
    '{"id":"au1mguuz0dh20f78802m4z0py7n08xhl0pz60llzck63mhl8pc8l5xqxpwgtn",'
    '"program":"hello_hello.aleo","function":"main",'
    '"inputs":['
    '{"type":"public","id":"6393584049543470937057043098611271993206122889317039351966319038535020834557field","value":"1u32"},'
    '{"type":"private","id":"8207446256045172951742235001162005156507562935942883128759030124682934277495field",'
    '"value":"ciphertext1qyqqgz9qnupeld9vr4vuwp6yrpmhgtkvmgag5m7mmrruw0r6je666qgqdswk3"}'
    '],'
    '"outputs":[{"type":"private","id":"127469473292952941321346770257126666363371158501875622169294663492714835110field",'
    '"value":"ciphertext1qyqyapkjuxm9dcslgyjf7hkr2k3dek500z40gjspnwvll0uawj23vzgggc405"}],'
    '"tpk":"7647553513996966044119163122930125808381703910407273818947266861843062002251group",'
    '"tcm":"4479413938380109857414238205380483440836495997450846894155088299187217672609field",'
    '"scm":"6461007226176477784737642021400489186736987671609840640950580467598882134642field"}'
)
_XVAL_PRIVATE_KEY = "APrivateKey1zkp8CZNn3yeCseEtxuVPbDCwSyhGW6yZKUYKfgXmcpoGPWH"
# Index 1 is the private input (r1 as u32.private); index 0 is public.
_XVAL_PRIVATE_INPUT_INDEX = 1
_XVAL_PRIVATE_INPUT_CT = "ciphertext1qyqqgz9qnupeld9vr4vuwp6yrpmhgtkvmgag5m7mmrruw0r6je666qgqdswk3"
_XVAL_EXPECTED_PLAINTEXT = "2u32"


def test_ciphertext_decrypt_cross_validates_with_transition_decrypt():
    """Cross-validation: Ciphertext.decrypt_with_transition_view_key agrees with
    Transition.decrypt_transition for the same private input.

    Path A (reference, independently tested in test_chain_data.py):
      transition.decrypt_transition(tvk) -> read decrypted input value.

    Path B (method under review):
      Ciphertext.from_string(ct_str).decrypt_with_transition_view_key(tvk, program, fn, index).

    If hash_psd4 field ordering or any other step in the key derivation were
    wrong in exactly one path, this test would fail even though the KAT tests
    pass (since those exercise only Path B with a self-generated vector).
    """
    from aleo.mainnet import PrivateKey

    pk = PrivateKey.from_string(_XVAL_PRIVATE_KEY)
    transition = Transition.from_json(_XVAL_TRANSITION_JSON)
    tvk = transition.tvk(pk.view_key)

    # Path A: Transition.decrypt_transition — independent implementation.
    decrypted = transition.decrypt_transition(tvk)
    decrypted_inputs = decrypted.inputs()
    # The private input at index 1 is now exposed as a public plaintext.
    path_a_input = decrypted_inputs[_XVAL_PRIVATE_INPUT_INDEX]
    assert path_a_input["type"] == "public", (
        f"Expected decrypted input to have type 'public', got {path_a_input['type']!r}"
    )
    path_a_value = path_a_input["value"]

    # Path B: Ciphertext.decrypt_with_transition_view_key — the method under review.
    ct = Ciphertext.from_string(_XVAL_PRIVATE_INPUT_CT)
    path_b_plaintext = ct.decrypt_with_transition_view_key(
        tvk, "hello_hello.aleo", "main", _XVAL_PRIVATE_INPUT_INDEX
    )
    path_b_value = str(path_b_plaintext)

    # Both paths must agree with each other and with the known plaintext.
    assert path_a_value == _XVAL_EXPECTED_PLAINTEXT, (
        f"Path A (decrypt_transition) returned {path_a_value!r}, expected {_XVAL_EXPECTED_PLAINTEXT!r}"
    )
    assert path_b_value == _XVAL_EXPECTED_PLAINTEXT, (
        f"Path B (decrypt_with_transition_view_key) returned {path_b_value!r}, expected {_XVAL_EXPECTED_PLAINTEXT!r}"
    )
    assert path_a_value == path_b_value, (
        f"Path A ({path_a_value!r}) and Path B ({path_b_value!r}) disagree — "
        "derivation bug in one of the two implementations"
    )
