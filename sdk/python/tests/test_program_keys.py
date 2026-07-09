"""Tests for W4d2 wasm-parity additions."""
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
