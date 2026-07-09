"""Tests for W4d2 wasm-parity additions."""
import pytest
from aleo.mainnet import Proof, DynamicRecord, RecordPlaintext


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
