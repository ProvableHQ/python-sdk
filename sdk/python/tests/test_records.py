import pytest
from aleo.mainnet import ViewKey, RecordCiphertext
from conftest import load_vectors


def test_record_decrypt_kat():
    v = load_vectors("records.json")["decrypt_kat"]
    ct = RecordCiphertext.from_string(v["ciphertext"])
    vk = ViewKey.from_string(v["view_key"])

    # Ownership checks
    assert ct.is_owner(vk) is True
    assert ct.is_owner(ViewKey.from_string(v["non_owner_view_key"])) is False

    # Decrypt and verify plaintext fields
    pt = ct.decrypt(vk)

    # str(pt.owner) returns "address.private" — strip the visibility qualifier
    assert str(pt.owner).split(".")[0] == v["owner"]
    assert str(pt.nonce) == v["nonce"]

    # Version: this is a v0-era record; snarkvm 4.7.3 reports _version=0u8
    assert isinstance(pt.version, int)
    assert pt.version == 0

    # Microcredits: assert the decrypted plaintext contains the expected value,
    # not just that the vector has the right constant (str(pt) contains "Xu64")
    assert f"{v['microcredits']}u64" in str(pt)
