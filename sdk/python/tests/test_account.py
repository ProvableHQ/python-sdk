from aleo.mainnet import PrivateKey, Signature
from conftest import load_vectors


def test_key_derivation_triples():
    for t in load_vectors("accounts.json")["triples"]:
        pk = PrivateKey.from_string(t["private_key"])
        assert str(pk.view_key) == t["view_key"]
        assert str(pk.address) == t["address"]


def test_sign_verify_roundtrip():
    msg = bytes(load_vectors("accounts.json")["message_bytes"])
    pk = PrivateKey.random()
    sig = pk.sign(msg)
    # verify lives on Signature: sig.verify(address, message) -> bool
    assert sig.verify(pk.address, msg) is True
    assert sig.verify(pk.address, b"tampered") is False
