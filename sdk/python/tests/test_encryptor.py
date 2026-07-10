from __future__ import annotations

import pytest
from aleo.mainnet import PrivateKey
from aleo.encryptor import Encryptor


def test_encrypt_decrypt_private_key_roundtrip():
    pk = PrivateKey.random()
    ct = Encryptor.encrypt_private_key_with_secret(pk, "s3cret")
    recovered = Encryptor.decrypt_private_key_with_secret(ct, "s3cret")
    assert recovered == pk


def test_wrong_secret_fails():
    pk = PrivateKey.random()
    ct = Encryptor.encrypt_private_key_with_secret(pk, "s3cret")
    # A wrong secret produces a bad blinding factor; the recovered seed will be
    # incorrect, yielding either a ValueError/RuntimeError from the crypto layer
    # (an outright rejection) or a different private key. AssertionError and
    # unexpected exception types (e.g. TypeError from an API break) must FAIL.
    try:
        recovered = Encryptor.decrypt_private_key_with_secret(ct, "wrong")
    except (ValueError, RuntimeError):
        return  # crypto layer rejected the wrong secret outright — acceptable
    assert recovered != pk, "wrong secret must not decrypt to the original key"
