"""Tests for security.py sealed-box helpers."""
from __future__ import annotations

import base64
import pytest


def test_encrypt_proving_request_returns_base64() -> None:
    from nacl.public import PrivateKey
    from aleo.security import encrypt_proving_request

    sk = PrivateKey.generate()
    pk_b64 = base64.b64encode(bytes(sk.public_key)).decode()

    message = b"hello world"
    result = encrypt_proving_request(pk_b64, message)

    # Should be valid standard base64
    decoded = base64.b64decode(result)
    assert len(decoded) > 0


def test_sealed_box_wire_shape() -> None:
    """Ciphertext is 32 (epk) + 16 (mac) + len(msg) bytes."""
    from nacl.public import PrivateKey
    from aleo.security import encrypt_proving_request

    sk = PrivateKey.generate()
    pk_b64 = base64.b64encode(bytes(sk.public_key)).decode()

    msg = b"test message bytes"
    result = encrypt_proving_request(pk_b64, msg)
    decoded = base64.b64decode(result)

    assert len(decoded) == 32 + 16 + len(msg)


def test_distinct_ciphertexts_per_call() -> None:
    """Each call produces a distinct ciphertext (ephemeral key)."""
    from nacl.public import PrivateKey
    from aleo.security import encrypt_proving_request

    sk = PrivateKey.generate()
    pk_b64 = base64.b64encode(bytes(sk.public_key)).decode()
    msg = b"same message"

    ct1 = encrypt_proving_request(pk_b64, msg)
    ct2 = encrypt_proving_request(pk_b64, msg)

    assert ct1 != ct2


def test_decrypt_roundtrip() -> None:
    """Ciphertext decrypts back to original message."""
    from nacl.public import PrivateKey, SealedBox
    from aleo.security import encrypt_proving_request

    sk = PrivateKey.generate()
    pk_b64 = base64.b64encode(bytes(sk.public_key)).decode()
    msg = b"roundtrip test data"

    ct = encrypt_proving_request(pk_b64, msg)
    decrypted = SealedBox(sk).decrypt(base64.b64decode(ct))
    assert decrypted == msg


@pytest.mark.parametrize("function_name,args,purpose", [
    ("encrypt_proving_request", ("dGVzdA==", b"msg"), "DPS encryption"),
    ("encrypt_registration_request", ("dGVzdA==", None, 0), "record scanner registration"),
])
def test_encrypt_requires_pynacl(monkeypatch, function_name, args, purpose) -> None:
    """An incomplete installation explains how to restore the required dependency."""
    import sys
    import aleo.security as security

    monkeypatch.setitem(sys.modules, "nacl", None)
    monkeypatch.setitem(sys.modules, "nacl.public", None)
    with pytest.raises(ImportError) as exc:
        getattr(security, function_name)(*args)
    assert str(exc.value) == (
        f"PyNaCl is required for {purpose}. "
        "Restore the required dependency with: python -m pip install 'pynacl>=1.5'"
    )
