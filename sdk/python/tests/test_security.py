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


def test_encrypt_requires_pynacl() -> None:
    """Helpful ImportError when pynacl is not installed."""
    import sys
    import importlib

    # Temporarily hide nacl
    original = sys.modules.get("nacl")
    original_public = sys.modules.get("nacl.public")

    sys.modules["nacl"] = None  # type: ignore[assignment]
    sys.modules["nacl.public"] = None  # type: ignore[assignment]

    try:
        # Re-import security with nacl hidden
        import aleo.security as sec_mod
        # Force reimport
        importlib.reload(sec_mod)
        with pytest.raises(ImportError, match="aleo\\[dps\\]"):
            sec_mod.encrypt_proving_request("dGVzdA==", b"msg")
    finally:
        if original is None:
            sys.modules.pop("nacl", None)
        else:
            sys.modules["nacl"] = original
        if original_public is None:
            sys.modules.pop("nacl.public", None)
        else:
            sys.modules["nacl.public"] = original_public
        importlib.reload(sec_mod)
