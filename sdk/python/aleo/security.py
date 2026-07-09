"""PyNaCl sealed-box helpers for DPS encryption."""
from __future__ import annotations

import base64


def encrypt_proving_request(public_key_b64: str, pr_bytes: bytes) -> str:
    """Encrypt bytes with a Curve25519 public key using a NaCl sealed box.

    Returns standard (non-URL-safe) base64.
    Raises ImportError with helpful message if pynacl is not installed.
    """
    try:
        from nacl.public import PublicKey, SealedBox  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "PyNaCl is required for DPS encryption. "
            "Install with: pip install aleo[dps]"
        ) from None

    recipient_key = PublicKey(base64.b64decode(public_key_b64))
    box = SealedBox(recipient_key)
    ciphertext = box.encrypt(pr_bytes)
    return base64.b64encode(ciphertext).decode("ascii")
