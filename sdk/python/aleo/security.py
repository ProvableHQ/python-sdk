# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""PyNaCl sealed-box helpers for DPS encryption."""
from __future__ import annotations

import base64
import struct
from typing import Any


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


def encrypt_registration_request(public_key_b64: str, view_key: Any, start_block: int) -> str:
    """Encrypt view_key LE bytes (32) + start_block u32 LE (4) with NaCl sealed box.

    Returns standard (non-URL-safe) base64.
    Wire shape: 32 (ephemeral pk) + 16 (MAC) + 36 (plaintext) = 84 bytes total ciphertext.
    """
    try:
        from nacl.public import PublicKey, SealedBox  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "PyNaCl is required for record scanner registration. "
            "Install with: pip install aleo[dps]"
        ) from None

    vk_bytes = bytes(view_key.bytes())      # 32 bytes, LE
    start_bytes = struct.pack("<I", start_block)  # 4 bytes, LE u32
    plaintext = vk_bytes + start_bytes           # 36 bytes total

    recipient_key = PublicKey(base64.b64decode(public_key_b64))
    box = SealedBox(recipient_key)
    ciphertext = box.encrypt(plaintext)           # 32 + 16 + 36 = 84 bytes
    return base64.b64encode(ciphertext).decode("ascii")
