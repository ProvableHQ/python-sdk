"""Bitcoin/Solana base58 (no checksum). Vendored so Solana recipients encode without the solana extra."""
from __future__ import annotations

ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_INDEX = {c: i for i, c in enumerate(ALPHABET)}


def b58decode(s: str) -> bytes:
    """Decode *s*; leading ``1`` characters become leading zero bytes. ``ValueError`` on a bad character."""
    if not isinstance(s, str):
        raise ValueError("base58 input must be a str")
    n = 0
    for ch in s:
        try:
            n = n * 58 + _INDEX[ch]
        except KeyError:
            raise ValueError(f"Invalid base58 character {ch!r}") from None
    body = n.to_bytes((n.bit_length() + 7) // 8, "big") if n else b""
    pad = len(s) - len(s.lstrip("1"))
    return bytes(pad) + body


def b58encode(data: bytes) -> str:
    """Encode *data*; leading zero bytes become leading ``1`` characters."""
    data = bytes(data)
    pad = len(data) - len(data.lstrip(b"\x00"))
    n = int.from_bytes(data, "big")
    out = []
    while n:
        n, rem = divmod(n, 58)
        out.append(ALPHABET[rem])
    return "1" * pad + "".join(reversed(out))


__all__ = ["ALPHABET", "b58decode", "b58encode"]
