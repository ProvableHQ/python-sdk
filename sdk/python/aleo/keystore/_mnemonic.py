"""Recovery-phrase checking and Aleo key derivation.

Network-neutral: the result is the 32-byte seed of a private key, which either
network module turns into a key with ``PrivateKey.from_seed_unchecked``.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import cast

from ._errors import KeystoreError

_WORD_COUNTS = (12, 15, 18, 21, 24)
_HARDENED = 0x80000000
_MASTER_TAG = b"bls12_377 seed"
# Coin type of the account path m/44'/<coin>'/<index>'/0'.
_COIN_TYPES = {"standard": 683, "legacy": 0}


@lru_cache(maxsize=1)
def _word_list() -> tuple[str, ...]:
    """The BIP-39 English word list, in order: position is the 11-bit value."""
    text = Path(__file__).with_name("_bip39_english.txt").read_text(encoding="utf-8")
    words = tuple(text.split())
    if len(words) != 2048:
        raise RuntimeError("The bundled BIP-39 word list is damaged.")
    return words


@lru_cache(maxsize=1)
def _word_indexes() -> dict[str, int]:
    """The BIP-39 English word list, mapping each word to its 11-bit value."""
    return {word: index for index, word in enumerate(_word_list())}


def entropy_to_phrase(entropy: bytes) -> str:
    """Encode random bytes as a BIP-39 English recovery phrase.

    Appends the checksum (the leading bits of the SHA-256 of *entropy*, one
    bit per 32 bits of entropy) and writes the result as 11-bit words, so 16
    bytes give 12 words and 32 bytes give 24.

    Args:
        entropy: 16, 20, 24, 28 or 32 random bytes.

    Returns:
        The phrase, as lower-case words separated by single spaces.

    Raises:
        ValueError: If *entropy* has another length.
    """
    if len(entropy) not in (16, 20, 24, 28, 32):
        raise ValueError(f"entropy must be 16, 20, 24, 28 or 32 bytes, not {len(entropy)}")
    checksum_length = len(entropy) // 4
    checksum = hashlib.sha256(entropy).digest()[0] >> (8 - checksum_length)
    bits = (int.from_bytes(entropy, "big") << checksum_length) | checksum
    count = (len(entropy) * 8 + checksum_length) // 11
    words = _word_list()
    return " ".join(words[(bits >> (11 * (count - 1 - position))) & 0x7FF] for position in range(count))


def generate_phrase(strength: int = 128) -> str:
    """Create a new BIP-39 English recovery phrase from the system's random source.

    Args:
        strength: Bits of randomness: 128 gives 12 words, 256 gives 24.

    Returns:
        The phrase. It is the root secret of every account derived from it.

    Raises:
        ValueError: If *strength* is neither 128 nor 256.
    """
    # A float such as 128.0 compares equal to 128, so the type is checked too.
    if isinstance(strength, bool) or not isinstance(cast(object, strength), int) or strength not in (128, 256):
        raise ValueError(f"strength must be 128 (12 words) or 256 (24 words), not {strength!r}")
    return entropy_to_phrase(secrets.token_bytes(strength // 8))


def clean_phrase(text: str) -> str:
    """Tidy a typed or pasted recovery phrase before it is checked.

    Returns:
        The phrase with surrounding white space removed, runs of white space
        collapsed to single spaces, and letters lower-cased.
    """
    return " ".join(text.split()).lower()


def _invalid(reason: str) -> KeystoreError:
    return KeystoreError("INVALID_MNEMONIC", f"Not a valid recovery phrase: {reason}.")


def check_phrase(phrase: str) -> None:
    """Raise ``INVALID_MNEMONIC`` unless *phrase* is a valid BIP-39 phrase.

    The phrase is taken as written: lower-case English words separated by
    single spaces. The message never repeats a word of the phrase.
    """
    words = phrase.split(" ")
    if len(words) not in _WORD_COUNTS:
        raise _invalid(
            "expected 12, 15, 18, 21 or 24 words separated by single spaces, "
            f"found {len(words)}"
        )
    indexes = _word_indexes()
    bits = 0
    for position, word in enumerate(words, start=1):
        value = indexes.get(word)
        if value is None:
            raise _invalid(f"word {position} is not in the BIP-39 English word list")
        bits = (bits << 11) | value
    checksum_length = len(words) // 3
    entropy = (bits >> checksum_length).to_bytes(len(words) * 4 // 3, "big")
    expected = hashlib.sha256(entropy).digest()[0] >> (8 - checksum_length)
    if bits & ((1 << checksum_length) - 1) != expected:
        raise _invalid("the checksum does not match, so a word is wrong or out of order")


def derive_seed(mnemonic: str, *, index: int = 0, derivation: str = "standard") -> bytes:
    """Return the private-key seed for one account of a recovery phrase.

    Follows the derivation wallets use for Aleo: the BIP-39 seed of the phrase
    (no passphrase), a master node tagged for the BLS12-377 curve, then the
    hardened path ``m/44'/683'/<index>'/0'``, or ``m/44'/0'/<index>'/0'`` for
    ``"legacy"``.

    Args:
        mnemonic: BIP-39 English phrase: lower-case words separated by single
            spaces.
        index: Account index, from 0 through 2**31 - 1.
        derivation: ``"standard"`` or ``"legacy"``.

    Returns:
        The 32 bytes that ``PrivateKey.from_seed_unchecked`` turns into the
        account's private key.

    Raises:
        KeystoreError: ``INVALID_MNEMONIC`` when the phrase fails the word
            list, the word count or the checksum.
        ValueError: *index* or *derivation* is out of range.
    """
    coin_type = _COIN_TYPES.get(derivation)
    if coin_type is None:
        raise ValueError(
            f"derivation must be 'standard' or 'legacy', not {derivation!r}"
        )
    # Checked at run time as well: an untyped caller can pass anything.
    position = cast(object, index)
    if not isinstance(position, int) or isinstance(position, bool) or not 0 <= position < _HARDENED:
        raise ValueError(
            f"index must be a whole number from 0 to 2**31 - 1, not {index!r}"
        )

    words = cast(object, mnemonic)
    if not isinstance(words, str):
        raise _invalid("it is not text")
    phrase = unicodedata.normalize("NFKD", words)
    check_phrase(phrase)
    seed = hashlib.pbkdf2_hmac("sha512", phrase.encode("utf-8"), b"mnemonic", 2048, 64)

    node = hmac.new(_MASTER_TAG, seed, hashlib.sha512).digest()
    for segment in (44, coin_type, position, 0):
        data = b"\x00" + node[:32] + (segment + _HARDENED).to_bytes(4, "big")
        node = hmac.new(node[32:], data, hashlib.sha512).digest()
    return node[:32]
