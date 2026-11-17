"""Key parsing and the password-encrypted form of a private key.

A password is stretched with scrypt, and the result is the secret for the
SDK's own private-key ciphertext. The stretching is what slows down password
guessing; the ciphertext scheme alone does not.
"""
from __future__ import annotations

import base64
import hashlib
import os
import re
import unicodedata
from typing import Optional, Tuple, TypedDict

# Private keys and their ciphertexts do not depend on the network, so the
# mainnet types serve accounts of either network.
from ..mainnet import PrivateKey, PrivateKeyCiphertext
from ._errors import KeystoreError

# scrypt cost used for every new encrypted account file.
SCRYPT_N = 131072
# scrypt block size. The only value a reader accepts.
SCRYPT_R = 8
# scrypt parallelisation. The only value a reader accepts.
SCRYPT_P = 1
# Smallest scrypt cost a reader accepts (2^14).
SCRYPT_MIN_N = 2**14
# Largest scrypt cost a reader accepts (2^20), which bounds memory at 1 GiB.
SCRYPT_MAX_N = 2**20
# Salt length for new files, and the shortest salt a reader accepts, in bytes.
SALT_BYTES = 16

# Length of the scrypt output, in bytes.
_SECRET_BYTES = 32
# Memory allowed to scrypt at the default cost, in bytes (256 MiB).
_BASE_MAXMEM = 256 * 1024 * 1024

_LONE_SURROGATE = re.compile("[\ud800-\udfff]")


class KdfParams(TypedDict):
    """scrypt parameters of an encrypted account file."""

    N: int
    r: int
    p: int
    salt: str


class EncryptedKey(TypedDict):
    """The ``key`` object of an encrypted account file."""

    type: str
    kdf: str
    kdfParams: KdfParams
    ciphertext: str


def parse_key(private_key: object, origin: str) -> Tuple[str, str]:
    """Parse an Aleo private key and derive its address.

    Args:
        private_key: Candidate private key string.
        origin: Where the string came from, named in the error message.

    Returns:
        The key in its canonical text form and the address it derives.

    Raises:
        KeystoreError: ``INVALID_KEY`` when the value is not a private key.
            The message names the origin and never the value.
    """
    parsed: Optional[Tuple[str, str]] = None
    if isinstance(private_key, str):
        try:
            key = PrivateKey.from_string(private_key)
            parsed = (str(key), str(key.address))
        except Exception:
            parsed = None
    if parsed is None:
        # Raised outside the handler so nothing derived from the input can
        # reach a log through the exception chain.
        raise KeystoreError("INVALID_KEY", f"{origin} is not a valid Aleo private key.")
    return parsed


def encode_password(password: str) -> bytes:
    """Return a password's Unicode NFKC form encoded as UTF-8.

    A lone surrogate has no UTF-8 form; it becomes U+FFFD, as in other UTF-8
    encoders, instead of failing.
    """
    normalized = unicodedata.normalize("NFKC", password)
    return _LONE_SURROGATE.sub("\ufffd", normalized).encode("utf-8")


def normalize_password(password: object) -> bytes:
    """Convert a password to the bytes that are stretched and stored.

    Returns:
        The password normalised to Unicode NFKC and encoded as UTF-8.

    Raises:
        KeystoreError: ``PASSWORD_REQUIRED`` when the password is empty.
    """
    encoded = encode_password(password) if isinstance(password, str) else b""
    if not encoded:
        raise KeystoreError("PASSWORD_REQUIRED", "The password must not be empty.")
    return encoded


def derive_secret(password: bytes, salt: bytes, n: int) -> str:
    """Stretch a password into the secret that encrypts the private key.

    Runs scrypt with ``r = 8``, ``p = 1`` and a 32-byte output. Takes about
    200 ms and 128 MiB at the default cost.

    Args:
        password: Normalised password bytes from :func:`normalize_password`.
        salt: Salt bytes.
        n: scrypt cost, a power of two.

    Returns:
        The 32-byte output as 64 lower-case hex characters.
    """
    # scrypt needs 128 * N * r bytes plus a few kilobytes. 256 MiB covers the
    # default cost; a file with a larger accepted cost gets what it needs, and
    # the reader's upper bound on N keeps that at about 1 GiB.
    maxmem = max(_BASE_MAXMEM, 128 * n * SCRYPT_R + 1024 * 1024)
    derived = hashlib.scrypt(
        password, salt=salt, n=n, r=SCRYPT_R, p=SCRYPT_P, maxmem=maxmem, dklen=_SECRET_BYTES
    )
    return derived.hex()


def encrypt_key(private_key: str, password: bytes) -> EncryptedKey:
    """Encrypt a private key under a password with a fresh random salt.

    Args:
        private_key: Aleo private key in text form.
        password: Normalised password bytes from :func:`normalize_password`.

    Returns:
        The ``key`` object for an encrypted account file.
    """
    salt = os.urandom(SALT_BYTES)
    secret = derive_secret(password, salt, SCRYPT_N)
    ciphertext = PrivateKeyCiphertext.encrypt_private_key(PrivateKey.from_string(private_key), secret)
    return {
        "type": "encrypted",
        "kdf": "scrypt",
        "kdfParams": {
            "N": SCRYPT_N,
            "r": SCRYPT_R,
            "p": SCRYPT_P,
            "salt": base64.b64encode(salt).decode("ascii"),
        },
        "ciphertext": str(ciphertext),
    }


def decrypt_key(key: EncryptedKey, address: str, password: bytes) -> str:
    """Decrypt the private key of an encrypted account file.

    Args:
        key: The file's ``key`` object, already checked against the format.
        address: Address recorded in the file.
        password: Normalised password bytes from :func:`normalize_password`.

    Returns:
        The private key in text form.

    Raises:
        KeystoreError: ``WRONG_PASSWORD`` when decryption fails or the
            decrypted key derives an address other than the file's.
    """
    params = key["kdfParams"]
    secret = derive_secret(password, base64.b64decode(params["salt"]), params["N"])
    decrypted: Optional[Tuple[str, str]] = None
    try:
        opened = PrivateKeyCiphertext.from_string(key["ciphertext"]).decrypt_to_private_key(secret)
        decrypted = (str(opened), str(opened.address))
    except Exception:
        decrypted = None
    # A decryption that succeeds still has to produce the recorded account.
    if decrypted is None or decrypted[1] != address:
        raise KeystoreError("WRONG_PASSWORD", f"The password does not unlock the account {address}.")
    return decrypted[0]
