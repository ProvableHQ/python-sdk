"""
Tests for new account-type methods added for W4b parity.

These tests target methods that extend Address, PrivateKey, PrivateKeyCiphertext,
ViewKey, and Signature beyond the baseline snarkvm bindings.  They act as a
specification / acceptance suite: they are expected to fail until the
corresponding Rust pyo3 surface is wired up.
"""

import pytest
from aleo.mainnet import (
    Address,
    Field,
    Plaintext,
    PrivateKey,
    Signature,
    ViewKey,
)
from aleo.encryptor import Encryptor


# ---------------------------------------------------------------------------
# Known-answer constants
# ---------------------------------------------------------------------------

CREDITS_PROGRAM_ID = "credits.aleo"
CREDITS_ADDRESS = "aleo1lqmly7ez2k48ajf5hs92ulphaqr05qm4n8qwzj8v0yprmasgpqgsez59gg"

SEED_UNCHECKED_BYTES = bytes([
    94, 91, 52, 251, 240, 230, 226, 35, 117, 253, 224, 210,
    175, 13, 205, 120, 155, 214, 7, 169, 66, 62, 206, 50,
    188, 40, 29, 122, 40, 250, 54, 18,
])
SEED_UNCHECKED_KEY = "APrivateKey1zkp8CZNn3yeCseEtxuVPbDCwSyhGW6yZKUYKfgXmcpoGPWH"

CIPHERTEXT_PK = "APrivateKey1zkpAYS46Dq4rnt9wdohyWMwdmjmTeMJKPZdp5AhvjXZDsVG"
CIPHERTEXT_STR = (
    "ciphertext1qvqg7rgvam3xdcu55pwu6sl8rxwefxaj5gwthk0yzln6jv5fastzup0qn0qftqlqq"
    "7jcckyx03fzv9kke0z9puwd7cl7jzyhxfy2f2juplz39dkqs6p24urhxymhv364qm3z8mvyklv5"
    "gr52n4fxr2z59jgqytyddj8"
)
CIPHERTEXT_SECRET = "mypassword"

VALID_ADDRESS = "aleo1rhgdu77hgyqd3xjj8ucu3jj9r2krwz6mnzyd80gncr5fxcwlh5rsvzp9px"


# ---------------------------------------------------------------------------
# Address parity
# ---------------------------------------------------------------------------

class TestAddressParity:
    def test_from_program_id_kat(self):
        addr = Address.from_program_id(CREDITS_PROGRAM_ID)
        assert str(addr) == CREDITS_ADDRESS

    def test_from_program_id_invalid(self):
        with pytest.raises(Exception):
            Address.from_program_id("not_a_program_id")

    def test_is_valid_true(self):
        assert Address.is_valid(VALID_ADDRESS) is True

    def test_is_valid_uppercase(self):
        assert Address.is_valid(VALID_ADDRESS.upper()) is True

    def test_is_valid_false(self):
        assert Address.is_valid("not_an_address") is False

    def test_is_valid_too_short(self):
        assert Address.is_valid("aleo1xyz") is False

    def test_to_bits_le_roundtrip(self):
        addr = Address.from_string(VALID_ADDRESS)
        bits = addr.to_bits_le()
        assert isinstance(bits, list)
        assert all(isinstance(b, bool) for b in bits)
        addr2 = Address.from_bits_le(bits)
        assert addr == addr2

    def test_to_fields_roundtrip(self):
        addr = Address.from_string(VALID_ADDRESS)
        fields = addr.to_fields()
        assert isinstance(fields, list)
        assert len(fields) > 0
        addr2 = Address.from_fields(fields)
        assert addr == addr2

    def test_to_plaintext(self):
        addr = Address.from_string(VALID_ADDRESS)
        p = addr.to_plaintext()
        assert isinstance(p, Plaintext)
        assert p.is_literal()
        assert p.plaintext_type == "address"

    def test_lossy_casts(self):
        addr = Address.from_string(VALID_ADDRESS)
        # These just need to not raise.
        from aleo.mainnet import Scalar, Boolean
        s = addr.to_scalar_lossy()
        assert isinstance(s, Scalar)
        b = addr.to_boolean_lossy()
        assert isinstance(b, Boolean)
        assert addr.to_u8_lossy() is not None
        assert addr.to_u16_lossy() is not None
        assert addr.to_u32_lossy() is not None
        assert addr.to_u64_lossy() is not None
        assert addr.to_u128_lossy() is not None
        assert addr.to_i8_lossy() is not None
        assert addr.to_i16_lossy() is not None
        assert addr.to_i32_lossy() is not None
        assert addr.to_i64_lossy() is not None
        assert addr.to_i128_lossy() is not None


# ---------------------------------------------------------------------------
# PrivateKey parity
# ---------------------------------------------------------------------------

class TestPrivateKeyParity:
    def test_from_seed_unchecked_kat(self):
        pk = PrivateKey.from_seed_unchecked(list(SEED_UNCHECKED_BYTES))
        assert str(pk) == SEED_UNCHECKED_KEY

    def test_from_seed_unchecked_deterministic(self):
        seed = list(range(32))
        pk1 = PrivateKey.from_seed_unchecked(seed)
        pk2 = PrivateKey.from_seed_unchecked(seed)
        assert pk1 == pk2

    def test_from_seed_unchecked_bad_length(self):
        with pytest.raises(Exception):
            PrivateKey.from_seed_unchecked([1, 2, 3])

    def test_sign_value_and_verify(self):
        pk = PrivateKey.random()
        msg = "42u64"
        sig = pk.sign_value(msg)
        assert isinstance(sig, Signature)
        addr = pk.address
        assert sig.verify_value(addr, msg)
        assert not sig.verify_value(addr, "43u64")

    def test_bytes_roundtrip(self):
        pk = PrivateKey.random()
        b = pk.bytes()
        pk2 = PrivateKey.from_bytes(b)
        assert pk == pk2

    def test_new_encrypted(self):
        ct = PrivateKey.new_encrypted(CIPHERTEXT_SECRET)
        assert ct is not None

    def test_to_ciphertext_roundtrip(self):
        pk = PrivateKey.random()
        ct = pk.to_ciphertext(CIPHERTEXT_SECRET)
        pk2 = PrivateKey.from_private_key_ciphertext(ct, CIPHERTEXT_SECRET)
        assert pk == pk2

    def test_from_private_key_ciphertext_wrong_secret(self):
        pk = PrivateKey.random()
        ct = pk.to_ciphertext(CIPHERTEXT_SECRET)
        with pytest.raises(Exception):
            PrivateKey.from_private_key_ciphertext(ct, "wrongpassword")


# ---------------------------------------------------------------------------
# PrivateKeyCiphertext parity
#
# PrivateKeyCiphertext is a distinct type from the symmetric Ciphertext used by
# Encryptor.  It wraps the snarkvm PrivateKeyCiphertext and is exposed as a
# first-class Python type.
# ---------------------------------------------------------------------------

class TestPrivateKeyCiphertext:
    def test_from_string_kat(self):
        from aleo.mainnet import PrivateKeyCiphertext
        ct = PrivateKeyCiphertext.from_string(CIPHERTEXT_STR)
        assert str(ct) == CIPHERTEXT_STR

    def test_decrypt_to_private_key_kat(self):
        from aleo.mainnet import PrivateKeyCiphertext
        pk_expected = PrivateKey.from_string(CIPHERTEXT_PK)
        ct = PrivateKeyCiphertext.from_string(CIPHERTEXT_STR)
        pk = ct.decrypt_to_private_key(CIPHERTEXT_SECRET)
        assert pk == pk_expected

    def test_decrypt_wrong_secret_fails(self):
        from aleo.mainnet import PrivateKeyCiphertext
        ct = PrivateKeyCiphertext.from_string(CIPHERTEXT_STR)
        with pytest.raises(Exception):
            ct.decrypt_to_private_key("badpassword")

    def test_encrypt_decrypt_roundtrip(self):
        from aleo.mainnet import PrivateKeyCiphertext
        pk = PrivateKey.random()
        ct = PrivateKeyCiphertext.encrypt_private_key(pk, CIPHERTEXT_SECRET)
        pk2 = ct.decrypt_to_private_key(CIPHERTEXT_SECRET)
        assert pk == pk2

    def test_eq(self):
        from aleo.mainnet import PrivateKeyCiphertext
        ct1 = PrivateKeyCiphertext.from_string(CIPHERTEXT_STR)
        ct2 = PrivateKeyCiphertext.from_string(CIPHERTEXT_STR)
        assert ct1 == ct2

    def test_different_runs_different_ciphertext(self):
        """Nonce is random, so encrypting the same key twice yields different ciphertexts."""
        from aleo.mainnet import PrivateKeyCiphertext
        pk = PrivateKey.random()
        ct1 = PrivateKeyCiphertext.encrypt_private_key(pk, CIPHERTEXT_SECRET)
        ct2 = PrivateKeyCiphertext.encrypt_private_key(pk, CIPHERTEXT_SECRET)
        assert ct1 != ct2

    def test_interop_with_python_encryptor(self):
        """Python Encryptor and Rust PrivateKeyCiphertext must interoperate.

        The Python Encryptor uses symmetric field-based encryption (Poseidon2
        over the seed field element).  PrivateKeyCiphertext uses the same
        underlying scheme, so both sides should be able to round-trip through
        the other.
        """
        from aleo.mainnet import PrivateKeyCiphertext

        pk = PrivateKey.random()

        # Encrypt in Python, decrypt in Rust.
        ct_py = Encryptor.encrypt_private_key_with_secret(pk, CIPHERTEXT_SECRET)
        ct_rust = PrivateKeyCiphertext.from_string(str(ct_py))
        pk_rust = ct_rust.decrypt_to_private_key(CIPHERTEXT_SECRET)
        assert pk == pk_rust

        # Encrypt in Rust, decrypt in Python.
        # The Python Encryptor uses aleo.Ciphertext, so convert via string.
        from aleo.mainnet import Ciphertext
        ct_rust2 = PrivateKeyCiphertext.encrypt_private_key(pk, CIPHERTEXT_SECRET)
        ct_as_ciphertext = Ciphertext.from_string(str(ct_rust2))
        pk_py = Encryptor.decrypt_private_key_with_secret(ct_as_ciphertext, CIPHERTEXT_SECRET)
        assert pk == pk_py


# ---------------------------------------------------------------------------
# ViewKey parity
# ---------------------------------------------------------------------------

class TestViewKeyParity:
    def test_bytes_roundtrip(self):
        pk = PrivateKey.random()
        vk = pk.view_key
        b = vk.bytes()
        vk2 = ViewKey.from_bytes(b)
        assert vk == vk2

    def test_bytes_type(self):
        pk = PrivateKey.random()
        vk = pk.view_key
        b = vk.bytes()
        assert isinstance(b, list)
        assert all(isinstance(x, int) for x in b)


# ---------------------------------------------------------------------------
# Signature parity
# ---------------------------------------------------------------------------

class TestSignatureParity:
    def test_to_address(self):
        pk = PrivateKey.random()
        msg = b"hello world"
        sig = Signature.sign(pk, msg)
        addr = sig.to_address()
        assert isinstance(addr, Address)
        assert addr == pk.address

    def test_to_fields(self):
        pk = PrivateKey.random()
        sig = Signature.sign(pk, b"test")
        fields = sig.to_fields()
        assert isinstance(fields, list)
        assert len(fields) > 0
        assert all(isinstance(f, Field) for f in fields)

    def test_to_bits_le(self):
        pk = PrivateKey.random()
        sig = Signature.sign(pk, b"test")
        bits = sig.to_bits_le()
        assert isinstance(bits, list)
        assert all(isinstance(b, bool) for b in bits)

    def test_bytes_roundtrip(self):
        pk = PrivateKey.random()
        sig = Signature.sign(pk, b"test")
        b = sig.bytes()
        sig2 = Signature.from_bytes(b)
        assert sig == sig2

    def test_from_bits_le_roundtrip(self):
        pk = PrivateKey.random()
        sig = Signature.sign(pk, b"test")
        bits = sig.to_bits_le()
        sig2 = Signature.from_bits_le(bits)
        assert sig == sig2

    def test_to_plaintext(self):
        pk = PrivateKey.random()
        sig = Signature.sign(pk, b"test")
        p = sig.to_plaintext()
        assert isinstance(p, Plaintext)
        assert p.is_literal()
        assert p.plaintext_type == "signature"

    def test_sign_value_verify_value(self):
        pk = PrivateKey.random()
        msg = "100u64"
        sig = Signature.sign_value(pk, msg)
        assert sig.verify_value(pk.address, msg)
        assert not sig.verify_value(pk.address, "99u64")
