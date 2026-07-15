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
        # Concrete expected values pinned as regression anchors.
        # These are deterministic derivations from VALID_ADDRESS — computed once
        # and hardcoded so future refactors cannot silently change the output.
        # Each lossy cast truncates the 256-bit address encoding to the target
        # type's bit-width; the exact values were observed on 2026-07-09.
        addr = Address.from_string(VALID_ADDRESS)
        from aleo.mainnet import Scalar, Boolean

        # to_field: address x-coordinate (full precision)
        assert str(addr.to_field()) == "3501665755452795161867664882580888971213780722176652848275908626939553697821field"

        # to_scalar_lossy: 252-bit truncation into the scalar field
        s = addr.to_scalar_lossy()
        assert isinstance(s, Scalar)
        assert str(s) == "1692414361119729608374368241820140411006437211776019035159383876815911047197scalar"

        # to_boolean_lossy: least significant bit of the encoding
        b = addr.to_boolean_lossy()
        assert isinstance(b, Boolean)
        assert str(b) == "true"

        # integer lossy casts: truncation to N-bit two's-complement / unsigned
        assert str(addr.to_u8_lossy()) == "29u8"
        assert str(addr.to_u16_lossy()) == "53277u16"
        assert str(addr.to_u32_lossy()) == "2078199837u32"
        assert str(addr.to_u64_lossy()) == "15564512705944408093u64"
        assert str(addr.to_u128_lossy()) == "34922309281260474190457069241198628893u128"
        assert str(addr.to_i8_lossy()) == "29i8"
        assert str(addr.to_i16_lossy()) == "-12259i16"
        assert str(addr.to_i32_lossy()) == "2078199837i32"
        assert str(addr.to_i64_lossy()) == "-2882231367765143523i64"
        assert str(addr.to_i128_lossy()) == "34922309281260474190457069241198628893i128"


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
        """Cross-scheme decryption must work in BOTH directions.

        The Python Encryptor and Rust PrivateKeyCiphertext implement the same
        underlying scheme (Poseidon2 hash-based symmetric encryption over the
        seed field element).  True scheme identity means each side can decrypt
        what the other encrypted — string-format compatibility alone is not
        sufficient.  Equality of the recovered PrivateKey is the only proof.
        """
        from aleo.mainnet import Ciphertext, PrivateKeyCiphertext

        SECRET = CIPHERTEXT_SECRET
        pk = PrivateKey.random()

        # --- Direction A: encrypt with pure-Python Encryptor, decrypt with Rust ---
        # The Python Encryptor derives its symmetric key independently from the
        # seed field via Network.hash_psd2.  Parsing the ciphertext string into
        # PrivateKeyCiphertext and calling decrypt_to_private_key with the same
        # secret must recover the identical original key.
        ct_py = Encryptor.encrypt_private_key_with_secret(pk, SECRET)
        ct_rust_parsed = PrivateKeyCiphertext.from_string(str(ct_py))
        pk_from_rust_decrypt = ct_rust_parsed.decrypt_to_private_key(SECRET)
        assert pk_from_rust_decrypt == pk, (
            "Direction A failed: Rust could not decrypt a Python-encrypted ciphertext. "
            "This means the two implementations use different encryption schemes."
        )

        # --- Direction B: encrypt with Rust PrivateKeyCiphertext, decrypt with Python ---
        # The Rust PrivateKeyCiphertext.encrypt_private_key derives its symmetric
        # key independently.  Converting to a Ciphertext via string round-trip and
        # decrypting with Encryptor.decrypt_private_key_with_secret must recover
        # the identical original key.
        ct_rust = PrivateKeyCiphertext.encrypt_private_key(pk, SECRET)
        ct_py_parsed = Ciphertext.from_string(str(ct_rust))
        pk_from_py_decrypt = Encryptor.decrypt_private_key_with_secret(ct_py_parsed, SECRET)
        assert pk_from_py_decrypt == pk, (
            "Direction B failed: Python could not decrypt a Rust-encrypted ciphertext. "
            "This means the two implementations use different encryption schemes."
        )


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
        assert isinstance(b, bytes)   # pyo3 0.23+: Vec<u8> maps to bytes
        assert len(b) == 32


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
