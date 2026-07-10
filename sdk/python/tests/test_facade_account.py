"""Tests for F2 facade: aleo.account module.

All tests are purely local (no network I/O).  They exercise:
- create() / from_private_key() / from_seed()
- export_encrypted() / import_encrypted()
- sign() / verify() — raw bytes
- sign_value() / verify_value() — structured Aleo Value signing
- default_account wiring on the Aleo client
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aleo import Aleo, HTTPProvider
from aleo.facade.account import AccountModule

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

BASE = "https://api.provable.com/v2"

VECTORS_DIR = Path(__file__).parent / "vectors"


def load_accounts() -> dict:  # type: ignore[type-arg]
    return json.loads((VECTORS_DIR / "accounts.json").read_text())


def make_client(network: str = "mainnet") -> Aleo:
    return Aleo(HTTPProvider(BASE, network=network))


# Known-Answer Test triple (first entry from accounts.json)
KAT = {
    "private_key": "APrivateKey1zkp8CZNn3yeCseEtxuVPbDCwSyhGW6yZKUYKfgXmcpoGPWH",
    "view_key": "AViewKey1mSnpFFC8Mj4fXbK5YiWgZ3mjiV8CxA79bYNa8ymUpTrw",
    "address": "aleo1rhgdu77hgyqd3xjj8ucu3jj9r2krwz6mnzyd80gncr5fxcwlh5rsvzp9px",
}

# ---------------------------------------------------------------------------
# AccountModule attachment
# ---------------------------------------------------------------------------


def test_account_module_attached() -> None:
    """aleo.account is an AccountModule instance."""
    a = make_client()
    assert isinstance(a.account, AccountModule)


def test_account_module_same_instance() -> None:
    """aleo.account returns the same object on every access."""
    a = make_client()
    assert a.account is a.account


# ---------------------------------------------------------------------------
# create()
# ---------------------------------------------------------------------------


def test_create_returns_account_with_aleo_address() -> None:
    """create() returns an Account whose address starts with 'aleo1'."""
    a = make_client()
    acct = a.account.create()
    assert str(acct.address).startswith("aleo1")


def test_create_unique() -> None:
    """Two consecutive create() calls produce different accounts."""
    a = make_client()
    a1 = a.account.create()
    a2 = a.account.create()
    assert str(a1.private_key) != str(a2.private_key)


def test_create_has_private_key_and_view_key() -> None:
    """Created account exposes private_key, view_key, and address."""
    a = make_client()
    acct = a.account.create()
    assert str(acct.private_key).startswith("APrivateKey1")
    assert str(acct.view_key).startswith("AViewKey1")


# ---------------------------------------------------------------------------
# from_private_key() — Known-Answer Tests
# ---------------------------------------------------------------------------


def test_from_private_key_string_reproduces_kat() -> None:
    """from_private_key(str) derives the correct view_key and address."""
    a = make_client()
    acct = a.account.from_private_key(KAT["private_key"])
    assert str(acct.view_key) == KAT["view_key"]
    assert str(acct.address) == KAT["address"]


def test_from_private_key_all_triples() -> None:
    """from_private_key matches all triples in accounts.json."""
    a = make_client()
    triples = load_accounts()["triples"]
    for triple in triples:
        acct = a.account.from_private_key(triple["private_key"])
        assert str(acct.view_key) == triple["view_key"]
        assert str(acct.address) == triple["address"]


def test_from_private_key_accepts_private_key_object() -> None:
    """from_private_key() accepts a PrivateKey object as well as a string."""
    import aleo.mainnet as m
    a = make_client()
    pk = m.PrivateKey.from_string(KAT["private_key"])
    acct = a.account.from_private_key(pk)
    assert str(acct.address) == KAT["address"]


def test_from_private_key_invalid_string_raises() -> None:
    """from_private_key() raises when given an invalid key string."""
    a = make_client()
    with pytest.raises(Exception):
        a.account.from_private_key("not_a_private_key")


# ---------------------------------------------------------------------------
# from_seed()
# ---------------------------------------------------------------------------


def test_from_seed_with_field_object() -> None:
    """from_seed(Field) returns an account with a valid aleo1 address."""
    import aleo.mainnet as m
    a = make_client()
    seed = m.Field.random()
    acct = a.account.from_seed(seed)
    assert str(acct.address).startswith("aleo1")


def test_from_seed_deterministic() -> None:
    """from_seed is deterministic: same seed → same account."""
    import aleo.mainnet as m
    a = make_client()
    seed = m.Field.random()
    acct1 = a.account.from_seed(seed)
    acct2 = a.account.from_seed(seed)
    assert str(acct1.address) == str(acct2.address)


def test_from_seed_different_seeds_differ() -> None:
    """Different seeds produce different accounts."""
    import aleo.mainnet as m
    a = make_client()
    s1 = m.Field.random()
    s2 = m.Field.random()
    acct1 = a.account.from_seed(s1)
    acct2 = a.account.from_seed(s2)
    assert str(acct1.address) != str(acct2.address)


# ---------------------------------------------------------------------------
# export_encrypted() / import_encrypted()
# ---------------------------------------------------------------------------


def test_export_import_encrypted_round_trip() -> None:
    """Encrypted export round-trips back to the original private key."""
    a = make_client()
    acct = a.account.create()
    ct = a.account.export_encrypted(acct, "supersecret")
    recovered = a.account.import_encrypted(ct, "supersecret")
    assert str(recovered.private_key) == str(acct.private_key)
    assert str(recovered.address) == str(acct.address)


def test_export_encrypted_produces_ciphertext() -> None:
    """export_encrypted returns a PrivateKeyCiphertext (serialises to a string)."""
    a = make_client()
    acct = a.account.create()
    ct = a.account.export_encrypted(acct, "pw")
    # Should serialise without error; ciphertext strings start with 'ciphertext1'
    ct_str = str(ct)
    assert ct_str.startswith("ciphertext1")


def test_import_encrypted_accepts_string() -> None:
    """import_encrypted accepts the serialised ciphertext string."""
    a = make_client()
    acct = a.account.create()
    ct = a.account.export_encrypted(acct, "pw")
    ct_str = str(ct)
    recovered = a.account.import_encrypted(ct_str, "pw")
    assert str(recovered.private_key) == str(acct.private_key)


def test_import_encrypted_wrong_secret_fails() -> None:
    """import_encrypted with a wrong secret does not yield the original key."""
    a = make_client()
    acct = a.account.create()
    ct = a.account.export_encrypted(acct, "correct")
    try:
        recovered = a.account.import_encrypted(ct, "wrong")
    except (ValueError, RuntimeError):
        return  # crypto layer rejected the wrong secret — expected
    # If it did not raise, the recovered key must differ from the original.
    assert str(recovered.private_key) != str(acct.private_key)


# ---------------------------------------------------------------------------
# sign() / verify()
# ---------------------------------------------------------------------------


def test_sign_verify_round_trip() -> None:
    """sign() + verify() round-trip on raw bytes returns True."""
    a = make_client()
    acct = a.account.create()
    msg = b"hello aleo"
    sig = a.account.sign(msg, acct)
    assert a.account.verify(acct.address, msg, sig) is True


def test_verify_tampered_message_returns_false() -> None:
    """verify() returns False for a tampered message."""
    a = make_client()
    acct = a.account.create()
    sig = a.account.sign(b"original", acct)
    assert a.account.verify(acct.address, b"tampered", sig) is False


def test_verify_wrong_address_returns_false() -> None:
    """verify() returns False when the address does not match the signer."""
    a = make_client()
    acct1 = a.account.create()
    acct2 = a.account.create()
    sig = a.account.sign(b"msg", acct1)
    assert a.account.verify(acct2.address, b"msg", sig) is False


def test_verify_accepts_address_string() -> None:
    """verify() accepts an address as a plain string."""
    a = make_client()
    acct = a.account.create()
    sig = a.account.sign(b"data", acct)
    assert a.account.verify(str(acct.address), b"data", sig) is True


def test_verify_accepts_signature_string() -> None:
    """verify() accepts a serialised signature string."""
    a = make_client()
    acct = a.account.create()
    sig = a.account.sign(b"data", acct)
    assert a.account.verify(acct.address, b"data", str(sig)) is True


# ---------------------------------------------------------------------------
# sign_value() / verify_value()
# ---------------------------------------------------------------------------


def test_sign_value_verify_value_round_trip() -> None:
    """sign_value() + verify_value() round-trip on an Aleo Value string."""
    a = make_client()
    acct = a.account.create()
    value = "100u64"
    sig = a.account.sign_value(value, acct)
    assert a.account.verify_value(acct.address, value, sig) is True


def test_verify_value_tampered_value_returns_false() -> None:
    """verify_value() returns False for a tampered value."""
    a = make_client()
    acct = a.account.create()
    sig = a.account.sign_value("100u64", acct)
    assert a.account.verify_value(acct.address, "999u64", sig) is False


def test_verify_value_wrong_address_returns_false() -> None:
    """verify_value() returns False when the address does not match."""
    a = make_client()
    acct1 = a.account.create()
    acct2 = a.account.create()
    sig = a.account.sign_value("42u64", acct1)
    assert a.account.verify_value(acct2.address, "42u64", sig) is False


def test_sign_value_accepts_boolean() -> None:
    """sign_value/verify_value work with Aleo boolean values."""
    a = make_client()
    acct = a.account.create()
    sig = a.account.sign_value("true", acct)
    assert a.account.verify_value(acct.address, "true", sig) is True
    assert a.account.verify_value(acct.address, "false", sig) is False


def test_verify_value_accepts_strings() -> None:
    """verify_value accepts address and signature as strings."""
    a = make_client()
    acct = a.account.create()
    sig = a.account.sign_value("7u32", acct)
    assert a.account.verify_value(str(acct.address), "7u32", str(sig)) is True


# ---------------------------------------------------------------------------
# default_account wiring
# ---------------------------------------------------------------------------


def test_sign_uses_default_account_when_omitted() -> None:
    """sign(msg) without an account uses aleo.default_account."""
    a = make_client()
    acct = a.account.create()
    a.default_account = acct
    sig = a.account.sign(b"default signer test")
    assert a.account.verify(acct.address, b"default signer test", sig) is True


def test_sign_value_uses_default_account_when_omitted() -> None:
    """sign_value(value) without an account uses aleo.default_account."""
    a = make_client()
    acct = a.account.create()
    a.default_account = acct
    sig = a.account.sign_value("55u64")
    assert a.account.verify_value(acct.address, "55u64", sig) is True


def test_sign_raises_when_no_account_and_no_default() -> None:
    """sign() raises ValueError when both account and default_account are None."""
    a = make_client()
    with pytest.raises(ValueError, match="default_account"):
        a.account.sign(b"no signer")


def test_sign_value_raises_when_no_account_and_no_default() -> None:
    """sign_value() raises ValueError when both account and default_account are None."""
    a = make_client()
    with pytest.raises(ValueError, match="default_account"):
        a.account.sign_value("1u64")


def test_explicit_account_overrides_default() -> None:
    """Explicit account arg takes precedence over default_account."""
    a = make_client()
    default_acct = a.account.create()
    explicit_acct = a.account.create()
    a.default_account = default_acct

    sig = a.account.sign(b"explicit", explicit_acct)
    # Verifies against the explicit account, not the default
    assert a.account.verify(explicit_acct.address, b"explicit", sig) is True
    assert a.account.verify(default_acct.address, b"explicit", sig) is False


# ---------------------------------------------------------------------------
# Facade import consistency
# ---------------------------------------------------------------------------


def test_account_module_importable_from_facade() -> None:
    """AccountModule is importable directly from aleo.facade."""
    from aleo.facade import AccountModule as AM
    assert AM is AccountModule


def test_account_module_repr() -> None:
    """AccountModule repr names the class and the provider network."""
    a = make_client()
    r = repr(a.account)
    assert "AccountModule" in r
    assert "mainnet" in r
