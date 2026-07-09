"""
Tests for the Account class surface.
No TS SDK vector file specific to Account class; vectors reuse existing
records.json KATs (view_key/ciphertext).
"""

import pytest
from aleo.mainnet import Account, PrivateKey, RecordCiphertext, ViewKey
from conftest import load_vectors


_MESSAGE = b"hello world"

# Known private key (from accounts.json triple 0 / account-data.ts beaconKey)
_KNOWN_PRIVKEY = "APrivateKey1zkp8CZNn3yeCseEtxuVPbDCwSyhGW6yZKUYKfgXmcpoGPWH"
_KNOWN_VIEWKEY = "AViewKey1mSnpFFC8Mj4fXbK5YiWgZ3mjiV8CxA79bYNa8ymUpTrw"
_KNOWN_ADDRESS = "aleo1rhgdu77hgyqd3xjj8ucu3jj9r2krwz6mnzyd80gncr5fxcwlh5rsvzp9px"


class TestAccountRandom:
    def test_random_returns_account(self):
        acc = Account.random()
        assert acc is not None

    def test_random_properties_accessible(self):
        acc = Account.random()
        pk_str = str(acc.private_key)
        vk_str = str(acc.view_key)
        addr_str = str(acc.address)
        assert pk_str.startswith("APrivateKey1")
        assert vk_str.startswith("AViewKey1")
        assert addr_str.startswith("aleo1")

    def test_random_unique(self):
        """Two random accounts should (with overwhelming probability) differ."""
        a1 = Account.random()
        a2 = Account.random()
        assert str(a1.private_key) != str(a2.private_key)


class TestAccountFromPrivateKey:
    def test_from_private_key_round_trips(self):
        pk = PrivateKey.from_string(_KNOWN_PRIVKEY)
        acc = Account.from_private_key(pk)
        assert str(acc.private_key) == _KNOWN_PRIVKEY
        assert str(acc.view_key) == _KNOWN_VIEWKEY
        assert str(acc.address) == _KNOWN_ADDRESS

    def test_from_random_private_key(self):
        pk = PrivateKey.random()
        acc = Account.from_private_key(pk)
        assert str(acc.address) == str(pk.address)
        assert str(acc.view_key) == str(pk.view_key)


class TestAccountSignVerify:
    def test_sign_verify_own_message(self):
        acc = Account.random()
        sig = acc.sign(_MESSAGE)
        assert acc.verify(sig, _MESSAGE) is True

    def test_verify_tampered_message_fails(self):
        acc = Account.random()
        sig = acc.sign(_MESSAGE)
        assert acc.verify(sig, b"tampered") is False

    def test_cross_account_verify_fails(self):
        acc1 = Account.random()
        acc2 = Account.random()
        sig = acc1.sign(_MESSAGE)
        assert acc2.verify(sig, _MESSAGE) is False


class TestAccountDecryptIsOwner:
    """Use existing records.json KAT ciphertext to validate decrypt/is_owner."""

    def _kat(self):
        return load_vectors("records.json")["decrypt_kat"]

    def test_is_owner_true(self):
        v = self._kat()
        pk = PrivateKey.from_string(v["private_key"])
        acc = Account.from_private_key(pk)
        ct = RecordCiphertext.from_string(v["ciphertext"])
        assert acc.is_owner(ct) is True

    def test_is_owner_false_for_foreign(self):
        """Beacon account must not own the 'foreign' ciphertext (account-data.ts)."""
        acc = Account.from_private_key(
            PrivateKey.from_string(_KNOWN_PRIVKEY)
        )
        foreign_ct = RecordCiphertext.from_string(
            "record1qyqsq553yxz8ylwqyqfmcfmwz03x6xsxf2h2kypcwhykzgm50ut4sus"
            "yqyxx66trwfhkxun9v35hguerqqpqzqyjt8kxnp28v83t460knvp0dq86a3r3dy"
            "ve945u0xqeksq323paqtegslprdc5zypksrja7rmctx90jnpeq5sqkwlfct7ygy9"
            "90a5pqs7y5pt0"
        )
        assert acc.is_owner(foreign_ct) is False

    def test_decrypt_returns_correct_owner(self):
        v = self._kat()
        pk = PrivateKey.from_string(v["private_key"])
        acc = Account.from_private_key(pk)
        ct = RecordCiphertext.from_string(v["ciphertext"])
        pt = acc.decrypt(ct)
        assert pt.owner.split(".")[0] == v["owner"]

    def test_decrypt_contains_microcredits(self):
        v = self._kat()
        pk = PrivateKey.from_string(v["private_key"])
        acc = Account.from_private_key(pk)
        ct = RecordCiphertext.from_string(v["ciphertext"])
        pt = acc.decrypt(ct)
        assert f"{v['microcredits']}u64" in str(pt)
