"""Tests for ``aleo.account.from_mnemonic``: accounts from a recovery phrase.

The expected keys and addresses come from the shared keystore sample files,
which the other Aleo SDKs derive from the same phrase. All tests are local.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import pytest

import aleo.keystore
from aleo import Aleo, AsyncAleo, HTTPProvider
from aleo.keystore import KeystoreError
from aleo.keystore._mnemonic import check_phrase, clean_phrase, derive_seed, entropy_to_phrase, generate_phrase

from keystore_helpers import VECTORS, check_secret

BASE = "https://api.provable.com/v2"
PHRASE: str = VECTORS["mnemonic"]["phrase"]
ACCOUNTS = VECTORS["mnemonic"]["accounts"]

# The published BIP-39 test phrase for 32 bytes of zero entropy.
PHRASE_24 = " ".join(["abandon"] * 23 + ["art"])

# SHA-256 of the BIP-39 English word list as published with the standard.
WORD_LIST_SHA256 = "2f5eed53a4727b4bf8880d8f3f199efc90e58503646d9ff8eff3a2ed3b24dbda"


def make_client(network: str = "mainnet") -> Aleo:
    return Aleo(HTTPProvider(BASE, network=network))


def invalid(call: Any) -> KeystoreError:
    with pytest.raises(KeystoreError) as raised:
        call()
    assert raised.value.code == "INVALID_MNEMONIC"
    return raised.value


def test_the_sample_files_hold_both_derivations_at_two_indexes() -> None:
    assert len(ACCOUNTS) == 4
    assert sorted((entry["derivation"], entry["index"]) for entry in ACCOUNTS) == [
        ("legacy", 0),
        ("legacy", 1),
        ("standard", 0),
        ("standard", 1),
    ]


@pytest.mark.parametrize("network", ["mainnet", "testnet"])
@pytest.mark.parametrize("expected", ACCOUNTS, ids=lambda entry: f"{entry['derivation']}-{entry['index']}")
def test_derives_every_sample_account(network: str, expected: Dict[str, Any]) -> None:
    client = make_client(network)
    account = client.account.from_mnemonic(PHRASE, index=expected["index"], derivation=expected["derivation"])
    assert str(account.address) == expected["address"]
    check_secret(str(account.private_key), expected["privateKey"])
    # The result is an ordinary account of the client's network.
    signature = client.account.sign(b"hello aleo", account)
    assert client.account.verify(str(account.address), b"hello aleo", signature)


def test_defaults_to_the_first_account_on_the_standard_path() -> None:
    first = next(entry for entry in ACCOUNTS if entry["derivation"] == "standard" and entry["index"] == 0)
    assert str(make_client().account.from_mnemonic(PHRASE).address) == first["address"]


def test_index_and_derivation_are_keyword_only() -> None:
    with pytest.raises(TypeError):
        make_client().account.from_mnemonic(PHRASE, 1)  # type: ignore[misc]


def test_does_not_change_the_default_account() -> None:
    client = make_client()
    client.account.from_mnemonic(PHRASE)
    assert client.default_account is None


def test_is_available_on_the_async_client() -> None:
    first = next(entry for entry in ACCOUNTS if entry["derivation"] == "standard" and entry["index"] == 0)
    client = AsyncAleo(HTTPProvider(BASE, network="testnet"))
    assert str(client.account.from_mnemonic(PHRASE).address) == first["address"]


def test_derives_from_a_24_word_phrase_and_a_high_index() -> None:
    client = make_client()
    first = client.account.from_mnemonic(PHRASE_24)
    again = client.account.from_mnemonic(PHRASE_24)
    assert str(first.address) == str(again.address)
    assert str(first.address) != str(client.account.from_mnemonic(PHRASE).address)
    last = client.account.from_mnemonic(PHRASE_24, index=2**31 - 1)
    assert str(last.address) != str(first.address)


def test_follows_the_published_derivation_steps() -> None:
    """An independent walk through the derivation gives the same seed."""
    import hmac

    seed = hashlib.pbkdf2_hmac("sha512", PHRASE.encode("utf-8"), b"mnemonic", 2048, 64)
    node = hmac.new(b"bls12_377 seed", seed, hashlib.sha512).digest()
    for segment in (44, 683, 1, 0):
        data = b"\x00" + node[:32] + (segment + 2**31).to_bytes(4, "big")
        node = hmac.new(node[32:], data, hashlib.sha512).digest()
    assert derive_seed(PHRASE, index=1) == node[:32]
    assert len(derive_seed(PHRASE, derivation="legacy")) == 32


@pytest.mark.parametrize(
    "phrase",
    [
        pytest.param(" ".join(PHRASE.split()[:-1] + ["abandon"]), id="a wrong checksum"),
        pytest.param(" ".join(PHRASE.split()[:-1]), id="eleven words"),
        pytest.param(PHRASE + " abandon", id="thirteen words"),
        pytest.param(" ".join(PHRASE.split()[:-1] + ["zzzzzz"]), id="a word outside the list"),
        pytest.param(PHRASE.upper(), id="upper case"),
        pytest.param(PHRASE.replace(" ", "  ", 1), id="two spaces"),
        pytest.param(" " + PHRASE, id="a leading space"),
        pytest.param(PHRASE + "\n", id="a trailing line ending"),
        pytest.param(PHRASE.replace(" ", "\t"), id="tabs"),
        pytest.param("", id="nothing"),
    ],
)
def test_rejects_an_invalid_phrase_without_repeating_it(phrase: str) -> None:
    error = invalid(lambda: make_client().account.from_mnemonic(phrase))
    assert isinstance(error, aleo.AleoError)
    assert "abandon" not in str(error)
    assert "zzzzzz" not in str(error)
    assert "ABANDON" not in str(error)


@pytest.mark.parametrize("phrase", [None, 7, b"abandon", ["abandon"] * 12])
def test_rejects_a_phrase_that_is_not_text(phrase: Any) -> None:
    invalid(lambda: make_client().account.from_mnemonic(phrase))


@pytest.mark.parametrize("index", [-1, 2**31, True, 1.5, "0", None])
def test_rejects_an_index_out_of_range(index: Any) -> None:
    with pytest.raises(ValueError):
        make_client().account.from_mnemonic(PHRASE, index=index)


@pytest.mark.parametrize("derivation", ["bip44", "", "Standard", None])
def test_rejects_an_unknown_derivation(derivation: Any) -> None:
    with pytest.raises(ValueError):
        make_client().account.from_mnemonic(PHRASE, derivation=derivation)


def test_accepts_every_standard_phrase_length() -> None:
    # The published all-zero-entropy phrases: "abandon" repeated, then the
    # one word that completes the checksum.
    for count, last in ((12, "about"), (15, "address"), (18, "agent"), (21, "admit"), (24, "art")):
        check_phrase(" ".join(["abandon"] * (count - 1) + [last]))


def test_tidies_a_pasted_phrase() -> None:
    assert clean_phrase("  Abandon\tABOUT \n art  ") == "abandon about art"


def test_ships_the_bip39_english_word_list_beside_the_module() -> None:
    path = Path(aleo.keystore.__file__).with_name("_bip39_english.txt")
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == WORD_LIST_SHA256
    words = data.decode("ascii").split()
    assert len(words) == 2048
    assert words[0] == "abandon" and words[-1] == "zoo"
    assert words == sorted(words)


def test_sample_files_are_unmodified_json() -> None:
    """The mnemonic vectors parse and name the phrase they were derived from."""
    path = Path(__file__).parent / "fixtures" / "keystore-vectors" / "vectors.json"
    assert json.loads(path.read_text(encoding="utf-8"))["mnemonic"]["phrase"] == PHRASE


# ---------------------------------------------------------------------------
# Creating a new phrase
# ---------------------------------------------------------------------------

# Published BIP-39 English test vectors: entropy (hex) and the phrase it encodes.
BIP39_VECTORS = [
    ("00" * 16, "abandon " * 11 + "about"),
    ("7f" * 16, "legal winner thank year wave sausage worth useful legal winner thank yellow"),
    ("80" * 16, "letter advice cage absurd amount doctor acoustic avoid letter advice cage above"),
    ("ff" * 16, "zoo " * 11 + "wrong"),
    ("00" * 32, "abandon " * 23 + "art"),
    ("ff" * 32, "zoo " * 23 + "vote"),
]


@pytest.mark.parametrize("entropy, phrase", BIP39_VECTORS)
def test_encodes_the_published_bip39_vectors(entropy: str, phrase: str) -> None:
    assert entropy_to_phrase(bytes.fromhex(entropy)) == phrase
    check_phrase(phrase)


def test_encodes_the_sample_phrase() -> None:
    assert entropy_to_phrase(bytes(16)) == VECTORS["mnemonic"]["phrase"]


@pytest.mark.parametrize("length", [0, 15, 17, 33, 64])
def test_rejects_entropy_of_another_length(length: int) -> None:
    with pytest.raises(ValueError):
        entropy_to_phrase(bytes(length))


@pytest.mark.parametrize("strength, words", [(128, 12), (256, 24)])
def test_generates_a_valid_phrase_of_each_strength(strength: int, words: int) -> None:
    aleo_client = Aleo(HTTPProvider("http://127.0.0.1:1"))
    phrase = aleo_client.account.generate_mnemonic(strength)
    assert len(phrase.split(" ")) == words
    assert phrase == clean_phrase(phrase)
    check_phrase(phrase)


def test_generates_twelve_words_by_default_and_a_new_phrase_each_time() -> None:
    aleo_client = Aleo(HTTPProvider("http://127.0.0.1:1"))
    phrases = {aleo_client.account.generate_mnemonic() for _ in range(8)}
    assert len(phrases) == 8
    assert all(len(phrase.split(" ")) == 12 for phrase in phrases)
    assert len(generate_phrase().split(" ")) == 12


@pytest.mark.parametrize("strength", [0, 127, 160, 192, 224, 512, True, "128", None, 128.0])
def test_rejects_another_strength(strength: Any) -> None:
    aleo_client = Aleo(HTTPProvider("http://127.0.0.1:1"))
    with pytest.raises(ValueError):
        aleo_client.account.generate_mnemonic(strength)
    with pytest.raises(ValueError):
        aleo_client.account.create_with_mnemonic(strength=strength)


@pytest.mark.parametrize("network", ["testnet", "mainnet"])
def test_creates_an_account_with_its_phrase(network: str) -> None:
    aleo_client = Aleo(HTTPProvider("http://127.0.0.1:1", network=network))
    phrase, account = aleo_client.account.create_with_mnemonic()
    assert len(phrase.split(" ")) == 12
    check_phrase(phrase)
    again = aleo_client.account.from_mnemonic(phrase)
    assert str(again.address) == str(account.address)
    check_secret(str(again.private_key), str(account.private_key))


def test_creates_at_an_index_and_derivation_with_a_longer_phrase() -> None:
    aleo_client = Aleo(HTTPProvider("http://127.0.0.1:1"))
    phrase, account = aleo_client.account.create_with_mnemonic(strength=256, index=3, derivation="legacy")
    assert len(phrase.split(" ")) == 24
    assert str(account.address) == str(aleo_client.account.from_mnemonic(phrase, index=3, derivation="legacy").address)
    assert str(account.address) != str(aleo_client.account.from_mnemonic(phrase).address)


def test_creating_with_a_phrase_takes_keywords_only_and_leaves_the_default_account() -> None:
    aleo_client = Aleo(HTTPProvider("http://127.0.0.1:1"))
    with pytest.raises(TypeError):
        aleo_client.account.create_with_mnemonic(256)  # type: ignore[misc]
    aleo_client.account.create_with_mnemonic()
    assert aleo_client.default_account is None


def test_creating_with_a_phrase_is_available_on_the_async_client() -> None:
    client = AsyncAleo(HTTPProvider("http://127.0.0.1:1"))
    phrase, account = client.account.create_with_mnemonic()
    assert str(client.account.from_mnemonic(phrase).address) == str(account.address)
