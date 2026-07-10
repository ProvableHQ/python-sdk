"""
KAT tests mined from the ProvableHQ TypeScript SDK.
Reference: github.com/ProvableHQ/sdk@543b41e0a7e3d6a8a49ae8d0870809dc92b88684

Sources:
  sdk/tests/data/account-data.ts  – seed, keys, ciphertexts
  sdk/tests/data/algebra.ts       – FieldGenerator, expectedPoseidon2Hash
  wasm/src/account/view_key.rs:182-183 – 4th derivation pair
  wasm/src/algorithms/bhp/mod.rs:162  – signature literal
  wasm/src/record/record_plaintext.rs:382-425,504-514 – battleship record
"""

import pytest
from aleo.mainnet import (
    Field,
    Network,
    PrivateKey,
    RecordCiphertext,
    RecordPlaintext,
    Signature,
    ViewKey,
)

# ---------------------------------------------------------------------------
# Seed → private-key KAT
# source: sdk/tests/data/account-data.ts (TS Uint8Array seed → beacon key)
# The TS wasm binding uses Field::from_bytes_le_mod_order(seed_bytes); our
# Python surface exposes PrivateKey.from_seed(Field).  We reconstruct the
# Field by interpreting the 32 seed bytes as a little-endian integer.
# ---------------------------------------------------------------------------

_SEED_BYTES = [
    94, 91, 52, 251, 240, 230, 226, 35, 117, 253, 224, 210, 175, 13, 205, 120,
    155, 214, 7, 169, 66, 62, 206, 50, 188, 40, 29, 122, 40, 250, 54, 18,
]
_SEED_PRIVKEY = "APrivateKey1zkp8CZNn3yeCseEtxuVPbDCwSyhGW6yZKUYKfgXmcpoGPWH"
_SEED_ADDRESS = "aleo1rhgdu77hgyqd3xjj8ucu3jj9r2krwz6mnzyd80gncr5fxcwlh5rsvzp9px"


def test_seed_to_private_key_kat():
    """32-byte LE seed → known private key (account-data.ts beacon key)."""
    n = int.from_bytes(bytes(_SEED_BYTES), "little")
    seed_field = Field.from_string(f"{n}field")
    pk = PrivateKey.from_seed(seed_field)
    assert str(pk) == _SEED_PRIVKEY
    assert str(pk.address) == _SEED_ADDRESS


# ---------------------------------------------------------------------------
# 4th derivation pair
# source: wasm/src/account/view_key.rs:182-183
# ---------------------------------------------------------------------------

_FOURTH_PRIVKEY = "APrivateKey1zkp4RyQ8Utj7aRcJgPQGEok8RMzWwUZzBhhgX6rhmBT8dcP"
_FOURTH_VIEWKEY = "AViewKey1i3fn5SECcVBtQMCVtTPSvdApoMYmg3ToJfNDfgHJAuoD"


def test_fourth_derivation_pair():
    """Private-key → view-key derivation (view_key.rs:182-183)."""
    pk = PrivateKey.from_string(_FOURTH_PRIVKEY)
    assert str(pk.view_key) == _FOURTH_VIEWKEY


# ---------------------------------------------------------------------------
# Foreign record non-ownership KAT
# source: sdk/tests/data/account-data.ts foreignCiphertextString / foreignViewKeyString
# The TS SDK only ever asserts NON-ownership for this ciphertext
# (account.test.ts:163); it never successfully decrypts it with any key.
# Verified against our binding: neither the beacon view key nor the "foreign"
# view key owns it, and decrypt raises (snarkvm 'Insufficient bits').
# ---------------------------------------------------------------------------

_FOREIGN_CT = (
    "record1qyqsq553yxz8ylwqyqfmcfmwz03x6xsxf2h2kypcwhykzgm50ut4sus"
    "yqyxx66trwfhkxun9v35hguerqqpqzqyjt8kxnp28v83t460knvp0dq86a3r3dy"
    "ve945u0xqeksq323paqtegslprdc5zypksrja7rmctx90jnpeq5sqkwlfct7ygy9"
    "90a5pqs7y5pt0"
)
_FOREIGN_VK = "AViewKey1ghtvuJQQzQ31xSiVh6X1PK8biEVhQBygRGV4KdYmq4JT"
_BEACON_VK = "AViewKey1mSnpFFC8Mj4fXbK5YiWgZ3mjiV8CxA79bYNa8ymUpTrw"


def test_foreign_record_ownership_kat():
    """Foreign ciphertext: neither key owns it; decrypt must raise."""
    ct = RecordCiphertext.from_string(_FOREIGN_CT)
    beacon_vk = ViewKey.from_string(_BEACON_VK)
    foreign_vk = ViewKey.from_string(_FOREIGN_VK)
    assert ct.is_owner(beacon_vk) is False
    assert ct.is_owner(foreign_vk) is False
    with pytest.raises(RuntimeError):
        ct.decrypt(foreign_vk)
    with pytest.raises(RuntimeError):
        ct.decrypt(beacon_vk)


# ---------------------------------------------------------------------------
# Poseidon2 hash KAT
# source: sdk/tests/data/algebra.ts + sdk/tests/algorithm.test.ts
# FieldGenerator Fg = 6901184695964460143517399399785179769303979738604374595034454667750561389951field
# Input [Fg, Fg², Fg³, Fg⁴]; hash_psd2 → expectedPoseidon2Hash
# ---------------------------------------------------------------------------

_FG = "6901184695964460143517399399785179769303979738604374595034454667750561389951field"
_POSEIDON2_EXPECTED = "5077032915756006405056357976663159304886914340125619713231037384461532417432field"


def test_poseidon2_hash_kat():
    """Poseidon2 hash of [Fg, Fg^2, Fg^3, Fg^4] equals the algebra.ts vector."""
    Fg = Field.from_string(_FG)
    F2 = Fg * Fg
    F3 = F2 * Fg
    F4 = F3 * Fg
    result = Network.hash_psd2([Fg, F2, F3, F4])
    assert str(result) == _POSEIDON2_EXPECTED


# ---------------------------------------------------------------------------
# Signature literal round-trip
# source: wasm/src/algorithms/bhp/mod.rs:162
# ---------------------------------------------------------------------------

_SIG_LITERAL = (
    "sign1lcpxtgqkp238x45fk79lkx5xz7sx37f56wl0hyemhv78dgzxyspykg6u26l"
    "x2a02tvat6zaflx530qtnme34gh702wclwr20rdxrsqcl7shvwsyhygt2yvkgzeq7"
    "zz2rdat4rrsr0cd9kwm6jddjcs9lps8s80v35rwvtkgg2gxprf4dge0tcet3pe7nf"
    "xupkvfuvh3sw2gpyv0km46"
)


def test_signature_literal_roundtrip():
    """Signature.from_string → str() round-trip and property accessibility."""
    sig = Signature.from_string(_SIG_LITERAL)
    assert str(sig) == _SIG_LITERAL
    # Properties must be accessible and non-empty
    challenge_str = str(sig.challenge)
    response_str = str(sig.response)
    ck_addr_str = str(sig.compute_key.address)
    assert challenge_str.endswith("scalar")
    assert response_str.endswith("scalar")
    assert ck_addr_str.startswith("aleo1")


# ---------------------------------------------------------------------------
# Battleship record – member access
# source: wasm/src/record/record_plaintext.rs:382-425, 504-514
# ---------------------------------------------------------------------------

_BATTLESHIP_RECORD = """{
  owner: aleo1kh5t7m30djl0ecdn4f5vuzp7dx0tcwh7ncquqjkm4matj2p2zqpqm6at48.private,
  metadata: {
    player1: aleo1kh5t7m30djl0ecdn4f5vuzp7dx0tcwh7ncquqjkm4matj2p2zqpqm6at48.private,
    player2: aleo1dreuxnmg9cny8ee9v2u0wr4v4affnwm09u2pytfwz0f2en2shgqsdsfjn6.private,
    nonce: 660310649780728486489183263981322848354071976582883879926426319832534836534field.private
  },
  id: 1953278585719525811355617404139099418855053112960441725284031425961000152405field.private,
  positions: 50794271u64.private,
  attempts: 0u64.private,
  hits: 0u64.private,
  _nonce: 5668100912391182624073500093436664635767788874314097667746354181784048204413group.public,
  _version: 0u8.public
}"""

_BATTLESHIP_OWNER = "aleo1kh5t7m30djl0ecdn4f5vuzp7dx0tcwh7ncquqjkm4matj2p2zqpqm6at48"


def test_battleship_record_owner():
    """RecordPlaintext.owner for the battleship record (record_plaintext.rs:511-514)."""
    rec = RecordPlaintext.from_string(_BATTLESHIP_RECORD)
    # .owner returns "address.private"; strip visibility qualifier
    assert rec.owner.split(".")[0] == _BATTLESHIP_OWNER


def test_battleship_record_nonce_and_version():
    """Battleship record nonce and version fields are correct."""
    rec = RecordPlaintext.from_string(_BATTLESHIP_RECORD)
    assert str(rec.nonce) == (
        "5668100912391182624073500093436664635767788874314097667746354181784048204413group"
    )
    assert rec.version == 0
