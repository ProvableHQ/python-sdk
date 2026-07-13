"""Tests for RecordScanner (sync, requests-based). All mocked — no live network."""
from __future__ import annotations

import base64
import json
from typing import Any

import pytest
import responses as resp_lib

from aleo.record_scanner import RecordScanner
from aleo._scanner_common import (
    compute_uuid,
    uuid_is_valid,
    DecryptionNotEnabledError,
    RecordNotFoundError,
    RecordScannerRequestError,
    UUIDError,
    ViewKeyNotStoredError,
)

# ---------------------------------------------------------------------------
# KAT constants (from TS SDK tests/data/records.ts)
# ---------------------------------------------------------------------------

GOLDEN_PRIVATE_KEY = "APrivateKey1zkp8CZNn3yeCseEtxuVPbDCwSyhGW6yZKUYKfgXmcpoGPWH"
GOLDEN_UUID = "7884164224800444110633570141944665301008802280502652120359195870264061098703field"
VIEW_KEY_STRING = "AViewKey1ccEt8A2Ryva5rxnKcAbn7wgTaTsb79tzkKHFpeKsm9NX"
RECORD_CIPHERTEXT_STRING = (
    "record1qyqsqpe2szk2wwwq56akkwx586hkndl3r8vzdwve32lm7elvphh37rsyqyxx66trwfhkxun9v35hguerqqpqzq"
    "rtjzeu6vah9x2me2exkgege824sd8x2379scspmrmtvczs0d93qttl7y92ga0k0rsexu409hu3vlehe3yxjhmey3frh2z5"
    "pxm5cmxsv4un97q"
)
RECORD_VIEW_KEY_STRING = "4445718830394614891114647247073357094867447866913203502139893824059966201724field"

CHECK_SNS_RESPONSE: dict[str, bool] = {
    "1621694306596217216370326054181178914897851479837084979111511176605457690717field": True,
    "5684626152578699086223993752521225507576791345254401210560771329591763880242field": False,
}

CHECK_TAGS_RESPONSE: dict[str, bool] = {
    "2965517500209150226508265073635793457193572667031485750956287906078711930968field": False,
    "8421937347379608036510120951995833971195343843566214313082589116311107280540field": False,
    "5941252181432651644402279701137165256963073258332916685063623109173576520831field": False,
}

BASE_URL = "https://record-scanner.aleo.org"
HOST = f"{BASE_URL}/mainnet"

# Real plaintext from decrypting RECORD_CIPHERTEXT_STRING with VIEW_KEY_STRING
RECORD_PLAINTEXT_STR = (
    "{\n  owner: aleo1j7qxyunfldj2lp8hsvy7mw5k8zaqgjfyr72x2gh3x4ewgae8v5gscf5jh3.private,\n"
    "  microcredits: 1500000000000000u64.private,\n"
    "  _nonce: 3077450429259593211617823051143573281856129402760267155982965992208217472983group.public,\n"
    "  _version: 0u8.public\n}"
)

# Owned records fixture (two credits records with plaintext, different microcredits)
OWNED_CREDITS_RECORDS = [
    {
        "record_ciphertext": RECORD_CIPHERTEXT_STRING,
        "record_plaintext": RECORD_PLAINTEXT_STR,
        "program_name": "credits.aleo",
        "record_name": "credits",
        "spent": False,
    },
    {
        "record_ciphertext": RECORD_CIPHERTEXT_STRING,
        "record_plaintext": RECORD_PLAINTEXT_STR,
        "program_name": "credits.aleo",
        "record_name": "credits",
        "spent": False,
    },
]

OWNED_RECORDS = [
    {"commitment": "cm1abc", "record_ciphertext": RECORD_CIPHERTEXT_STRING, "spent": False},
    {"commitment": "cm1def", "record_ciphertext": RECORD_CIPHERTEXT_STRING, "spent": False},
]

# Pubkey fixture for registration tests
PUBKEY_FIXTURE: dict[str, str] = {
    "key_id": "test-key-id-001",
    "public_key": "",  # filled in per-test with a generated key
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _golden_view_key() -> Any:
    from aleo.mainnet import PrivateKey
    return PrivateKey.from_string(GOLDEN_PRIVATE_KEY).view_key


def _make_nacl_keypair() -> tuple[Any, str]:
    """Return (nacl.PrivateKey, base64 public key)."""
    from nacl.public import PrivateKey as NaclPrivateKey
    sk = NaclPrivateKey.generate()
    pk_b64 = base64.b64encode(bytes(sk.public_key)).decode()
    return sk, pk_b64


def _make_scanner(**kwargs: Any) -> RecordScanner:
    return RecordScanner(BASE_URL, **kwargs)


# ---------------------------------------------------------------------------
# 1. URL construction
# ---------------------------------------------------------------------------

def test_url_construction() -> None:
    scanner = RecordScanner("https://record-scanner.aleo.org")
    assert scanner.url == "https://record-scanner.aleo.org/mainnet"


def test_url_construction_custom_network() -> None:
    scanner = RecordScanner("https://record-scanner.aleo.org", network="testnet")
    assert scanner.url == "https://record-scanner.aleo.org/testnet"


# ---------------------------------------------------------------------------
# 2. Trailing network raises ValueError
# ---------------------------------------------------------------------------

def test_trailing_network_raises_mainnet() -> None:
    with pytest.raises(ValueError, match="should not include"):
        RecordScanner("https://record-scanner.aleo.org/mainnet")


def test_trailing_network_raises_testnet() -> None:
    with pytest.raises(ValueError, match="should not include"):
        RecordScanner("https://record-scanner.aleo.org/testnet")


# ---------------------------------------------------------------------------
# 3. API key normalization — string
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_api_key_string() -> None:
    """api_key as string → X-Provable-API-Key header on requests."""
    resp_lib.add(resp_lib.GET, f"{HOST}/pubkey", json={"key_id": "k", "public_key": "dGVzdA=="})
    resp_lib.add(resp_lib.POST, f"{HOST}/register/encrypted", json={"uuid": GOLDEN_UUID, "status": None})

    scanner = _make_scanner(api_key="1234567890")
    vk = _golden_view_key()

    # Use a real nacl key so encryption works
    sk, pk_b64 = _make_nacl_keypair()
    resp_lib.reset()
    resp_lib.add(resp_lib.GET, f"{HOST}/pubkey", json={"key_id": "k", "public_key": pk_b64})
    resp_lib.add(resp_lib.POST, f"{HOST}/register/encrypted", json={"uuid": GOLDEN_UUID, "status": None})

    result = scanner.register_encrypted(vk, 0)
    assert result["ok"] is True

    # Check GET /pubkey had api key header
    get_req = resp_lib.calls[0].request
    assert get_req.headers.get("X-Provable-API-Key") == "1234567890"


# ---------------------------------------------------------------------------
# 4. API key normalization — custom tuple
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_api_key_custom_tuple() -> None:
    """api_key as (header, value) tuple → correct header on requests."""
    sk, pk_b64 = _make_nacl_keypair()
    resp_lib.add(resp_lib.GET, f"{HOST}/pubkey", json={"key_id": "k", "public_key": pk_b64})
    resp_lib.add(resp_lib.POST, f"{HOST}/register/encrypted", json={"uuid": GOLDEN_UUID, "status": None})

    scanner = _make_scanner(api_key=("Some-API-Key", "myval"))
    vk = _golden_view_key()
    result = scanner.register_encrypted(vk, 0)
    assert result["ok"] is True

    get_req = resp_lib.calls[0].request
    assert get_req.headers.get("Some-API-Key") == "myval"


# ---------------------------------------------------------------------------
# 5. register_encrypted flow
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_register_encrypted_flow() -> None:
    """Register flow: GET /pubkey, POST /register/encrypted, check ciphertext decrypts."""
    sk, pk_b64 = _make_nacl_keypair()
    resp_lib.add(resp_lib.GET, f"{HOST}/pubkey", json={"key_id": "k1", "public_key": pk_b64})
    resp_lib.add(
        resp_lib.POST,
        f"{HOST}/register/encrypted",
        json={"uuid": GOLDEN_UUID, "status": None},
    )

    scanner = _make_scanner()
    vk = _golden_view_key()
    result = scanner.register_encrypted(vk, 0)

    assert result["ok"] is True
    assert result["data"]["uuid"] == GOLDEN_UUID

    # Verify URL and method
    assert "/pubkey" in resp_lib.calls[0].request.url
    assert resp_lib.calls[0].request.method == "GET"
    assert "/register/encrypted" in resp_lib.calls[1].request.url
    assert resp_lib.calls[1].request.method == "POST"
    assert resp_lib.calls[1].request.headers.get("Content-Type") == "application/json"

    # Verify ciphertext decrypts to vk_bytes + LE u32
    post_body = json.loads(resp_lib.calls[1].request.body)
    ciphertext_b64 = post_body["ciphertext"]
    raw = base64.b64decode(ciphertext_b64)

    # Wire shape: 32 (epk) + 16 (MAC) + 36 (plaintext) = 84 bytes
    assert len(raw) == 84

    from nacl.public import SealedBox
    plaintext = SealedBox(sk).decrypt(raw)
    assert len(plaintext) == 36

    vk_bytes = bytes(vk.bytes())
    import struct
    start_bytes = struct.pack("<I", 0)
    assert plaintext == vk_bytes + start_bytes


@resp_lib.activate
def test_register_encrypted_distinct_ciphertexts() -> None:
    """Each register call produces a distinct ciphertext (ephemeral key)."""
    sk, pk_b64 = _make_nacl_keypair()
    resp_lib.add(resp_lib.GET, f"{HOST}/pubkey", json={"key_id": "k", "public_key": pk_b64})
    resp_lib.add(resp_lib.POST, f"{HOST}/register/encrypted", json={"uuid": GOLDEN_UUID, "status": None})
    resp_lib.add(resp_lib.GET, f"{HOST}/pubkey", json={"key_id": "k", "public_key": pk_b64})
    resp_lib.add(resp_lib.POST, f"{HOST}/register/encrypted", json={"uuid": GOLDEN_UUID, "status": None})

    scanner = _make_scanner()
    vk = _golden_view_key()
    scanner.register_encrypted(vk, 0)
    scanner.register_encrypted(vk, 0)

    ct1 = json.loads(resp_lib.calls[1].request.body)["ciphertext"]
    ct2 = json.loads(resp_lib.calls[3].request.body)["ciphertext"]
    assert ct1 != ct2


# ---------------------------------------------------------------------------
# 6. revoke — body is quoted JSON string
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_revoke_body_is_quoted_string() -> None:
    """revoke() POSTs a quoted JSON string body like '"uuid"'."""
    resp_lib.add(resp_lib.POST, f"{HOST}/revoke", json={"status": "OK"})

    scanner = _make_scanner()
    scanner._uuid = None
    # Manually set UUID using a real field
    from aleo.mainnet import Field
    scanner._uuid = Field.from_string(GOLDEN_UUID)

    result = scanner.revoke()

    assert result["ok"] is True
    assert result["data"]["status"] == "OK"

    post_body = resp_lib.calls[0].request.body
    # Body should be the uuid string quoted as JSON
    assert post_body == json.dumps(GOLDEN_UUID)
    assert post_body == f'"{GOLDEN_UUID}"'


# ---------------------------------------------------------------------------
# 7. revoke — no UUID raises UUIDError (no HTTP call)
# ---------------------------------------------------------------------------

def test_revoke_no_uuid_raises() -> None:
    scanner = _make_scanner()
    with pytest.raises(UUIDError):
        scanner.revoke()


# ---------------------------------------------------------------------------
# 8. revoke — invalid UUID raises UUIDError (no HTTP call)
# ---------------------------------------------------------------------------

def test_revoke_invalid_uuid_raises() -> None:
    scanner = _make_scanner()
    with pytest.raises(UUIDError) as exc_info:
        scanner.revoke("not-a-valid-field")
    assert exc_info.value.uuid == "not-a-valid-field"


# ---------------------------------------------------------------------------
# 9. status — body and shape
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_status_body_and_shape() -> None:
    resp_lib.add(
        resp_lib.POST,
        f"{HOST}/status",
        json={"synced": True, "percentage": 100.0},
    )

    scanner = _make_scanner()
    from aleo.mainnet import Field
    scanner._uuid = Field.from_string(GOLDEN_UUID)

    result = scanner.status()

    assert result["ok"] is True
    assert result["data"]["synced"] is True
    assert result["data"]["percentage"] == 100.0

    # Body is quoted uuid string
    body = resp_lib.calls[0].request.body
    assert body == json.dumps(GOLDEN_UUID)


# ---------------------------------------------------------------------------
# 10. owned — UUID resolution and filter mutation
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_owned_uuid_resolution_and_mutation() -> None:
    """Filter with invalid uuid → scanner's uuid used; filter mutated."""
    resp_lib.add(resp_lib.POST, f"{HOST}/records/owned", json=[])

    scanner = _make_scanner()
    from aleo.mainnet import Field
    scanner._uuid = Field.from_string(GOLDEN_UUID)

    filter_dict: dict[str, Any] = {"uuid": "invalid-uuid", "unspent": True}
    result = scanner.owned(filter_dict)  # type: ignore[arg-type]

    assert result["ok"] is True
    # Filter was mutated to valid uuid
    assert filter_dict["uuid"] == GOLDEN_UUID

    # Body == json.dumps(filter) after mutation
    body = resp_lib.calls[0].request.body
    assert json.loads(body) == filter_dict


@resp_lib.activate
def test_owned_refreshes_jwt_on_auth_failure() -> None:
    """owned() re-mints the JWT and retries once when the scanner reports the
    JWT invalidated out-of-band (shared-consumer rotation: prover mint kills the
    scanner's JWT → 'No credentials found for given iss')."""
    from aleo.mainnet import Field

    # Two JWT mints: the initial one, then the forced refresh after the 401.
    resp_lib.add(
        resp_lib.POST, f"{BASE_URL}/jwts/cid",
        headers={"Authorization": "Bearer J1"}, json={"exp": 9999999999},
    )
    resp_lib.add(
        resp_lib.POST, f"{BASE_URL}/jwts/cid",
        headers={"Authorization": "Bearer J2"}, json={"exp": 9999999999},
    )
    # owned: first 401 with the iss error body, then 200.
    resp_lib.add(
        resp_lib.POST, f"{HOST}/records/owned", status=401,
        json={"message": "No credentials found for given 'iss'"},
    )
    resp_lib.add(resp_lib.POST, f"{HOST}/records/owned", json=[])

    scanner = RecordScanner(BASE_URL, network="mainnet", api_key="ak", consumer_id="cid")
    scanner._uuid = Field.from_string(GOLDEN_UUID)

    result = scanner.owned({"uuid": GOLDEN_UUID, "unspent": True})  # type: ignore[arg-type]

    assert result["ok"] is True
    owned_calls = [x for x in resp_lib.calls if "/records/owned" in x.request.url]
    assert len(owned_calls) == 2  # 401 then a fresh-JWT retry
    assert owned_calls[0].request.headers.get("Authorization") == "Bearer J1"
    assert owned_calls[1].request.headers.get("Authorization") == "Bearer J2"


# ---------------------------------------------------------------------------
# 11. owned — verbatim body with complex OwnedFilter
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_owned_verbatim_body() -> None:
    """Complex OwnedFilter → body == json.dumps(filter) after uuid mutation."""
    resp_lib.add(resp_lib.POST, f"{HOST}/records/owned", json=[])

    scanner = _make_scanner()
    from aleo.mainnet import Field
    scanner._uuid = Field.from_string(GOLDEN_UUID)

    filter_dict: dict[str, Any] = {
        "uuid": "test-uuid",  # invalid — scanner's uuid will be used
        "unspent": True,
        "filter": {"program": "credits.aleo", "record": "credits"},
        "responseFilter": {"record_ciphertext": True, "spent": True},
    }

    result = scanner.owned(filter_dict)  # type: ignore[arg-type]
    assert result["ok"] is True

    # After owned(), filter["uuid"] == GOLDEN_UUID
    assert filter_dict["uuid"] == GOLDEN_UUID

    body = json.loads(resp_lib.calls[0].request.body)
    assert body == filter_dict


# ---------------------------------------------------------------------------
# 12. owned — 422 auto re-register
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_owned_422_auto_reregister() -> None:
    """On 422 with auto_re_register=True + stored vk → re-register and retry."""
    from aleo.mainnet import Field
    sk, pk_b64 = _make_nacl_keypair()

    # First owned call → 422
    resp_lib.add(resp_lib.POST, f"{HOST}/records/owned", status=422, json={})
    # pubkey for re-registration
    resp_lib.add(resp_lib.GET, f"{HOST}/pubkey", json={"key_id": "k", "public_key": pk_b64})
    # register/encrypted
    resp_lib.add(resp_lib.POST, f"{HOST}/register/encrypted", json={"uuid": GOLDEN_UUID, "status": None})
    # Second owned call → 200
    resp_lib.add(resp_lib.POST, f"{HOST}/records/owned", json=OWNED_RECORDS)

    scanner = _make_scanner(auto_re_register=True)
    vk = _golden_view_key()
    scanner._uuid = Field.from_string(GOLDEN_UUID)
    scanner._view_keys[GOLDEN_UUID] = vk

    filter_dict: dict[str, Any] = {"uuid": GOLDEN_UUID}
    result = scanner.owned(filter_dict)  # type: ignore[arg-type]

    assert result["ok"] is True
    assert len(result["data"]) == 2

    # Verify register was called (pubkey + register/encrypted calls exist)
    urls = [c.request.url for c in resp_lib.calls]
    assert any("/pubkey" in u for u in urls)
    assert any("/register/encrypted" in u for u in urls)


# ---------------------------------------------------------------------------
# 13. encrypted_records — filter passthrough
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_encrypted_records_filter_passthrough() -> None:
    records_filter = {"programs": ["credits.aleo"], "start": 0, "end": 100}
    resp_lib.add(resp_lib.POST, f"{HOST}/records/encrypted", json=[{"record_ciphertext": "ct1"}])

    scanner = _make_scanner()
    result = scanner.encrypted(records_filter)  # type: ignore[arg-type]

    assert result["ok"] is True
    assert result["data"][0]["record_ciphertext"] == "ct1"

    assert "/records/encrypted" in resp_lib.calls[0].request.url
    body = json.loads(resp_lib.calls[0].request.body)
    assert body == records_filter


# ---------------------------------------------------------------------------
# 14. check_serial_numbers
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_check_serial_numbers() -> None:
    resp_lib.add(resp_lib.POST, f"{HOST}/records/sns", json=CHECK_SNS_RESPONSE)

    scanner = _make_scanner()
    sns = list(CHECK_SNS_RESPONSE.keys())
    result = scanner.check_serial_numbers(sns)

    assert result["ok"] is True
    assert result["data"] == CHECK_SNS_RESPONSE
    assert "/records/sns" in resp_lib.calls[0].request.url
    assert json.loads(resp_lib.calls[0].request.body) == sns


# ---------------------------------------------------------------------------
# 15. check_tags
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_check_tags() -> None:
    resp_lib.add(resp_lib.POST, f"{HOST}/records/tags", json=CHECK_TAGS_RESPONSE)

    scanner = _make_scanner()
    tags = list(CHECK_TAGS_RESPONSE.keys())
    result = scanner.check_tags(tags)

    assert result["ok"] is True
    assert result["data"] == CHECK_TAGS_RESPONSE
    assert "/records/tags" in resp_lib.calls[0].request.url
    assert json.loads(resp_lib.calls[0].request.body) == tags


# ---------------------------------------------------------------------------
# 16. compute_uuid — golden KAT
# ---------------------------------------------------------------------------

def test_compute_uuid_golden_kat() -> None:
    from aleo.mainnet import PrivateKey
    pk = PrivateKey.from_string(GOLDEN_PRIVATE_KEY)
    vk = pk.view_key
    uuid_field = compute_uuid(vk)
    assert str(uuid_field) == GOLDEN_UUID


# ---------------------------------------------------------------------------
# 17. uuid_is_valid
# ---------------------------------------------------------------------------

def test_uuid_is_valid_true() -> None:
    assert uuid_is_valid(GOLDEN_UUID) is True


def test_uuid_is_valid_false() -> None:
    assert uuid_is_valid("not-a-uuid") is False
    assert uuid_is_valid("") is False


# ---------------------------------------------------------------------------
# 18. JWT origin table
# ---------------------------------------------------------------------------

def test_jwt_origin_table() -> None:
    """JWT refresh goes to scheme+host (no path)."""
    from aleo._client_common import jwt_origin
    cases = [
        ("https://api.provable.com/v2", "https://api.provable.com"),
        ("https://record-scanner.aleo.org/mainnet", "https://record-scanner.aleo.org"),
        ("http://localhost:8080/some/path", "http://localhost:8080"),
        ("https://scanner.example.com", "https://scanner.example.com"),
    ]
    for url, expected in cases:
        assert jwt_origin(url) == expected, f"jwt_origin({url!r}) != {expected!r}"


@resp_lib.activate
def test_jwt_origin_used_for_jwt_refresh() -> None:
    """JWT POST goes to origin/jwts/{consumer_id}, not the scanner url."""
    jwt_url = "https://record-scanner.aleo.org/jwts/consumer123"
    resp_lib.add(
        resp_lib.POST,
        jwt_url,
        json={"exp": 9999999999},
        headers={"Authorization": "Bearer test-jwt"},
    )
    resp_lib.add(resp_lib.POST, f"{HOST}/revoke", json={"status": "OK"})

    scanner = _make_scanner(
        api_key="apikey",
        consumer_id="consumer123",
    )
    from aleo.mainnet import Field
    scanner._uuid = Field.from_string(GOLDEN_UUID)
    # Force JWT refresh by leaving jwt_data as None
    scanner.jwt_data = None

    result = scanner.revoke()
    assert result["ok"] is True

    jwt_call = resp_lib.calls[0]
    assert "jwts/consumer123" in jwt_call.request.url
    assert "record-scanner.aleo.org" in jwt_call.request.url
    # Must NOT include /mainnet path
    assert "/mainnet" not in jwt_call.request.url


# ---------------------------------------------------------------------------
# 19. find_credits_record — decrypt_not_enabled_error
# ---------------------------------------------------------------------------

def test_find_credits_record_decrypt_not_enabled_error() -> None:
    scanner = _make_scanner(decrypt_enabled=False)
    from aleo.mainnet import Field
    scanner._uuid = Field.from_string(GOLDEN_UUID)

    with pytest.raises(DecryptionNotEnabledError):
        scanner.find_credits_record(1_000_000, {})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 20. find_credits_record — view_key_not_stored_error
# ---------------------------------------------------------------------------

def test_find_credits_record_view_key_not_stored_error() -> None:
    scanner = _make_scanner(decrypt_enabled=True)
    from aleo.mainnet import Field
    scanner._uuid = Field.from_string(GOLDEN_UUID)
    # No view key stored

    with pytest.raises(ViewKeyNotStoredError) as exc_info:
        scanner.find_credits_record(1_000_000, {})  # type: ignore[arg-type]
    assert exc_info.value.uuid == GOLDEN_UUID


# ---------------------------------------------------------------------------
# 21. find_credits_record — success
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_find_credits_record_success() -> None:
    """find_credits_record returns first record with >= microcredits."""
    resp_lib.add(resp_lib.POST, f"{HOST}/records/owned", json=OWNED_CREDITS_RECORDS)

    scanner = _make_scanner(decrypt_enabled=True)
    from aleo.mainnet import Field
    scanner._uuid = Field.from_string(GOLDEN_UUID)
    vk = _golden_view_key()
    scanner._view_keys[GOLDEN_UUID] = vk

    # RECORD_PLAINTEXT_STR has 1_500_000_000_000_000 microcredits
    result = scanner.find_credits_record(1_000_000, {})  # type: ignore[arg-type]
    assert "record_plaintext" in result


# ---------------------------------------------------------------------------
# 22. decrypt in-place
# ---------------------------------------------------------------------------

def test_decrypt_in_place() -> None:
    from aleo.mainnet import ViewKey
    vk = ViewKey.from_string(VIEW_KEY_STRING)
    scanner = _make_scanner()

    records: list[dict[str, Any]] = [
        {"record_ciphertext": RECORD_CIPHERTEXT_STRING},
        {"record_ciphertext": ""},  # empty — should be skipped
        {"record_ciphertext": "invalid-ct"},  # invalid — silently skipped
    ]
    scanner.decrypt(vk, records)

    # First record decrypted
    assert "record_plaintext" in records[0]
    assert "microcredits" in records[0]["record_plaintext"]
    # Second record: no plaintext set (empty ciphertext)
    assert "record_plaintext" not in records[1]
    # Third record: silently skipped (bad ciphertext)
    assert "record_plaintext" not in records[2]


# ---------------------------------------------------------------------------
# 23. find_record — returns first
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_find_record_returns_first() -> None:
    resp_lib.add(resp_lib.POST, f"{HOST}/records/owned", json=OWNED_RECORDS)

    scanner = _make_scanner()
    from aleo.mainnet import Field
    scanner._uuid = Field.from_string(GOLDEN_UUID)

    record = scanner.find_record({"uuid": GOLDEN_UUID})  # type: ignore[arg-type]
    assert record == OWNED_RECORDS[0]
