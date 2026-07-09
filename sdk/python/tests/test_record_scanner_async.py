"""Tests for AsyncRecordScanner (async, httpx-based). All mocked — no live network."""
from __future__ import annotations

import base64
import json
from typing import Any, Callable

import httpx
import pytest

from aleo.async_record_scanner import AsyncRecordScanner
from aleo._scanner_common import (
    compute_uuid,
    DecryptionNotEnabledError,
    RecordScannerRequestError,
    UUIDError,
    ViewKeyNotStoredError,
)

pytestmark = pytest.mark.asyncio

# ---------------------------------------------------------------------------
# KAT constants
# ---------------------------------------------------------------------------

GOLDEN_PRIVATE_KEY = "APrivateKey1zkp8CZNn3yeCseEtxuVPbDCwSyhGW6yZKUYKfgXmcpoGPWH"
GOLDEN_UUID = "7884164224800444110633570141944665301008802280502652120359195870264061098703field"
VIEW_KEY_STRING = "AViewKey1ccEt8A2Ryva5rxnKcAbn7wgTaTsb79tzkKHFpeKsm9NX"
RECORD_CIPHERTEXT_STRING = (
    "record1qyqsqpe2szk2wwwq56akkwx586hkndl3r8vzdwve32lm7elvphh37rsyqyxx66trwfhkxun9v35hguerqqpqzq"
    "rtjzeu6vah9x2me2exkgege824sd8x2379scspmrmtvczs0d93qttl7y92ga0k0rsexu409hu3vlehe3yxjhmey3frh2z5"
    "pxm5cmxsv4un97q"
)

CHECK_SNS_RESPONSE: dict[str, bool] = {
    "1621694306596217216370326054181178914897851479837084979111511176605457690717field": True,
    "5684626152578699086223993752521225507576791345254401210560771329591763880242field": False,
}

CHECK_TAGS_RESPONSE: dict[str, bool] = {
    "2965517500209150226508265073635793457193572667031485750956287906078711930968field": False,
    "8421937347379608036510120951995833971195343843566214313082589116311107280540field": False,
}

BASE_URL = "https://record-scanner.aleo.org"
HOST = f"{BASE_URL}/mainnet"

RECORD_PLAINTEXT_STR = (
    "{\n  owner: aleo1j7qxyunfldj2lp8hsvy7mw5k8zaqgjfyr72x2gh3x4ewgae8v5gscf5jh3.private,\n"
    "  microcredits: 1500000000000000u64.private,\n"
    "  _nonce: 3077450429259593211617823051143573281856129402760267155982965992208217472983group.public,\n"
    "  _version: 0u8.public\n}"
)

OWNED_CREDITS_RECORDS = [
    {
        "record_ciphertext": RECORD_CIPHERTEXT_STRING,
        "record_plaintext": RECORD_PLAINTEXT_STR,
        "program_name": "credits.aleo",
        "record_name": "credits",
        "spent": False,
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _golden_view_key() -> Any:
    from aleo.mainnet import PrivateKey
    return PrivateKey.from_string(GOLDEN_PRIVATE_KEY).view_key


def _make_nacl_keypair() -> tuple[Any, str]:
    from nacl.public import PrivateKey as NaclPrivateKey
    sk = NaclPrivateKey.generate()
    pk_b64 = base64.b64encode(bytes(sk.public_key)).decode()
    return sk, pk_b64


def route_handler(
    routes: dict[str, Any],
) -> Callable[[httpx.Request], httpx.Response]:
    """Return a handler for httpx.MockTransport."""
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        for pattern, response in routes.items():
            if pattern in url:
                if callable(response):
                    return response(request)
                return response
        return httpx.Response(404, text="not found")
    return handler


def jr(data: Any, status: int = 200, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status, json=data, headers=headers or {})


def _make_scanner(
    routes: dict[str, Any] | None = None,
    **kwargs: Any,
) -> AsyncRecordScanner:
    transport = httpx.MockTransport(route_handler(routes or {}))
    return AsyncRecordScanner(BASE_URL, transport=transport, **kwargs)


# ---------------------------------------------------------------------------
# 1. URL construction
# ---------------------------------------------------------------------------

async def test_async_url_construction() -> None:
    scanner = AsyncRecordScanner("https://record-scanner.aleo.org")
    assert scanner.url == "https://record-scanner.aleo.org/mainnet"


# ---------------------------------------------------------------------------
# 2. Trailing network raises ValueError
# ---------------------------------------------------------------------------

async def test_async_trailing_network_raises() -> None:
    with pytest.raises(ValueError, match="should not include"):
        AsyncRecordScanner("https://record-scanner.aleo.org/mainnet")

    with pytest.raises(ValueError, match="should not include"):
        AsyncRecordScanner("https://record-scanner.aleo.org/testnet")


# ---------------------------------------------------------------------------
# 3. register_encrypted flow
# ---------------------------------------------------------------------------

async def test_async_register_encrypted_flow() -> None:
    """Register flow: GET /pubkey, POST /register/encrypted, ciphertext decrypts OK."""
    sk, pk_b64 = _make_nacl_keypair()

    routes = {
        "/pubkey": jr({"key_id": "k1", "public_key": pk_b64}),
        "/register/encrypted": jr({"uuid": GOLDEN_UUID, "status": None}),
    }

    scanner = _make_scanner(routes)
    vk = _golden_view_key()
    result = await scanner.register_encrypted(vk, 0)

    assert result["ok"] is True
    assert result["data"]["uuid"] == GOLDEN_UUID
    assert scanner._uuid is not None
    assert str(scanner._uuid) == GOLDEN_UUID


async def test_async_register_encrypted_wire_shape() -> None:
    """Ciphertext in POST body decrypts to vk_bytes + LE u32."""
    sk, pk_b64 = _make_nacl_keypair()
    captured: list[httpx.Request] = []

    def capture_register(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return jr({"uuid": GOLDEN_UUID, "status": None})

    routes = {
        "/pubkey": jr({"key_id": "k", "public_key": pk_b64}),
        "/register/encrypted": capture_register,
    }

    scanner = _make_scanner(routes)
    vk = _golden_view_key()
    await scanner.register_encrypted(vk, 42)

    assert len(captured) == 1
    body = json.loads(captured[0].content)
    raw = base64.b64decode(body["ciphertext"])

    # Wire shape: 84 bytes
    assert len(raw) == 84

    from nacl.public import SealedBox
    import struct
    plaintext = SealedBox(sk).decrypt(raw)
    assert len(plaintext) == 36
    assert plaintext[:32] == bytes(vk.bytes())
    assert plaintext[32:] == struct.pack("<I", 42)


# ---------------------------------------------------------------------------
# 4. revoke — body is quoted JSON string
# ---------------------------------------------------------------------------

async def test_async_revoke_body_is_quoted_string() -> None:
    captured: list[httpx.Request] = []

    def capture_revoke(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return jr({"status": "OK"})

    routes = {"/revoke": capture_revoke}
    scanner = _make_scanner(routes)
    from aleo.mainnet import Field
    scanner._uuid = Field.from_string(GOLDEN_UUID)

    result = await scanner.revoke()
    assert result["ok"] is True

    body = captured[0].content.decode()
    assert body == json.dumps(GOLDEN_UUID)
    assert body == f'"{GOLDEN_UUID}"'


# ---------------------------------------------------------------------------
# 5. revoke — no UUID raises UUIDError
# ---------------------------------------------------------------------------

async def test_async_revoke_no_uuid_raises() -> None:
    scanner = _make_scanner()
    with pytest.raises(UUIDError):
        await scanner.revoke()


# ---------------------------------------------------------------------------
# 6. owned — UUID mutation
# ---------------------------------------------------------------------------

async def test_async_owned_uuid_mutation() -> None:
    """Filter with invalid uuid → scanner uuid used; filter mutated."""
    captured: list[httpx.Request] = []

    def capture_owned(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return jr([])

    routes = {"/records/owned": capture_owned}
    scanner = _make_scanner(routes)
    from aleo.mainnet import Field
    scanner._uuid = Field.from_string(GOLDEN_UUID)

    filter_dict: dict[str, Any] = {"uuid": "invalid-uuid", "unspent": True}
    result = await scanner.owned(filter_dict)  # type: ignore[arg-type]

    assert result["ok"] is True
    assert filter_dict["uuid"] == GOLDEN_UUID

    body = json.loads(captured[0].content)
    assert body == filter_dict


# ---------------------------------------------------------------------------
# 7. compute_uuid — golden KAT
# ---------------------------------------------------------------------------

async def test_async_compute_uuid_golden_kat() -> None:
    from aleo.mainnet import PrivateKey
    pk = PrivateKey.from_string(GOLDEN_PRIVATE_KEY)
    vk = pk.view_key
    uuid_field = compute_uuid(vk)
    assert str(uuid_field) == GOLDEN_UUID


# ---------------------------------------------------------------------------
# 8. check_serial_numbers
# ---------------------------------------------------------------------------

async def test_async_check_serial_numbers() -> None:
    routes = {"/records/sns": jr(CHECK_SNS_RESPONSE)}
    scanner = _make_scanner(routes)
    sns = list(CHECK_SNS_RESPONSE.keys())
    result = await scanner.check_serial_numbers(sns)

    assert result["ok"] is True
    assert result["data"] == CHECK_SNS_RESPONSE


# ---------------------------------------------------------------------------
# 9. check_tags
# ---------------------------------------------------------------------------

async def test_async_check_tags() -> None:
    routes = {"/records/tags": jr(CHECK_TAGS_RESPONSE)}
    scanner = _make_scanner(routes)
    tags = list(CHECK_TAGS_RESPONSE.keys())
    result = await scanner.check_tags(tags)

    assert result["ok"] is True
    assert result["data"] == CHECK_TAGS_RESPONSE


# ---------------------------------------------------------------------------
# 10. find_credits_record — decrypt_not_enabled_error
# ---------------------------------------------------------------------------

async def test_async_find_credits_record_decrypt_not_enabled_error() -> None:
    scanner = _make_scanner(decrypt_enabled=False)
    from aleo.mainnet import Field
    scanner._uuid = Field.from_string(GOLDEN_UUID)

    with pytest.raises(DecryptionNotEnabledError):
        await scanner.find_credits_record(1_000_000, {})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Extra: find_credits_record view key not stored
# ---------------------------------------------------------------------------

async def test_async_find_credits_record_view_key_not_stored() -> None:
    scanner = _make_scanner(decrypt_enabled=True)
    from aleo.mainnet import Field
    scanner._uuid = Field.from_string(GOLDEN_UUID)

    with pytest.raises(ViewKeyNotStoredError):
        await scanner.find_credits_record(1_000_000, {})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Extra: revoke invalid UUID raises UUIDError
# ---------------------------------------------------------------------------

async def test_async_revoke_invalid_uuid_raises() -> None:
    scanner = _make_scanner()
    with pytest.raises(UUIDError) as exc_info:
        await scanner.revoke("not-a-valid-field")
    assert exc_info.value.uuid == "not-a-valid-field"


# ---------------------------------------------------------------------------
# Extra: find_record returns first
# ---------------------------------------------------------------------------

async def test_async_find_record_returns_first() -> None:
    owned_records = [
        {"commitment": "cm1abc", "record_ciphertext": RECORD_CIPHERTEXT_STRING},
        {"commitment": "cm1def", "record_ciphertext": RECORD_CIPHERTEXT_STRING},
    ]
    routes = {"/records/owned": jr(owned_records)}
    scanner = _make_scanner(routes)
    from aleo.mainnet import Field
    scanner._uuid = Field.from_string(GOLDEN_UUID)

    record = await scanner.find_record({"uuid": GOLDEN_UUID})  # type: ignore[arg-type]
    assert record == owned_records[0]
