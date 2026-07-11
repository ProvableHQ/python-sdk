"""Tests for F6 facade: aleo.records + RecordProvider (aleo.facade.records).

All tests are FAST and OFFLINE. The delegated RecordScanner's HTTP endpoints are
mocked with ``responses`` (reusing the idiom from test_record_scanner.py); the
scanner is injected onto the module via ``aleo.records.scanner = ...`` so no live
scanner service is contacted. The F5 integration test injects a FAKE
RecordProvider to prove the private-fee seam in call.py is closed.
"""
from __future__ import annotations

import base64
import json
from typing import Any

import pytest
import responses as resp_lib

from aleo import Aleo, HTTPProvider
from aleo.mainnet import PrivateKey, RecordPlaintext, ViewKey
from aleo.facade.records import RecordsModule
from aleo.facade.call import BoundCall
from aleo.facade.errors import ExecutionError
from aleo._facade_common import RecordProvider
from aleo._scanner_common import compute_uuid
from aleo.record_scanner import RecordScanner

# ---------------------------------------------------------------------------
# KAT constants (from test_record_scanner.py / TS SDK tests/data/records.ts)
# ---------------------------------------------------------------------------

BASE = "https://api.provable.com/v2"
SCANNER_BASE = "https://api.provable.com/v2"
HOST = f"{SCANNER_BASE}/mainnet"

VIEW_KEY_STRING = "AViewKey1ccEt8A2Ryva5rxnKcAbn7wgTaTsb79tzkKHFpeKsm9NX"
GOLDEN_PRIVATE_KEY = "APrivateKey1zkp8CZNn3yeCseEtxuVPbDCwSyhGW6yZKUYKfgXmcpoGPWH"
GOLDEN_UUID = "7884164224800444110633570141944665301008802280502652120359195870264061098703field"

RECORD_CIPHERTEXT_STRING = (
    "record1qyqsqpe2szk2wwwq56akkwx586hkndl3r8vzdwve32lm7elvphh37rsyqyxx66trwfhkxun9v35hguerqqpqzq"
    "rtjzeu6vah9x2me2exkgege824sd8x2379scspmrmtvczs0d93qttl7y92ga0k0rsexu409hu3vlehe3yxjhmey3frh2z5"
    "pxm5cmxsv4un97q"
)

# Real plaintext from decrypting RECORD_CIPHERTEXT_STRING with VIEW_KEY_STRING
RECORD_PLAINTEXT_STR = (
    "{\n  owner: aleo1j7qxyunfldj2lp8hsvy7mw5k8zaqgjfyr72x2gh3x4ewgae8v5gscf5jh3.private,\n"
    "  microcredits: 1500000000000000u64.private,\n"
    "  _nonce: 3077450429259593211617823051143573281856129402760267155982965992208217472983group.public,\n"
    "  _version: 0u8.public\n}"
)
RECORD_NONCE = "3077450429259593211617823051143573281856129402760267155982965992208217472983group"
RECORD_MICROCREDITS = 1_500_000_000_000_000

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


def _client(network: str = "mainnet") -> Aleo:
    return Aleo(HTTPProvider(BASE, network=network))


def _golden_view_key() -> Any:
    return PrivateKey.from_string(GOLDEN_PRIVATE_KEY).view_key


def _golden_account() -> Any:
    from aleo.mainnet import Account
    return Account.from_private_key(PrivateKey.from_string(GOLDEN_PRIVATE_KEY))


def _record_view_key_account() -> Any:
    """An account-like object exposing the RECORD_VIEW_KEY_STRING view key."""
    class _Acct:
        view_key = ViewKey.from_string(VIEW_KEY_STRING)
    return _Acct()


def _inject_scanner(a: Aleo, **kwargs: Any) -> RecordScanner:
    scanner = RecordScanner(SCANNER_BASE, **kwargs)
    a.records.scanner = scanner
    return scanner


def _make_nacl_keypair() -> tuple[Any, str]:
    from nacl.public import PrivateKey as NaclPrivateKey
    sk = NaclPrivateKey.generate()
    return sk, base64.b64encode(bytes(sk.public_key)).decode()


# ---------------------------------------------------------------------------
# 1. Module wiring + protocol conformance
# ---------------------------------------------------------------------------


def test_records_module_attached() -> None:
    a = _client()
    assert isinstance(a.records, RecordsModule)


def test_record_provider_defaults_to_records() -> None:
    a = _client()
    assert a.record_provider is a.records


def test_records_module_satisfies_protocol() -> None:
    a = _client()
    assert isinstance(a.records, RecordProvider)


def test_record_provider_settable_and_clearable() -> None:
    a = _client()
    sentinel = object()
    a.record_provider = sentinel
    assert a.record_provider is sentinel
    a.record_provider = None
    assert a.record_provider is None


def test_scanner_built_from_provider_config() -> None:
    a = _client()
    scanner = a.records.scanner
    # The scanner is a Provable service at the API origin under /scanner (NOT
    # the read node's /v2 base); the scanner then appends /<network>. So a
    # provider pointed at https://api.provable.com/v2 yields a scanner at
    # https://api.provable.com/scanner/mainnet.
    assert scanner.url == "https://api.provable.com/scanner/mainnet"


def test_scanner_inherits_provider_creds() -> None:
    # api_key + consumer_id set on the provider (shared with the delegated
    # prover) must reach the scanner so it can mint/refresh its own JWT.
    a = Aleo(HTTPProvider(BASE, api_key="secret-key", consumer_id="consumer-42"))
    scanner = a.records.scanner
    assert scanner.consumer_id == "consumer-42"
    assert scanner._api_key is not None
    assert scanner._api_key["value"] == "secret-key"


# ---------------------------------------------------------------------------
# 2. register — sets account, enables decrypt, hits register endpoint
# ---------------------------------------------------------------------------


@resp_lib.activate
def test_register_flow() -> None:
    sk, pk_b64 = _make_nacl_keypair()
    resp_lib.add(resp_lib.GET, f"{HOST}/pubkey", json={"key_id": "k1", "public_key": pk_b64})
    resp_lib.add(
        resp_lib.POST, f"{HOST}/register/encrypted", json={"uuid": GOLDEN_UUID, "status": None}
    )

    a = _client()
    scanner = _inject_scanner(a)
    acct = _golden_account()

    result = a.records.register(acct, 0)
    assert result["ok"] is True
    assert result["data"]["uuid"] == GOLDEN_UUID

    # Decrypt enabled + UUID set to the account's derived uuid.
    assert scanner.decrypt_enabled is True
    assert str(scanner._uuid) == GOLDEN_UUID

    # Endpoints hit: GET /pubkey then POST /register/encrypted.
    assert "/pubkey" in resp_lib.calls[0].request.url
    assert "/register/encrypted" in resp_lib.calls[1].request.url


# ---------------------------------------------------------------------------
# 3. find — builds and passes the expected OwnedFilter
# ---------------------------------------------------------------------------


@resp_lib.activate
def test_find_passes_owned_filter() -> None:
    resp_lib.add(resp_lib.POST, f"{HOST}/records/owned", json=OWNED_CREDITS_RECORDS)

    a = _client()
    _inject_scanner(a)
    acct = _golden_account()

    records = a.records.find(acct, program="credits.aleo", record="credits", unspent=True)
    assert records == OWNED_CREDITS_RECORDS

    body = json.loads(resp_lib.calls[0].request.body)
    expected = {
        "unspent": True,
        "uuid": str(compute_uuid(acct.view_key)),
        "filter": {"program": "credits.aleo", "record": "credits"},
    }
    assert body == expected


@resp_lib.activate
def test_find_with_nonces() -> None:
    resp_lib.add(resp_lib.POST, f"{HOST}/records/owned", json=[])

    a = _client()
    _inject_scanner(a)
    acct = _golden_account()

    a.records.find(acct, program="credits.aleo", record="credits", nonces=[RECORD_NONCE])
    body = json.loads(resp_lib.calls[0].request.body)
    assert body["nonces"] == [RECORD_NONCE]


# ---------------------------------------------------------------------------
# 4. find_credits — at_least filter
# ---------------------------------------------------------------------------


@resp_lib.activate
def test_find_credits_at_least_covering() -> None:
    resp_lib.add(resp_lib.POST, f"{HOST}/records/owned", json=OWNED_CREDITS_RECORDS)

    a = _client()
    _inject_scanner(a, decrypt_enabled=True)
    acct = _golden_account()

    records = a.records.find_credits(acct, at_least=1_000_000)
    assert len(records) == 1
    assert records[0]["record_plaintext"] == RECORD_PLAINTEXT_STR

    body = json.loads(resp_lib.calls[0].request.body)
    assert body["filter"]["program"] == "credits.aleo"
    assert body["filter"]["record"] == "credits"


@resp_lib.activate
def test_find_credits_at_least_none_when_too_large() -> None:
    resp_lib.add(resp_lib.POST, f"{HOST}/records/owned", json=OWNED_CREDITS_RECORDS)

    a = _client()
    _inject_scanner(a, decrypt_enabled=True)
    acct = _golden_account()

    # No record covers 10^18 microcredits → empty list (RecordNotFound swallowed).
    records = a.records.find_credits(acct, at_least=10 ** 18)
    assert records == []


# ---------------------------------------------------------------------------
# 5. get_unspent — covering / None / exclude_nonces
# ---------------------------------------------------------------------------


@resp_lib.activate
def test_get_unspent_returns_covering_plaintext() -> None:
    resp_lib.add(resp_lib.POST, f"{HOST}/records/owned", json=OWNED_CREDITS_RECORDS)

    a = _client()
    _inject_scanner(a, decrypt_enabled=True)
    # Set the account so the scanner has the uuid + view key.
    a.records._account = _golden_account()
    a.records.scanner.set_account(a.records._account)

    pt = a.records.get_unspent(
        program="credits.aleo", record="credits", min_microcredits=1_000_000
    )
    assert isinstance(pt, RecordPlaintext)
    assert int(pt.microcredits) == RECORD_MICROCREDITS


@resp_lib.activate
def test_get_unspent_none_when_uncovered() -> None:
    resp_lib.add(resp_lib.POST, f"{HOST}/records/owned", json=OWNED_CREDITS_RECORDS)

    a = _client()
    _inject_scanner(a, decrypt_enabled=True)
    a.records._account = _golden_account()
    a.records.scanner.set_account(a.records._account)

    pt = a.records.get_unspent(
        program="credits.aleo", record="credits", min_microcredits=10 ** 18
    )
    assert pt is None


@resp_lib.activate
def test_get_unspent_skips_excluded_nonce() -> None:
    resp_lib.add(resp_lib.POST, f"{HOST}/records/owned", json=OWNED_CREDITS_RECORDS)

    a = _client()
    _inject_scanner(a, decrypt_enabled=True)
    a.records._account = _golden_account()
    a.records.scanner.set_account(a.records._account)

    # The only candidate's nonce is excluded → None.
    pt = a.records.get_unspent(
        program="credits.aleo",
        record="credits",
        min_microcredits=1_000_000,
        exclude_nonces=(RECORD_NONCE,),
    )
    assert pt is None


# ---------------------------------------------------------------------------
# 6. decrypt KAT (via scanner.decrypt, reusing the record fixture)
# ---------------------------------------------------------------------------


def test_decrypt_kat() -> None:
    a = _client()
    scanner = _inject_scanner(a)
    vk = ViewKey.from_string(VIEW_KEY_STRING)
    records: list[dict[str, Any]] = [{"record_ciphertext": RECORD_CIPHERTEXT_STRING}]
    scanner.decrypt(vk, records)
    assert "record_plaintext" in records[0]
    pt = RecordPlaintext.from_string(records[0]["record_plaintext"])
    assert int(pt.microcredits) == RECORD_MICROCREDITS


# ---------------------------------------------------------------------------
# 7. F5 integration — closing the private-fee auto-sourcing seam
# ---------------------------------------------------------------------------


class _FakeProvider:
    """A minimal RecordProvider stub returning a fixed RecordPlaintext."""

    def __init__(self, record: Any) -> None:
        self._record = record
        self.calls: list[dict[str, Any]] = []

    def get_unspent(
        self,
        *,
        program: str,
        record: str,
        min_microcredits: int | None = None,
        exclude_nonces: tuple[str, ...] = (),
    ) -> Any:
        self.calls.append(
            {
                "program": program,
                "record": record,
                "min_microcredits": min_microcredits,
                "exclude_nonces": exclude_nonces,
            }
        )
        return self._record

    def find(self, **filters: Any) -> list[Any]:
        return []


def _bound(a: Aleo) -> tuple[BoundCall, Any]:
    from aleo.mainnet import Program as RawProgram
    from aleo.facade.programs import Program

    raw = RawProgram.from_source(RawProgram.credits().source)
    prog = Program(a, raw)
    acct = a.account.from_private_key(PrivateKey.random())
    bc = prog.functions.transfer_public(str(acct.address), 10)
    assert isinstance(bc, BoundCall)
    return bc, acct


def test_fake_provider_satisfies_protocol() -> None:
    fake = _FakeProvider(RecordPlaintext.from_string(RECORD_PLAINTEXT_STR))
    assert isinstance(fake, RecordProvider)


def test_resolve_fee_record_consumes_provider() -> None:
    """_resolve_fee_record auto-sources from aleo.record_provider (seam closed)."""
    a = _client()
    bc, _ = _bound(a)

    known = RecordPlaintext.from_string(RECORD_PLAINTEXT_STR)
    fake = _FakeProvider(known)
    a.record_provider = fake

    got = bc._resolve_fee_record(None, min_microcredits=5000)
    assert got is known
    assert fake.calls == [
        {
            "program": "credits.aleo",
            "record": "credits",
            "min_microcredits": 5000,
            "exclude_nonces": (),
        }
    ]


def test_authorize_fee_private_uses_provider_record() -> None:
    """The private-fee path threads the provider record into authorize_fee_private."""
    a = _client()
    bc, acct = _bound(a)

    known = RecordPlaintext.from_string(RECORD_PLAINTEXT_STR)
    a.record_provider = _FakeProvider(known)

    captured: dict[str, Any] = {}

    class _Execution:
        execution_id = "exec-id"

    class _ProcessShim:
        def execution_cost(self, _execution: Any) -> Any:
            return (1000, (900, 100))

        def authorize_fee_private(
            self, _pk: Any, record: Any, base_fee: int, priority: int, exec_id: Any
        ) -> Any:
            captured["record"] = record
            captured["base_fee"] = base_fee
            captured["priority"] = priority
            return "FEE_AUTH"

    a._process = _ProcessShim()  # type: ignore[attr-defined]

    fee_auth = bc._authorize_fee(
        acct, _Execution(), priority_fee=100, fee_record=None, private_fee=True
    )
    assert fee_auth == "FEE_AUTH"
    # The provider's record was consumed by authorize_fee_private.
    assert captured["record"] is known
    assert captured["base_fee"] == 1000
    assert captured["priority"] == 100


def test_private_fee_provider_returns_none_errors() -> None:
    """Provider returns None → clear ExecutionError."""
    a = _client()
    bc, _ = _bound(a)
    a.record_provider = _FakeProvider(None)

    with pytest.raises(ExecutionError, match="No unspent credits record"):
        bc._resolve_fee_record(None, min_microcredits=5000)


def test_private_fee_no_provider_errors() -> None:
    """No provider configured → clear ExecutionError."""
    a = _client()
    bc, _ = _bound(a)
    a.record_provider = None

    with pytest.raises(ExecutionError, match="record provider"):
        bc._resolve_fee_record(None, min_microcredits=5000)
