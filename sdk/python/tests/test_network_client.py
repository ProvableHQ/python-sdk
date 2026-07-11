"""Tests for AleoNetworkClient (sync, requests-based). All mocked — no live network."""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch, call
import pytest
import responses as resp_lib
from responses import RequestsMock

from aleo.network_client import AleoNetworkClient
from aleo._client_common import AleoNetworkError, jwt_origin

BASE = "https://api.provable.com/v2"
NET = "mainnet"
HOST = f"{BASE}/{NET}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_client(**kwargs: Any) -> AleoNetworkClient:
    return AleoNetworkClient(BASE, network=NET, **kwargs)


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_get_block_url() -> None:
    resp_lib.add(resp_lib.GET, f"{HOST}/block/42", json={"height": 42})
    c = make_client()
    result = c.get_block(42)
    assert result["height"] == 42
    assert resp_lib.calls[0].request.url == f"{HOST}/block/42"


@resp_lib.activate
def test_get_latest_block_url() -> None:
    resp_lib.add(resp_lib.GET, f"{HOST}/block/latest", json={"height": 100})
    c = make_client()
    result = c.get_latest_block()
    assert result["height"] == 100


@resp_lib.activate
def test_get_latest_height_url() -> None:
    resp_lib.add(resp_lib.GET, f"{HOST}/block/height/latest", json=9999)
    c = make_client()
    assert c.get_latest_height() == 9999


@resp_lib.activate
def test_get_latest_block_hash_url() -> None:
    resp_lib.add(resp_lib.GET, f"{HOST}/block/hash/latest", json="ab1234hash")
    c = make_client()
    assert c.get_latest_block_hash() == "ab1234hash"


@resp_lib.activate
def test_get_committee_by_height_url() -> None:
    resp_lib.add(resp_lib.GET, f"{HOST}/committee/100", json={"members": []})
    c = make_client()
    result = c.get_committee_by_height(100)
    assert result == {"members": []}


@resp_lib.activate
def test_get_state_root_url() -> None:
    resp_lib.add(resp_lib.GET, f"{HOST}/stateRoot/latest", json="sr1abc")
    c = make_client()
    assert c.get_state_root() == "sr1abc"


@resp_lib.activate
def test_get_program_url() -> None:
    resp_lib.add(resp_lib.GET, f"{HOST}/program/hello.aleo", json="program hello.aleo;")
    c = make_client()
    result = c.get_program("hello.aleo")
    assert "hello.aleo" in result


@resp_lib.activate
def test_get_program_with_edition_url() -> None:
    resp_lib.add(resp_lib.GET, f"{HOST}/program/hello.aleo/2", json="program hello.aleo;")
    c = make_client()
    result = c.get_program("hello.aleo", edition=2)
    assert "hello.aleo" in result
    assert "/program/hello.aleo/2" in resp_lib.calls[0].request.url


@resp_lib.activate
def test_get_transaction_url() -> None:
    resp_lib.add(resp_lib.GET, f"{HOST}/transaction/at1abc", json={"id": "at1abc"})
    c = make_client()
    result = c.get_transaction("at1abc")
    assert result["id"] == "at1abc"


@resp_lib.activate
def test_get_confirmed_transaction_url() -> None:
    resp_lib.add(resp_lib.GET, f"{HOST}/transaction/confirmed/at1abc", json={"status": "accepted"})
    c = make_client()
    result = c.get_confirmed_transaction("at1abc")
    assert result["status"] == "accepted"


@resp_lib.activate
def test_get_mapping_value_url() -> None:
    resp_lib.add(resp_lib.GET, f"{HOST}/program/credits.aleo/mapping/account/aleo1abc", json="500u64")
    c = make_client()
    val = c.get_program_mapping_value("credits.aleo", "account", "aleo1abc")
    assert val == "500u64"


@resp_lib.activate
def test_get_deployment_tx_id_strips_quotes() -> None:
    resp_lib.add(
        resp_lib.GET,
        f"{HOST}/find/transactionID/deployment/hello.aleo",
        json='"at1realid"',
    )
    c = make_client()
    result = c.get_deployment_transaction_id_for_program("hello.aleo")
    assert '"' not in result
    assert "at1realid" in result


# ---------------------------------------------------------------------------
# Block range validation
# ---------------------------------------------------------------------------

def test_block_range_negative_start_raises() -> None:
    c = make_client()
    with pytest.raises(ValueError, match="start"):
        c.get_block_range(-1, 10)


def test_block_range_start_gt_end_raises() -> None:
    c = make_client()
    with pytest.raises(ValueError, match="start"):
        c.get_block_range(50, 10)


def test_block_range_span_exceeds_50_raises() -> None:
    c = make_client()
    with pytest.raises(ValueError, match="50"):
        c.get_block_range(0, 51)


@resp_lib.activate
def test_block_range_exactly_50_ok() -> None:
    resp_lib.add(resp_lib.GET, f"{HOST}/blocks", json=[])
    c = make_client()
    c.get_block_range(0, 50)  # span=50, should not raise
    assert len(resp_lib.calls) == 1


# ---------------------------------------------------------------------------
# get_program_imports dedup (DFS, each program fetched exactly once)
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_get_program_imports_dedup() -> None:
    """2-level import fixture: top->mid->leaf. Each fetched exactly once."""
    leaf_src = "program leaf.aleo;\nfunction noop:\n    input r0 as u32.public;\n    output r0 as u32.public;\n"
    mid_src = f"import leaf.aleo;\nprogram mid.aleo;\nfunction noop:\n    input r0 as u32.public;\n    output r0 as u32.public;\n"
    top_src = f"import mid.aleo;\nprogram top.aleo;\nfunction noop:\n    input r0 as u32.public;\n    output r0 as u32.public;\n"

    resp_lib.add(resp_lib.GET, f"{HOST}/program/top.aleo", json=top_src)
    resp_lib.add(resp_lib.GET, f"{HOST}/program/mid.aleo", json=mid_src)
    resp_lib.add(resp_lib.GET, f"{HOST}/program/leaf.aleo", json=leaf_src)

    c = make_client()
    imports = c.get_program_imports("top.aleo")

    # Should contain both imports
    assert "mid.aleo" in imports
    assert "leaf.aleo" in imports

    # Each program fetched exactly once (dedup check via call count)
    urls = [call.request.url.split("?")[0] for call in resp_lib.calls]
    # top.aleo fetched once (during get_program_imports), mid.aleo once, leaf.aleo once
    mid_count = sum(1 for u in urls if "/program/mid.aleo" in u)
    leaf_count = sum(1 for u in urls if "/program/leaf.aleo" in u)
    assert mid_count == 1, f"mid.aleo fetched {mid_count} times, expected 1"
    assert leaf_count == 1, f"leaf.aleo fetched {leaf_count} times, expected 1"


# ---------------------------------------------------------------------------
# Retry: 503 retried, 400 not retried
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_retry_on_503() -> None:
    """503s should be retried; succeed on 3rd attempt."""
    resp_lib.add(resp_lib.GET, f"{HOST}/block/latest", status=503)
    resp_lib.add(resp_lib.GET, f"{HOST}/block/latest", status=503)
    resp_lib.add(resp_lib.GET, f"{HOST}/block/latest", json={"height": 1})

    c = make_client()
    # Patch sleep to speed up test
    with patch("aleo._client_common.time.sleep"):
        result = c.get_latest_block()
    assert result["height"] == 1
    assert len(resp_lib.calls) == 3


@resp_lib.activate
def test_no_retry_on_400() -> None:
    """400 errors should NOT be retried."""
    resp_lib.add(resp_lib.GET, f"{HOST}/block/latest", status=400, body="bad request")

    c = make_client()
    with pytest.raises(AleoNetworkError) as exc_info:
        c.get_latest_block()
    assert exc_info.value.status == 400
    assert len(resp_lib.calls) == 1


# ---------------------------------------------------------------------------
# Headers: default, per-method, custom transport suppression
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_default_sdk_headers_present() -> None:
    resp_lib.add(resp_lib.GET, f"{HOST}/block/latest", json={})
    c = make_client()
    c.get_latest_block()
    req_headers = resp_lib.calls[0].request.headers
    assert "X-Aleo-SDK-Version" in req_headers
    assert "X-Aleo-environment" in req_headers
    assert req_headers["X-Aleo-environment"] == "python"


@resp_lib.activate
def test_per_method_header() -> None:
    resp_lib.add(resp_lib.GET, f"{HOST}/block/42", json={})
    c = make_client()
    c.get_block(42)
    req_headers = resp_lib.calls[0].request.headers
    assert req_headers.get("X-ALEO-METHOD") == "getBlock"


def test_custom_transport_callable_used_for_requests() -> None:
    """A callable transport is invoked for every HTTP request."""
    import requests as _requests
    calls: list[tuple[str, str]] = []

    def my_transport(method: str, url: str, **kwargs: Any) -> _requests.Response:
        calls.append((method, url))
        r = _requests.Response()
        r.status_code = 200
        r._content = b'{"height": 77}'
        return r

    c = AleoNetworkClient(BASE, network=NET, transport=my_transport)
    result = c.get_block(77)
    assert result["height"] == 77
    assert len(calls) == 1
    assert calls[0][0] == "GET"
    assert "/block/77" in calls[0][1]


@resp_lib.activate
def test_custom_transport_suppresses_sdk_headers() -> None:
    resp_lib.add(resp_lib.GET, f"{HOST}/block/latest", json={})
    # Passing transport=True simulates custom transport (has_custom_transport=True)
    c = AleoNetworkClient(BASE, network=NET, transport=object())
    c.get_latest_block()
    req_headers = resp_lib.calls[0].request.headers
    assert "X-Aleo-SDK-Version" not in req_headers
    assert "X-ALEO-METHOD" not in req_headers


@resp_lib.activate
def test_set_header() -> None:
    resp_lib.add(resp_lib.GET, f"{HOST}/block/latest", json={})
    c = make_client()
    c.set_header("X-Custom-Foo", "bar")
    c.get_latest_block()
    assert resp_lib.calls[0].request.headers.get("X-Custom-Foo") == "bar"


@resp_lib.activate
def test_remove_header() -> None:
    resp_lib.add(resp_lib.GET, f"{HOST}/block/latest", json={})
    c = make_client()
    c.set_header("X-Custom-Foo", "bar")
    c.remove_header("X-Custom-Foo")
    c.get_latest_block()
    assert "X-Custom-Foo" not in resp_lib.calls[0].request.headers


# ---------------------------------------------------------------------------
# JWT origin derivation
# ---------------------------------------------------------------------------

def test_jwt_origin_standard() -> None:
    assert jwt_origin("https://api.provable.com/v2") == "https://api.provable.com"


def test_jwt_origin_with_port() -> None:
    assert jwt_origin("http://localhost:3030/v2") == "http://localhost:3030"


def test_jwt_origin_trailing_slash() -> None:
    assert jwt_origin("https://api.provable.com/v2/") == "https://api.provable.com"


def test_jwt_origin_localhost() -> None:
    assert jwt_origin("http://localhost/v1") == "http://localhost"


@resp_lib.activate
def test_jwt_fetched_from_correct_origin() -> None:
    """JWT refresh hits {origin}/jwts/{consumer_id}."""
    jwt_url = "https://api.provable.com/jwts/cid123"
    resp_lib.add(
        resp_lib.POST, jwt_url,
        json={"exp": 9999999999},
        headers={"authorization": "Bearer tok123"},
    )
    resp_lib.add(resp_lib.GET, f"{HOST}/block/latest", json={"height": 1})

    c = make_client(api_key="key123", consumer_id="cid123")
    # Force JWT refresh by leaving jwt_data None
    with patch("aleo._client_common.time.sleep"):
        jwt = c._refresh_jwt("key123", "cid123")
    assert jwt["jwt"] == "Bearer tok123"
    assert jwt["expiration"] == 9999999999 * 1000


@resp_lib.activate
def test_jwt_from_response_header() -> None:
    jwt_url = "https://api.provable.com/jwts/cid999"
    resp_lib.add(
        resp_lib.POST, jwt_url,
        json={"exp": 9999999999},
        headers={"authorization": "Bearer mytoken"},
    )
    c = make_client(api_key="mykey", consumer_id="cid999")
    jwt = c._refresh_jwt("mykey", "cid999")
    assert jwt["jwt"] == "Bearer mytoken"


@resp_lib.activate
def test_jwt_expiry_within_5min_triggers_refresh() -> None:
    """If JWT expires within 5 minutes, refresh is triggered."""
    import time
    # Set expiration to 2 min from now (within 5 min window)
    exp_ms = int(time.time() * 1000) + 2 * 60 * 1000
    jwt_url = "https://api.provable.com/jwts/cid1"
    resp_lib.add(
        resp_lib.POST, jwt_url,
        json={"exp": 9999999999},
        headers={"authorization": "Bearer refreshed"},
    )
    resp_lib.add(resp_lib.GET, f"{HOST}/block/latest", json={"height": 1})

    c = make_client(api_key="k", consumer_id="cid1")
    c.jwt_data = {"jwt": "Bearer old", "expiration": exp_ms}
    # _ensure_jwt should detect near-expiry and refresh
    new_jwt = c._ensure_jwt("k", "cid1", c.jwt_data)
    assert new_jwt is not None
    assert new_jwt["jwt"] == "Bearer refreshed"


# ---------------------------------------------------------------------------
# submit_transaction: verbose_errors query param
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_submit_transaction_verbose_errors_true() -> None:
    resp_lib.add(resp_lib.POST, f"{HOST}/transaction/broadcast", json="at1txid")
    c = make_client()
    c.set_verbose_errors(True)
    result = c.submit_transaction('{"fake": "tx"}')
    url = resp_lib.calls[0].request.url
    assert "check_transaction=true" in url
    assert result == "at1txid"


@resp_lib.activate
def test_submit_transaction_verbose_errors_false() -> None:
    resp_lib.add(resp_lib.POST, f"{HOST}/transaction/broadcast", json="at1txid")
    c = make_client()
    c.set_verbose_errors(False)
    c.submit_transaction('{"fake": "tx"}')
    url = resp_lib.calls[0].request.url
    assert "check_transaction" not in url


# ---------------------------------------------------------------------------
# wait_for_transaction_confirmation
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_wait_for_confirmation_accepted() -> None:
    resp_lib.add(
        resp_lib.GET, f"{HOST}/transaction/confirmed/at1abc",
        json={"status": "accepted", "id": "at1abc"},
    )
    c = make_client()
    result = c.wait_for_transaction_confirmation("at1abc", check_interval=0.01, timeout=5.0)
    assert result["status"] == "accepted"


@resp_lib.activate
def test_wait_for_confirmation_rejected() -> None:
    resp_lib.add(
        resp_lib.GET, f"{HOST}/transaction/confirmed/at1rej",
        json={"status": "rejected"},
    )
    c = make_client()
    with pytest.raises(AleoNetworkError, match="rejected"):
        c.wait_for_transaction_confirmation("at1rej", check_interval=0.01, timeout=5.0)


@resp_lib.activate
def test_wait_for_confirmation_timeout() -> None:
    resp_lib.add(resp_lib.GET, f"{HOST}/transaction/confirmed/at1slow", status=404)
    c = make_client()
    with pytest.raises(TimeoutError):
        c.wait_for_transaction_confirmation("at1slow", check_interval=0.01, timeout=0.05)


@resp_lib.activate
def test_wait_for_confirmation_invalid_url_fatal() -> None:
    resp_lib.add(
        resp_lib.GET, f"{HOST}/transaction/confirmed/bad",
        status=400, body="Invalid URL: bad transaction id",
    )
    c = make_client()
    with pytest.raises(AleoNetworkError, match="Malformed"):
        c.wait_for_transaction_confirmation("bad", check_interval=0.01, timeout=5.0)


# ---------------------------------------------------------------------------
# DPS: submit_proving_request_safe / submit_proving_request
# ---------------------------------------------------------------------------

def _load_proving_request() -> Any:
    """Load the KAT proving request from vectors."""
    import json
    from pathlib import Path
    v = json.loads((Path(__file__).parent / "vectors" / "proving_request.json").read_text())
    pr_str = v["PUZZLE_SPINNER_V002_PROVING_REQUEST"]
    from aleo.mainnet import ProvingRequest  # type: ignore[attr-defined]
    return ProvingRequest.from_string(pr_str)


def _build_request_variant_pr() -> Any:
    """Build a Request-variant ProvingRequest via ExecutionRequest.sign."""
    from aleo.mainnet import PrivateKey, ExecutionRequest, ProvingRequest  # type: ignore[attr-defined]
    private_key = PrivateKey.from_string(
        "APrivateKey1zkp8CZNn3yeCseEtxuVPbDCwSyhGW6yZKUYKfgXmcpoGPWH"
    )
    exec_req = ExecutionRequest.sign(
        private_key,
        "credits.aleo",
        "transfer_public",
        ["aleo1rhgdu77hgyqd3xjj8ucu3jj9r2krwz6mnzyd80gncr5fxcwlh5rsvzp9px", "100u64"],
        ["address.public", "u64.public"],
        None,
        None,
        True,
        False,
    )
    return ProvingRequest.from_request(exec_req, None, False)


@pytest.fixture
def nacl_keypair() -> Any:
    """Generate a real X25519 keypair for DPS mock tests."""
    from nacl.public import PrivateKey
    import base64
    sk = PrivateKey.generate()
    pk_b64 = base64.b64encode(bytes(sk.public_key)).decode()
    return sk, pk_b64


@resp_lib.activate
def test_dps_authorization_variant_routes_correctly(nacl_keypair: Any) -> None:
    """Authorization-variant PR hits /prove/authorization; ciphertext decrypts correctly."""
    from nacl.public import PrivateKey, SealedBox
    import base64

    sk, pk_b64 = nacl_keypair
    prover = "https://prover.provable.prove"

    resp_lib.add(
        resp_lib.GET, f"{prover}/mainnet/pubkey",
        json={"key_id": "k1", "public_key": pk_b64},
        headers={"set-cookie": "session=abc"},
    )
    resp_lib.add(
        resp_lib.POST, f"{prover}/mainnet/prove/authorization",
        json={"transaction": "at1fake", "broadcast_result": {"status": "accepted"}},
    )

    c = AleoNetworkClient(BASE, network=NET, prover_uri=prover)
    # Inject a non-expiring JWT so no refresh is triggered
    c.jwt_data = {"jwt": "Bearer testjwt", "expiration": 99999999999999}

    pr = _load_proving_request()

    result = c.submit_proving_request_safe(pr)
    assert result["ok"] is True
    assert result["data"]["transaction"] == "at1fake"

    # Verify the POST hit /prove/authorization
    post_calls = [c for c in resp_lib.calls if c.request.method == "POST"]
    assert any("prove/authorization" in c.request.url for c in post_calls)

    # Verify ciphertext decrypts to PR bytes
    post_body = json.loads(post_calls[-1].request.body)
    ciphertext = base64.b64decode(post_body["ciphertext"])
    box = SealedBox(sk)
    decrypted = box.decrypt(ciphertext)
    assert decrypted == bytes(pr.bytes())


@resp_lib.activate
def test_dps_request_variant_routes_to_prove_request(nacl_keypair: Any) -> None:
    """Request-variant PR (passed as string) hits /prove/request; ciphertext decrypts correctly."""
    from nacl.public import SealedBox
    import base64

    sk, pk_b64 = nacl_keypair
    prover = "https://prover.provable.prove"

    resp_lib.add(
        resp_lib.GET, f"{prover}/mainnet/pubkey",
        json={"key_id": "k1", "public_key": pk_b64},
    )
    resp_lib.add(
        resp_lib.POST, f"{prover}/mainnet/prove/request",
        json={"transaction": "at1req", "broadcast_result": {"status": "accepted"}},
    )

    c = AleoNetworkClient(BASE, network=NET, prover_uri=prover)
    c.jwt_data = {"jwt": "Bearer testjwt", "expiration": 99999999999999}

    pr = _build_request_variant_pr()
    pr_str = str(pr)  # pass STRING to test deserialization routing

    result = c.submit_proving_request_safe(pr_str)
    assert result["ok"] is True

    # Verify the POST hit /prove/request (not /prove/authorization)
    post_calls = [c for c in resp_lib.calls if c.request.method == "POST"]
    assert len(post_calls) == 1
    assert "prove/request" in post_calls[0].request.url

    # Verify ciphertext decrypts to same bytes as the original PR
    post_body = json.loads(post_calls[0].request.body)
    ciphertext = base64.b64decode(post_body["ciphertext"])
    box = SealedBox(sk)
    decrypted = box.decrypt(ciphertext)
    assert decrypted == bytes(pr.bytes())


@resp_lib.activate
def test_dps_400_returned_verbatim_not_retried(nacl_keypair: Any) -> None:
    """400 from prover returned as ok=False; not retried."""
    _, pk_b64 = nacl_keypair
    prover = "https://prover.provable.prove"

    resp_lib.add(
        resp_lib.GET, f"{prover}/mainnet/pubkey",
        json={"key_id": "k1", "public_key": pk_b64},
    )
    resp_lib.add(
        resp_lib.POST, f"{prover}/mainnet/prove/authorization",
        status=400, json={"message": "bad input"},
    )

    c = AleoNetworkClient(BASE, network=NET, prover_uri=prover)
    c.jwt_data = {"jwt": "Bearer testjwt", "expiration": 99999999999999}

    pr = _load_proving_request()

    result = c.submit_proving_request_safe(pr)
    assert result["ok"] is False
    assert result["status"] == 400
    # Only one POST call (not retried)
    post_calls = [c for c in resp_lib.calls if c.request.method == "POST"]
    assert len(post_calls) == 1


@resp_lib.activate
def test_dps_503_retried(nacl_keypair: Any) -> None:
    """503 from prover is retried."""
    _, pk_b64 = nacl_keypair
    prover = "https://prover.provable.prove"

    # pubkey fetched each retry
    for _ in range(3):
        resp_lib.add(
            resp_lib.GET, f"{prover}/mainnet/pubkey",
            json={"key_id": "k1", "public_key": pk_b64},
        )
    resp_lib.add(resp_lib.POST, f"{prover}/mainnet/prove/authorization", status=503)
    resp_lib.add(resp_lib.POST, f"{prover}/mainnet/prove/authorization", status=503)
    resp_lib.add(
        resp_lib.POST, f"{prover}/mainnet/prove/authorization",
        json={"transaction": "at1ok", "broadcast_result": {"status": "accepted"}},
    )

    c = AleoNetworkClient(BASE, network=NET, prover_uri=prover)
    c.jwt_data = {"jwt": "Bearer testjwt", "expiration": 99999999999999}

    pr = _load_proving_request()

    with patch("aleo._client_common.time.sleep"):
        result = c.submit_proving_request_safe(pr)

    assert result["ok"] is True
    post_calls = [c for c in resp_lib.calls if c.request.method == "POST"]
    assert len(post_calls) == 3


@resp_lib.activate
def test_dps_cookie_echoed(nacl_keypair: Any) -> None:
    """set-cookie from /pubkey is echoed back as Cookie on the prove POST."""
    _, pk_b64 = nacl_keypair
    prover = "https://prover.provable.prove"

    resp_lib.add(
        resp_lib.GET, f"{prover}/mainnet/pubkey",
        json={"key_id": "k1", "public_key": pk_b64},
        headers={"set-cookie": "session=mysession"},
    )
    resp_lib.add(
        resp_lib.POST, f"{prover}/mainnet/prove/authorization",
        json={"transaction": "at1ok", "broadcast_result": {"status": "accepted"}},
    )

    c = AleoNetworkClient(BASE, network=NET, prover_uri=prover)
    c.jwt_data = {"jwt": "Bearer testjwt", "expiration": 99999999999999}

    pr = _load_proving_request()

    c.submit_proving_request_safe(pr)
    post_calls = [c for c in resp_lib.calls if c.request.method == "POST"]
    assert post_calls[-1].request.headers.get("Cookie") == "session=mysession"


@resp_lib.activate
def test_dps_defaults_to_prove_service_base(nacl_keypair: Any) -> None:
    """With no prover_uri, the DPS handshake targets the API origin under the
    ``/prove/{network}`` prefix (a Provable service sibling to ``/scanner``),
    NOT the read node's ``/v2/{network}`` base."""
    _, pk_b64 = nacl_keypair
    # BASE is https://api.provable.com/v2 → origin https://api.provable.com,
    # so the prover base is https://api.provable.com/prove/mainnet.
    resp_lib.add(
        resp_lib.GET, "https://api.provable.com/prove/mainnet/pubkey",
        json={"key_id": "k1", "public_key": pk_b64},
        headers={"set-cookie": "session=x"},
    )
    resp_lib.add(
        resp_lib.POST, "https://api.provable.com/prove/mainnet/prove/authorization",
        json={"transaction": "at1ok", "broadcast_result": {"status": "accepted"}},
    )

    c = AleoNetworkClient(BASE, network=NET)  # no prover_uri
    c.jwt_data = {"jwt": "Bearer testjwt", "expiration": 99999999999999}

    result = c.submit_proving_request_safe(_load_proving_request())
    assert result["ok"] is True
    # pubkey hit /prove/{network}/pubkey, never the /v2 read base.
    get_urls = [call.request.url for call in resp_lib.calls if call.request.method == "GET"]
    assert any(u == "https://api.provable.com/prove/mainnet/pubkey" for u in get_urls)
    assert not any("/v2" in u for u in get_urls)


@resp_lib.activate
def test_dps_authorization_header_sent(nacl_keypair: Any) -> None:
    """JWT is sent as Authorization header on prove POST."""
    _, pk_b64 = nacl_keypair
    prover = "https://prover.provable.prove"

    resp_lib.add(
        resp_lib.GET, f"{prover}/mainnet/pubkey",
        json={"key_id": "k1", "public_key": pk_b64},
    )
    resp_lib.add(
        resp_lib.POST, f"{prover}/mainnet/prove/authorization",
        json={"transaction": "at1ok", "broadcast_result": {"status": "accepted"}},
    )

    c = AleoNetworkClient(BASE, network=NET, prover_uri=prover)
    c.jwt_data = {"jwt": "Bearer myjwt", "expiration": 99999999999999}

    pr = _load_proving_request()

    c.submit_proving_request_safe(pr)
    post_calls = [c for c in resp_lib.calls if c.request.method == "POST"]
    assert post_calls[-1].request.headers.get("Authorization") == "Bearer myjwt"


@resp_lib.activate
def test_dps_submit_proving_request_raises_on_failure(nacl_keypair: Any) -> None:
    """submit_proving_request raises AleoProvingError on failure."""
    from aleo._client_common import AleoProvingError
    _, pk_b64 = nacl_keypair
    prover = "https://prover.provable.prove"

    resp_lib.add(
        resp_lib.GET, f"{prover}/mainnet/pubkey",
        json={"key_id": "k1", "public_key": pk_b64},
    )
    resp_lib.add(
        resp_lib.POST, f"{prover}/mainnet/prove/authorization",
        status=400, json={"message": "invalid request"},
    )

    c = AleoNetworkClient(BASE, network=NET, prover_uri=prover)
    c.jwt_data = {"jwt": "Bearer jwt", "expiration": 99999999999999}

    pr = _load_proving_request()

    with pytest.raises(AleoProvingError) as exc_info:
        c.submit_proving_request(pr)
    assert exc_info.value.status == 400


# ---------------------------------------------------------------------------
# Mutators
# ---------------------------------------------------------------------------

def test_set_account() -> None:
    c = make_client()
    mock_acct = MagicMock()
    c.set_account(mock_acct)
    assert c.get_account() is mock_acct


def test_network_in_host() -> None:
    c = AleoNetworkClient("https://api.provable.com/v2", network="testnet")
    assert c._host == "https://api.provable.com/v2/testnet"


def test_set_host_updates_host() -> None:
    c = make_client()
    c.set_host("https://other.example.com/v3")
    assert c._host == f"https://other.example.com/v3/{NET}"


def test_set_prover_uri() -> None:
    c = make_client()
    c.set_prover_uri("https://prover.example.com")
    assert c._prover_uri == f"https://prover.example.com/{NET}"
