"""Tests for AsyncAleoNetworkClient (async, httpx-based). All mocked — no live network."""
from __future__ import annotations

import json
import base64
from typing import Any, Callable
from unittest.mock import patch
import pytest
import httpx

from aleo.async_network_client import AsyncAleoNetworkClient
from aleo._client_common import AleoNetworkError, AleoProvingError, make_default_headers

BASE = "https://api.provable.com/v2"
NET = "mainnet"
HOST = f"{BASE}/{NET}"

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Mock transport helpers
# ---------------------------------------------------------------------------

def route_handler(
    routes: dict[str, Any],
) -> Callable[[httpx.Request], httpx.Response]:
    """Return a handler function for httpx.MockTransport.

    Each key in routes is a URL substring to match; value is either an
    httpx.Response or a callable(request) -> Response.
    """
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
    """Shorthand: JSON response."""
    return httpx.Response(status, json=data, headers=headers or {})


def make_client(
    routes: dict[str, Any] | None = None,
    **overrides: Any,
) -> AsyncAleoNetworkClient:
    """Build AsyncAleoNetworkClient bypassing __init__ to inject a mock transport."""
    transport = httpx.MockTransport(route_handler(routes or {}))
    c: AsyncAleoNetworkClient = object.__new__(AsyncAleoNetworkClient)
    c._base_url = BASE
    c._network = NET
    c._host = HOST
    c._has_custom_transport = False
    c._transport = None
    c._account = None
    c._verbose_errors = True
    c.api_key = None
    c.consumer_id = None
    c.jwt_data = None
    c._prover_uri = None
    c._record_scanner_uri = None
    c.headers = make_default_headers()
    c._client = httpx.AsyncClient(transport=transport)
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


def make_client_with_handler(
    handler: Callable[[httpx.Request], httpx.Response],
    **overrides: Any,
) -> AsyncAleoNetworkClient:
    transport = httpx.MockTransport(handler)
    c: AsyncAleoNetworkClient = object.__new__(AsyncAleoNetworkClient)
    c._base_url = BASE
    c._network = NET
    c._host = HOST
    c._has_custom_transport = False
    c._transport = None
    c._account = None
    c._verbose_errors = True
    c.api_key = None
    c.consumer_id = None
    c.jwt_data = None
    c._prover_uri = None
    c._record_scanner_uri = None
    c.headers = make_default_headers()
    c._client = httpx.AsyncClient(transport=transport)
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_get_block() -> None:
    c = make_client({f"{HOST}/block/42": jr({"height": 42})})
    result = await c.get_block(42)
    assert result["height"] == 42


@pytest.mark.asyncio
async def test_async_get_latest_height() -> None:
    c = make_client({f"{HOST}/block/height/latest": jr(5000)})
    assert await c.get_latest_height() == 5000


@pytest.mark.asyncio
async def test_async_get_latest_block_hash() -> None:
    c = make_client({f"{HOST}/block/hash/latest": jr("hash123")})
    assert await c.get_latest_block_hash() == "hash123"


@pytest.mark.asyncio
async def test_async_get_state_root() -> None:
    c = make_client({f"{HOST}/stateRoot/latest": jr("sr1abc")})
    assert await c.get_state_root() == "sr1abc"


@pytest.mark.asyncio
async def test_async_get_program() -> None:
    c = make_client({f"{HOST}/program/hello.aleo": jr("program hello.aleo;")})
    result = await c.get_program("hello.aleo")
    assert "hello.aleo" in result


@pytest.mark.asyncio
async def test_async_get_transaction() -> None:
    c = make_client({f"{HOST}/transaction/at1abc": jr({"id": "at1abc"})})
    result = await c.get_transaction("at1abc")
    assert result["id"] == "at1abc"


@pytest.mark.asyncio
async def test_async_get_confirmed_transaction() -> None:
    c = make_client({f"{HOST}/transaction/confirmed/at1abc": jr({"status": "accepted"})})
    result = await c.get_confirmed_transaction("at1abc")
    assert result["status"] == "accepted"


@pytest.mark.asyncio
async def test_async_get_deployment_tx_id_strips_quotes() -> None:
    c = make_client({
        f"{HOST}/find/transactionID/deployment/hello.aleo": jr('"at1realid"'),
    })
    result = await c.get_deployment_transaction_id_for_program("hello.aleo")
    assert '"' not in result
    assert "at1realid" in result


# ---------------------------------------------------------------------------
# Block range validation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_block_range_negative_start_raises() -> None:
    c = make_client()
    with pytest.raises(ValueError, match="start"):
        await c.get_block_range(-1, 10)


@pytest.mark.asyncio
async def test_async_block_range_span_exceeds_50_raises() -> None:
    c = make_client()
    with pytest.raises(ValueError, match="50"):
        await c.get_block_range(0, 51)


# ---------------------------------------------------------------------------
# Retry on 5xx
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_retry_on_503() -> None:
    call_count = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return httpx.Response(503, text="service unavailable")
        return jr({"height": 7})

    c = make_client_with_handler(handler)
    with patch("asyncio.sleep"):
        result = await c.get_latest_block()
    assert result["height"] == 7
    assert call_count == 3


@pytest.mark.asyncio
async def test_async_no_retry_on_400() -> None:
    call_count = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(400, text="bad request")

    c = make_client_with_handler(handler)
    with pytest.raises(AleoNetworkError) as exc_info:
        await c.get_latest_block()
    assert exc_info.value.status == 400
    assert call_count == 1


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_default_sdk_headers_present() -> None:
    captured: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return jr({})

    c = make_client_with_handler(handler)
    await c.get_latest_block()
    assert "X-Aleo-SDK-Version" in captured[0].headers
    assert captured[0].headers.get("X-Aleo-environment") == "python"


@pytest.mark.asyncio
async def test_async_per_method_header() -> None:
    captured: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return jr({"height": 1})

    c = make_client_with_handler(handler)
    await c.get_block(42)
    assert captured[0].headers.get("X-ALEO-METHOD") == "getBlock"


# ---------------------------------------------------------------------------
# wait_for_confirmation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_wait_for_confirmation_accepted() -> None:
    c = make_client({
        f"{HOST}/transaction/confirmed/at1abc": jr({"status": "accepted"}),
    })
    result = await c.wait_for_transaction_confirmation("at1abc", check_interval=0.01, timeout=5.0)
    assert result["status"] == "accepted"


@pytest.mark.asyncio
async def test_async_wait_for_confirmation_rejected() -> None:
    c = make_client({
        f"{HOST}/transaction/confirmed/at1rej": jr({"status": "rejected"}),
    })
    with pytest.raises(AleoNetworkError, match="rejected"):
        await c.wait_for_transaction_confirmation("at1rej", check_interval=0.01, timeout=5.0)


@pytest.mark.asyncio
async def test_async_wait_for_confirmation_timeout() -> None:
    c = make_client({
        f"{HOST}/transaction/confirmed/at1slow": httpx.Response(404, text="not found"),
    })
    with pytest.raises(TimeoutError):
        await c.wait_for_transaction_confirmation("at1slow", check_interval=0.01, timeout=0.05)


# ---------------------------------------------------------------------------
# submit_transaction verbose_errors
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_submit_transaction_verbose_errors_true() -> None:
    captured: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return jr("at1txid")

    c = make_client_with_handler(handler)
    result = await c.submit_transaction('{"fake": "tx"}')
    assert "check_transaction=true" in str(captured[0].url)
    assert result == "at1txid"


# ---------------------------------------------------------------------------
# DPS (async) — representative subset
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.asyncio
async def test_async_dps_authorization_routes_correctly() -> None:
    """Async DPS: authorization-variant PR hits /prove/authorization; ciphertext decrypts correctly."""
    from nacl.public import PrivateKey, SealedBox

    sk = PrivateKey.generate()
    pk_b64 = base64.b64encode(bytes(sk.public_key)).decode()
    prover = f"https://prover.provable.prove/{NET}"
    captured_posts: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        if "/pubkey" in url:
            return jr(
                {"key_id": "k1", "public_key": pk_b64},
                headers={"set-cookie": "session=abc"},
            )
        if "/prove/authorization" in url and req.method == "POST":
            captured_posts.append(req)
            return jr({"transaction": "at1fake", "broadcast_result": {"status": "accepted"}})
        return httpx.Response(404)

    try:
        from pathlib import Path
        v = json.loads((Path(__file__).parent / "vectors" / "proving_request.json").read_text())
        from aleo.mainnet import ProvingRequest  # type: ignore[attr-defined]
        pr = ProvingRequest.from_string(v["PUZZLE_SPINNER_V002_PROVING_REQUEST"])
    except Exception:
        pytest.skip("ProvingRequest WASM not available")

    c = make_client_with_handler(handler, _prover_uri=prover)
    c.jwt_data = {"jwt": "Bearer testjwt", "expiration": 99999999999999}

    result = await c.submit_proving_request_safe(pr)
    assert result["ok"] is True
    assert len(captured_posts) >= 1

    # Verify ciphertext decrypts correctly
    post_body = json.loads(captured_posts[0].content)
    ciphertext = base64.b64decode(post_body["ciphertext"])
    box = SealedBox(sk)
    decrypted = box.decrypt(ciphertext)
    assert decrypted == bytes(pr.bytes())


@pytest.mark.slow
@pytest.mark.asyncio
async def test_async_dps_400_not_retried() -> None:
    """Async DPS: 400 returned verbatim, not retried."""
    from nacl.public import PrivateKey
    sk = PrivateKey.generate()
    pk_b64 = base64.b64encode(bytes(sk.public_key)).decode()
    prover = f"https://prover.provable.prove/{NET}"
    post_count = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal post_count
        url = str(req.url)
        if "/pubkey" in url:
            return jr({"key_id": "k1", "public_key": pk_b64})
        if req.method == "POST":
            post_count += 1
            return httpx.Response(400, json={"message": "bad input"})
        return httpx.Response(404)

    try:
        from pathlib import Path
        v = json.loads((Path(__file__).parent / "vectors" / "proving_request.json").read_text())
        from aleo.mainnet import ProvingRequest  # type: ignore[attr-defined]
        pr = ProvingRequest.from_string(v["PUZZLE_SPINNER_V002_PROVING_REQUEST"])
    except Exception:
        pytest.skip("ProvingRequest WASM not available")

    c = make_client_with_handler(handler, _prover_uri=prover)
    c.jwt_data = {"jwt": "Bearer jwt", "expiration": 99999999999999}

    result = await c.submit_proving_request_safe(pr)
    assert result["ok"] is False
    assert result["status"] == 400
    assert post_count == 1
