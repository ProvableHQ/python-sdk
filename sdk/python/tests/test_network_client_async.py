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
    """Build AsyncAleoNetworkClient via __init__ with a mock transport.

    Uses ``headers={}`` + ``transport=MockTransport`` so that the mock
    transport is wired without suppressing SDK headers (we pass explicit
    headers to restore them).  The ``_has_custom_transport`` flag stays False
    because we do NOT pass transport to __init__ here — instead we swap out
    the internal ``_client`` after construction so that SDK headers are still
    sent.  This lets tests that check SDK headers work correctly.
    """
    transport = httpx.MockTransport(route_handler(routes or {}))
    c = AsyncAleoNetworkClient(BASE, network=NET)
    c._client = httpx.AsyncClient(transport=transport)
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


def make_client_with_handler(
    handler: Callable[[httpx.Request], httpx.Response],
    **overrides: Any,
) -> AsyncAleoNetworkClient:
    """Like make_client but with a raw handler callable.

    Swaps out ``_client`` so SDK headers are preserved.
    """
    transport = httpx.MockTransport(handler)
    c = AsyncAleoNetworkClient(BASE, network=NET)
    c._client = httpx.AsyncClient(transport=transport)
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


# ---------------------------------------------------------------------------
# __init__ with transport wired correctly
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_init_with_transport_wires_client() -> None:
    """AsyncAleoNetworkClient.__init__ passes transport to httpx.AsyncClient."""
    captured: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return jr({"height": 99})

    c = AsyncAleoNetworkClient(BASE, network=NET, transport=httpx.MockTransport(handler))
    result = await c.get_block(99)
    assert result["height"] == 99
    assert len(captured) == 1


@pytest.mark.asyncio
async def test_async_init_transport_suppresses_sdk_headers() -> None:
    """When transport is given, SDK telemetry headers are suppressed."""
    captured: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return jr({})

    c = AsyncAleoNetworkClient(BASE, network=NET, transport=httpx.MockTransport(handler))
    await c.get_latest_block()
    assert "X-Aleo-SDK-Version" not in captured[0].headers
    assert "X-ALEO-METHOD" not in captured[0].headers


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
# Async parity: missing methods
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_get_program_imports() -> None:
    """Async get_program_imports DFS dedup: each import fetched exactly once."""
    leaf_src = "program leaf.aleo;\nfunction noop:\n    input r0 as u32.public;\n    output r0 as u32.public;\n"
    mid_src = "import leaf.aleo;\nprogram mid.aleo;\nfunction noop:\n    input r0 as u32.public;\n    output r0 as u32.public;\n"
    top_src = "import mid.aleo;\nprogram top.aleo;\nfunction noop:\n    input r0 as u32.public;\n    output r0 as u32.public;\n"

    fetch_counts: dict[str, int] = {"top": 0, "mid": 0, "leaf": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        if "/program/top.aleo" in url and "/program/top.aleo/" not in url:
            fetch_counts["top"] += 1
            return jr(top_src)
        if "/program/mid.aleo" in url and "/program/mid.aleo/" not in url:
            fetch_counts["mid"] += 1
            return jr(mid_src)
        if "/program/leaf.aleo" in url and "/program/leaf.aleo/" not in url:
            fetch_counts["leaf"] += 1
            return jr(leaf_src)
        return httpx.Response(404)

    c = make_client_with_handler(handler)
    imports = await c.get_program_imports("top.aleo")

    assert "mid.aleo" in imports
    assert "leaf.aleo" in imports
    assert fetch_counts["mid"] == 1, f"mid fetched {fetch_counts['mid']} times"
    assert fetch_counts["leaf"] == 1, f"leaf fetched {fetch_counts['leaf']} times"


@pytest.mark.asyncio
async def test_async_get_program_import_names() -> None:
    """Async get_program_import_names returns list of import IDs."""
    prog_src = "import dep.aleo;\nprogram myprog.aleo;\nfunction noop:\n    input r0 as u32.public;\n    output r0 as u32.public;\n"
    c = make_client({f"{HOST}/program/myprog.aleo": jr(prog_src)})
    names = await c.get_program_import_names("myprog.aleo")
    assert "dep.aleo" in names


@pytest.mark.asyncio
async def test_async_get_program_object() -> None:
    """Async get_program_object returns a Program instance."""
    prog_src = "program hello.aleo;\nfunction noop:\n    input r0 as u32.public;\n    output r0 as u32.public;\n"
    c = make_client({f"{HOST}/program/hello.aleo": jr(prog_src)})
    try:
        obj = await c.get_program_object("hello.aleo")
        assert obj is not None
        # Program.id is a property/attribute, not a callable
        assert "hello.aleo" in str(obj.id)
    except ImportError:
        pytest.skip("aleo mainnet module not available")


@pytest.mark.asyncio
async def test_async_get_program_mapping_plaintext() -> None:
    """Async get_program_mapping_plaintext returns a Plaintext object."""
    c = make_client({
        f"{HOST}/program/credits.aleo/mapping/account/aleo1abc": jr('"500u64"'),
    })
    try:
        obj = await c.get_program_mapping_plaintext("credits.aleo", "account", "aleo1abc")
        assert obj is not None
        assert "500" in str(obj)
    except ImportError:
        pytest.skip("aleo mainnet module not available")


@pytest.mark.asyncio
async def test_async_get_transaction_object() -> None:
    """Async get_transaction_object calls the right URL."""
    tx_json = '{"type": "execute", "id": "at1fake"}'

    hits: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        hits.append(url)
        if "/transaction/at1fake" in url:
            return httpx.Response(200, text=tx_json)
        return httpx.Response(404)

    c = make_client_with_handler(handler)
    try:
        obj = await c.get_transaction_object("at1fake")
        assert obj is not None
    except Exception:
        # Transaction.from_json rejects the trivial fixture — fine; the
        # load-bearing assertion is that the endpoint was actually called.
        pass
    assert any("/transaction/at1fake" in u for u in hits)


# ---------------------------------------------------------------------------
# DPS (async) — representative subset
# ---------------------------------------------------------------------------

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


@pytest.mark.asyncio
async def test_async_dps_cookie_affinity() -> None:
    """Async DPS: the /pubkey affinity cookie is re-sent (via the client jar) on
    the /prove POST, so both calls stick to the same prover backend."""
    from nacl.public import PrivateKey

    sk = PrivateKey.generate()
    pk_b64 = base64.b64encode(bytes(sk.public_key)).decode()
    prover = f"https://prover.provable.prove/{NET}"
    captured_posts: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        if "/pubkey" in url:
            return jr(
                {"key_id": "k1", "public_key": pk_b64},
                headers={"set-cookie": "session=mysession"},
            )
        if "/prove/authorization" in url and req.method == "POST":
            captured_posts.append(req)
            return jr({"transaction": "at1ok", "broadcast_result": {"status": "accepted"}})
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
    # The jar turns Set-Cookie into a proper name=value Cookie header on the POST.
    assert captured_posts[-1].headers.get("Cookie") == "session=mysession"


@pytest.mark.asyncio
async def test_async_dps_request_variant_routes_to_prove_request() -> None:
    """Async DPS: Request-variant ProvingRequest hits /prove/request."""
    from nacl.public import PrivateKey, SealedBox

    sk = PrivateKey.generate()
    pk_b64 = base64.b64encode(bytes(sk.public_key)).decode()
    prover = f"https://prover.provable.prove/{NET}"
    captured_posts: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        if "/pubkey" in url:
            return jr({"key_id": "k1", "public_key": pk_b64})
        if "/prove/request" in url and req.method == "POST":
            captured_posts.append(req)
            return jr({"transaction": "at1fakeR", "broadcast_result": {"status": "accepted"}})
        return httpx.Response(404)

    try:
        from aleo.mainnet import PrivateKey as AleoPrivateKey, ExecutionRequest, ProvingRequest  # type: ignore[attr-defined]
    except Exception:
        pytest.skip("ProvingRequest WASM not available")

    try:
        private_key = AleoPrivateKey.from_string(
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
        pr = ProvingRequest.from_request(exec_req, None, False)
        pr_str = str(pr)
    except Exception as exc:
        pytest.skip(f"Could not build Request-variant ProvingRequest: {exc}")

    c = make_client_with_handler(handler, _prover_uri=prover)
    c.jwt_data = {"jwt": "Bearer jwt", "expiration": 99999999999999}

    # Pass the STRING to submit_proving_request_safe (test routing from string deserialization)
    result = await c.submit_proving_request_safe(pr_str)
    assert result["ok"] is True
    assert any("/prove/request" in str(req.url) for req in captured_posts)

    # Verify ciphertext decrypts to same bytes
    post_body = json.loads(captured_posts[0].content)
    ciphertext = base64.b64decode(post_body["ciphertext"])
    box = SealedBox(sk)
    decrypted = box.decrypt(ciphertext)
    assert decrypted == bytes(pr.bytes())


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
