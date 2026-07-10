"""Tests for F7 facade: AsyncAleo async facade.

All tests are FAST and OFFLINE — no live network.  Network I/O is mocked via
httpx.MockTransport (same pattern used by test_network_client_async.py).

Coverage (parity proof set, ~12 tests):
- AsyncAleo client wiring: network_id / network_name / repr.
- is_connected true / false.
- One awaited network read (get_program_mapping_value).
- programs.get + dir(functions) includes transfer_public + coercion assert.
- authorize returns AuthorizationResult (local, sync call on async facade).
- delegate with mocked async DPS: fee_authorization=None by default + submit awaited.
- async records: find passes the filter through; get_unspent covering/None.
- one awaited mapping read.
- account.create / account.sign (sync) on AsyncAleo.
"""
from __future__ import annotations

from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import httpx

from aleo import AsyncAleo, HTTPProvider
from aleo.mainnet import Program as RawProgram, PrivateKey
from aleo.facade.async_client import (
    AsyncBoundCall,
    AsyncNetworkModule,
    AsyncProgramsModule,
    AsyncRecordsModule,
)
from aleo.facade.call import AuthorizationResult

BASE = "https://api.provable.com/v2"
NET = "mainnet"
HOST = f"{BASE}/{NET}"

CREDITS_SOURCE = RawProgram.credits().source


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def route_handler(
    routes: dict[str, Any],
) -> Callable[[httpx.Request], httpx.Response]:
    """httpx.MockTransport handler (same idiom as test_network_client_async.py)."""
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        for pattern, response in routes.items():
            if pattern in url:
                if callable(response):
                    return response(request)
                return response
        return httpx.Response(404, text="not found")
    return handler


def jr(data: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=data)


def _make_aleo(routes: dict[str, Any] | None = None) -> AsyncAleo:
    """Build an AsyncAleo wired to an httpx.MockTransport."""
    transport = httpx.MockTransport(route_handler(routes or {}))
    a = AsyncAleo(HTTPProvider(BASE, network=NET))
    # Swap the internal async client's httpx.AsyncClient for a mocked one
    a._async_client._client = httpx.AsyncClient(transport=transport)
    return a


def _account(a: AsyncAleo) -> Any:
    return a.account.from_private_key(PrivateKey.random())


# ---------------------------------------------------------------------------
# Client wiring: network_id / network_name / repr (sync, no await)
# ---------------------------------------------------------------------------


def test_async_aleo_network_id() -> None:
    a = AsyncAleo(HTTPProvider(BASE, network="mainnet"))
    assert a.network_id == 0


def test_async_aleo_network_name() -> None:
    a = AsyncAleo(HTTPProvider(BASE, network="mainnet"))
    assert a.network_name == "mainnet"


def test_async_aleo_repr() -> None:
    a = AsyncAleo(HTTPProvider(BASE))
    assert "AsyncAleo" in repr(a)
    assert "HTTPProvider" in repr(a)


def test_async_aleo_httpprovider_shortcut() -> None:
    """AsyncAleo.HTTPProvider is the same class as HTTPProvider."""
    assert AsyncAleo.HTTPProvider is HTTPProvider


# ---------------------------------------------------------------------------
# is_connected (async)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_is_connected_true() -> None:
    a = _make_aleo({f"{HOST}/block/height/latest": jr(100)})
    assert await a.is_connected() is True


@pytest.mark.asyncio
async def test_is_connected_false() -> None:
    # All routes return 404 → connection probe fails → False
    a = _make_aleo({})
    assert await a.is_connected() is False


# ---------------------------------------------------------------------------
# One awaited network read: get_program_mapping_value
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_awaited_mapping_read_via_network_client() -> None:
    """Directly awaiting the async network client works."""
    a = _make_aleo(
        {f"{HOST}/program/credits.aleo/mapping/account/aleo1qqqq": jr("1000000u64")}
    )
    val = await a._async_client.get_program_mapping_value(
        "credits.aleo", "account", "aleo1qqqq"
    )
    assert "1000000" in str(val)


# ---------------------------------------------------------------------------
# programs.get + dir(functions) + coercion (async fetch, sync ProgramFunctions)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_programs_get_and_functions_dir() -> None:
    """programs.get fetches source and builds a ProgramFunctions with function names."""
    a = _make_aleo({f"{HOST}/program/credits.aleo": jr(CREDITS_SOURCE)})
    prog = await a.programs.get("credits.aleo")
    assert "transfer_public" in dir(prog.functions)
    assert "transfer_public" in prog.functions


@pytest.mark.asyncio
async def test_programs_get_coercion() -> None:
    """Coercion on the async program's functions is sync — test it directly."""
    a = _make_aleo({f"{HOST}/program/credits.aleo": jr(CREDITS_SOURCE)})
    prog = await a.programs.get("credits.aleo")
    acct = _account(a)
    bc = prog.functions.transfer_public(str(acct.address), 10)
    assert isinstance(bc, AsyncBoundCall)
    # Coercion: bare int 10 + u64 type → "10u64"
    assert bc.args[1] == "10u64"


# ---------------------------------------------------------------------------
# authorize (local, SYNC on async facade)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_authorize_is_sync_and_returns_authorization_result() -> None:
    """AsyncBoundCall.authorize is synchronous (local Rust, no network)."""
    a = AsyncAleo(HTTPProvider(BASE))
    acct = _account(a)
    raw = RawProgram.from_source(CREDITS_SOURCE)
    from aleo.facade.async_client import AsyncProgram
    prog = AsyncProgram(a, raw)
    bc = prog.functions.transfer_public(str(acct.address), 10)
    # No await — authorize is sync
    result = bc.authorize(acct)
    assert isinstance(result, AuthorizationResult)
    assert result.function_name == "transfer_public"


# ---------------------------------------------------------------------------
# Async mapping read via programs module
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_mapping_get() -> None:
    """program.mapping(name).get(key) is awaitable."""
    # Order matters: put the more specific (longer) pattern first so it matches
    # before the shorter program-source route.
    routes: dict[str, Any] = {}
    routes[f"{HOST}/program/credits.aleo/mapping/account/aleo1qqqq"] = jr("500u64")
    routes[f"{HOST}/program/credits.aleo"] = jr(CREDITS_SOURCE)
    a = _make_aleo(routes)
    prog = await a.programs.get("credits.aleo")
    val = await prog.mapping("account").get("aleo1qqqq")
    assert "500" in str(val)


# ---------------------------------------------------------------------------
# delegate with mocked async DPS submit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delegate_fee_master_pays_by_default() -> None:
    """Default delegate: fee_authorization=None + submit_proving_request awaited."""
    a = AsyncAleo(HTTPProvider(BASE))
    acct = _account(a)
    raw = RawProgram.from_source(CREDITS_SOURCE)
    from aleo.facade.async_client import AsyncProgram
    prog = AsyncProgram(a, raw)
    bc = prog.functions.transfer_public(str(acct.address), 10)

    mock_result = {"transaction_id": "at1asyncmock"}
    mock_submit = AsyncMock(return_value=mock_result)
    a._async_client.submit_proving_request = mock_submit

    out = await bc.delegate(acct)
    assert out == mock_result
    mock_submit.assert_awaited_once()
    request = mock_submit.call_args.args[0]
    # No fee authorization by default — fee master pays
    assert request.fee_authorization() is None
    assert str(request.authorization().function_name()) == "transfer_public"
    assert request.broadcast is True


@pytest.mark.asyncio
async def test_delegate_broadcast_false() -> None:
    a = AsyncAleo(HTTPProvider(BASE))
    acct = _account(a)
    raw = RawProgram.from_source(CREDITS_SOURCE)
    from aleo.facade.async_client import AsyncProgram
    prog = AsyncProgram(a, raw)
    bc = prog.functions.transfer_public(str(acct.address), 10)

    mock_submit = AsyncMock(return_value={})
    a._async_client.submit_proving_request = mock_submit

    await bc.delegate(acct, broadcast=False)
    request = mock_submit.call_args.args[0]
    assert request.broadcast is False
    assert request.fee_authorization() is None


# ---------------------------------------------------------------------------
# Async records: find filter pass-through + get_unspent covering / None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_records_find_passes_filter() -> None:
    """AsyncRecordsModule.find passes the OwnedFilter to the scanner."""
    a = AsyncAleo(HTTPProvider(BASE))
    acct = _account(a)

    captured: list[Any] = []

    async def fake_find_records(filt: Any) -> list[Any]:
        captured.append(filt)
        return []

    a.records.scanner.set_account(acct)
    a.records.scanner.find_records = fake_find_records  # type: ignore[method-assign]

    await a.records.find(
        acct, program="credits.aleo", record="credits", unspent=True
    )
    assert len(captured) == 1
    filt = captured[0]
    assert filt["unspent"] is True
    assert filt["filter"]["program"] == "credits.aleo"
    assert filt["filter"]["record"] == "credits"


@pytest.mark.asyncio
async def test_async_records_get_unspent_covering() -> None:
    """get_unspent returns a parsed RecordPlaintext when a covering record exists."""
    a = AsyncAleo(HTTPProvider(BASE))
    acct = _account(a)

    fake_plaintext = MagicMock()
    fake_plaintext.nonce = "fake_nonce"
    fake_plaintext.microcredits = 2_000_000

    # Mock find_credits at the class level (patch.object on the class) so self is
    # passed normally; our replacement accepts (self, account=None, at_least=None).
    async def fake_find_credits(
        self_: Any, account: Any = None, at_least: int | None = None
    ) -> list[Any]:
        return [{"record_plaintext": "fake_str", "unspent": True}]

    fake_net = MagicMock()
    fake_net.RecordPlaintext.from_string.return_value = fake_plaintext

    with patch.object(AsyncRecordsModule, "find_credits", fake_find_credits):
        with patch.object(a.records, "_net", return_value=fake_net):
            result = await a.records.get_unspent(
                program="credits.aleo",
                record="credits",
                min_microcredits=1_000_000,
            )
    assert result is fake_plaintext


@pytest.mark.asyncio
async def test_async_records_get_unspent_none_when_no_candidates() -> None:
    """get_unspent returns None when find_credits returns empty list."""
    a = AsyncAleo(HTTPProvider(BASE))

    async def fake_find_credits(
        self_: Any, account: Any = None, at_least: int | None = None
    ) -> list[Any]:
        return []

    with patch.object(AsyncRecordsModule, "find_credits", fake_find_credits):
        result = await a.records.get_unspent(
            program="credits.aleo",
            record="credits",
            min_microcredits=1_000_000,
        )
    assert result is None


# ---------------------------------------------------------------------------
# Account create / sign (sync) on AsyncAleo
# ---------------------------------------------------------------------------


def test_account_create_sync_on_async_aleo() -> None:
    """account.create() is sync even on the async facade."""
    a = AsyncAleo(HTTPProvider(BASE))
    acct = a.account.create()
    assert hasattr(acct, "private_key")
    assert hasattr(acct, "address")


def test_account_sign_sync_on_async_aleo() -> None:
    """account.sign(message, account) is sync and produces a valid signature."""
    a = AsyncAleo(HTTPProvider(BASE))
    acct = a.account.create()
    import os
    data = os.urandom(32)
    sig = a.account.sign(data, acct)
    assert sig is not None
    assert a.account.verify(str(acct.address), data, sig) is True


# ---------------------------------------------------------------------------
# Export check
# ---------------------------------------------------------------------------


def test_async_aleo_exported_from_top_level_aleo() -> None:
    """AsyncAleo is importable from the top-level aleo package."""
    from aleo import AsyncAleo as AA  # noqa: F401
    assert AA is AsyncAleo
