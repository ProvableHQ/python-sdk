"""Live testnet object suite — network-correctness guard.

These tests exercise the SDK's MAJOR OBJECTS (``AleoNetworkClient`` /
``AsyncAleoNetworkClient`` and the ``Aleo`` / ``AsyncAleo`` facade) against the
**REAL testnet** REST API.  They are the systematic guard that would have caught
the two network-awareness bugs fixed in f9e7494 (scanner) and bd13ea5 (network
client): both parsed testnet responses into *mainnet* extension types because
the parse sites hardcoded ``from .mainnet import ...`` regardless of
``self._network``.  The fix routes every parse through ``self._net()`` /
``programs._net()``, which picks the module matching the configured network.

The guard here is structural, not value-based: a client built with
``network="testnet"`` must parse testnet node responses into **testnet**
extension types.  The extension modules compile to ``builtins``-named classes,
so we cannot assert on ``type(...).__module__``; instead we assert the calls
**succeed and return correctly-shaped objects** (``.transitions()`` works, a
``Program`` parses, a mapping value comes back).  A regression to the old
hardcoded-mainnet path would surface here as a cross-extension parse failure or
a wrong-network object.

Markers / gating
----------------
* Module-level ``pytestmark = pytest.mark.live`` — real-API network tests, run
  ONLY locally with ``-m live``.  CI never runs these: the fast lanes filter
  ``-m "not slow and not live and not devnode"`` and the proving lane runs
  ``-m slow`` (``live`` is not ``slow``), so ``live`` is excluded everywhere in
  CI.  (They also need the testnet extension built + live network egress, which
  the default offline jobs don't provide.)

* Env-gated + offline-safe: ``ALEO_E2E_ENDPOINT`` (default the public explorer
  v2 root), ``network="testnet"``.  READ-ONLY — needs NO credentials and NO
  funded key.  At import time we probe the endpoint once; if it is unreachable
  (offline CI) the whole module skips cleanly via
  ``pytest.skip(allow_module_level=True)``.

* Transient 503s: live calls are wrapped in a small retry (mirroring
  ``_prepare_with_retry`` in ``tests/test_proving.py``) so nginx infra noise
  does not masquerade as a network-awareness regression.
"""
from __future__ import annotations

import os
import time
from typing import Any, Awaitable, Callable, TypeVar

import pytest

from aleo import (
    AleoNetworkClient,
    AsyncAleoNetworkClient,
    Aleo,
    AsyncAleo,
    HTTPProvider,
)

# `live` → real-API tests, run only locally with `-m live`. CI excludes `live`
# on every lane (fast lanes: `not ... and not live`; proving lane: `-m slow`).
pytestmark = pytest.mark.live

_ENDPOINT = os.environ.get(
    "ALEO_E2E_ENDPOINT", "https://api.explorer.provable.com/v2"
)
_NETWORK = "testnet"

# ── Offline-safe module gate: probe the endpoint once ───────────────────────
# If the endpoint is unreachable we skip the WHOLE module at import time so an
# offline CI run reports "skipped", not a wall of connection errors.
try:  # pragma: no cover - network-dependent
    _probe = AleoNetworkClient(_ENDPOINT, network=_NETWORK)
    _LATEST_HEIGHT: int = _probe.get_latest_height()
except Exception as _exc:  # noqa: BLE001 - any connection failure ⇒ skip cleanly
    pytest.skip(
        f"testnet endpoint {_ENDPOINT!r} unreachable ({_exc}) — skipping live "
        "testnet object tests.",
        allow_module_level=True,
    )


# ── Retry helpers (mirror _prepare_with_retry in tests/test_proving.py) ─────

_T = TypeVar("_T")


def _with_retry(
    fn: Callable[[], _T], *, attempts: int = 3, delay: float = 10.0
) -> _T:
    """Run *fn*, retrying transient outages (nginx 503s surface as errors)."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - HTTP failures surface broadly
            last = exc
            if attempt < attempts - 1:
                time.sleep(delay)
    assert last is not None
    raise last


async def _with_retry_async(
    fn: Callable[[], Awaitable[_T]], *, attempts: int = 3, delay: float = 10.0
) -> _T:
    """Async twin of :func:`_with_retry`."""
    import asyncio

    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < attempts - 1:
                await asyncio.sleep(delay)
    assert last is not None
    raise last


# ── Real tx-id sourcing ─────────────────────────────────────────────────────
# get_transaction_object is the call that caught the client bug, so we feed it a
# REAL testnet tx id pulled from a recent block.  Testnet is quiet — most recent
# blocks carry zero transactions — so we scan backwards (in block ranges for
# throughput) until we find a block whose `transactions` list is non-empty and
# read the first tx's id from the documented shape:
#     block["transactions"][i]["transaction"]["id"]


def _find_recent_tx_id(client: AleoNetworkClient, *, max_back: int = 4000) -> str:
    """Return a real tx id from a recent testnet block (or skip if none found)."""
    top = _with_retry(client.get_latest_height)
    start = top
    while start > top - max_back and start >= 0:
        lo = max(start - 49, 0)
        # get_block_range end is exclusive of the last element in some builds;
        # request start+1 so `start` itself is included.
        blocks = _with_retry(lambda lo=lo, hi=start + 1: client.get_block_range(lo, hi))
        for blk in blocks:
            txs = (blk.get("transactions") or []) if isinstance(blk, dict) else []
            for wrapper in txs:
                inner = wrapper.get("transaction") if isinstance(wrapper, dict) else None
                tid = inner.get("id") if isinstance(inner, dict) else None
                if tid:
                    return str(tid)
        start = lo - 1
    pytest.skip(
        f"no transaction found in the last {max_back} testnet blocks "
        "(testnet was quiet) — cannot exercise get_transaction_object."
    )


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def sync_client() -> AleoNetworkClient:
    """A testnet-configured sync network client."""
    return AleoNetworkClient(_ENDPOINT, network=_NETWORK)


@pytest.fixture(scope="module")
def known_address(sync_client: AleoNetworkClient) -> str:
    """A real testnet address with on-chain state (a current committee member).

    Sourcing it from the live committee avoids hardcoding an address that may
    later be reset on testnet; the account mapping lookup then has a real key.
    """
    committee: Any = _with_retry(sync_client.get_latest_committee)
    members = committee["members"] if isinstance(committee, dict) else {}
    assert members, "testnet committee returned no members"
    return str(next(iter(members)))


@pytest.fixture(scope="module")
def sample_tx_id(sync_client: AleoNetworkClient) -> str:
    """A real, recent testnet transaction id (module-scoped: sourced once)."""
    return _find_recent_tx_id(sync_client)


@pytest.fixture(scope="module")
def sync_facade() -> Aleo:
    """A testnet-configured :class:`Aleo` facade."""
    return Aleo(HTTPProvider(_ENDPOINT, network=_NETWORK))


def _new_async_client() -> AsyncAleoNetworkClient:
    return AsyncAleoNetworkClient(_ENDPOINT, network=_NETWORK)


def _new_async_facade() -> AsyncAleo:
    return AsyncAleo(HTTPProvider(_ENDPOINT, network=_NETWORK))


# ─────────────────────────────────────────────────────────────────────────────
# Sync AleoNetworkClient — chain reads
# ─────────────────────────────────────────────────────────────────────────────


def test_sync_get_latest_height(sync_client: AleoNetworkClient) -> None:
    height = _with_retry(sync_client.get_latest_height)
    assert isinstance(height, int)
    assert height > 0


def test_sync_get_latest_block(sync_client: AleoNetworkClient) -> None:
    block = _with_retry(sync_client.get_latest_block)
    assert isinstance(block, dict)
    # Documented block shape — header with height metadata.
    assert "header" in block
    assert "transactions" in block


def test_sync_get_state_root(sync_client: AleoNetworkClient) -> None:
    root = _with_retry(sync_client.get_state_root)
    assert isinstance(root, str)
    assert root.strip() != ""


# ─────────────────────────────────────────────────────────────────────────────
# Sync AleoNetworkClient — network-aware PARSING (the bug class)
# ─────────────────────────────────────────────────────────────────────────────


def test_sync_get_program_parses_on_testnet(sync_client: AleoNetworkClient) -> None:
    """get_program → source; feed to testnet.Program.from_source (Program path)."""
    from aleo import testnet  # type: ignore[attr-defined]

    source = _with_retry(lambda: sync_client.get_program("credits.aleo"))
    assert isinstance(source, str)
    assert "program credits.aleo" in source
    program = testnet.Program.from_source(source)
    assert str(program.id) == "credits.aleo"


def test_sync_get_program_object_is_testnet_typed(
    sync_client: AleoNetworkClient,
) -> None:
    """get_program_object routes through self._net() — must parse on testnet."""
    program = _with_retry(lambda: sync_client.get_program_object("credits.aleo"))
    assert str(program.id) == "credits.aleo"
    # transfer_public is a real credits function — proves the object is usable.
    assert any(str(f) == "transfer_public" for f in program.functions)


def test_sync_get_program_mapping_value(
    sync_client: AleoNetworkClient, known_address: str
) -> None:
    """account mapping read for a real testnet address (raw value path)."""
    value = _with_retry(
        lambda: sync_client.get_program_mapping_value(
            "credits.aleo", "account", known_address
        )
    )
    # Committee members hold a bonded balance; value is a "…u64" literal.
    assert isinstance(value, str)
    assert value.strip().endswith("u64")


def test_sync_get_program_mapping_plaintext_parses(
    sync_client: AleoNetworkClient, known_address: str
) -> None:
    """Plaintext parse path — routes through self._net().Plaintext on testnet."""
    plaintext = _with_retry(
        lambda: sync_client.get_program_mapping_plaintext(
            "credits.aleo", "account", known_address
        )
    )
    assert str(plaintext).strip().endswith("u64")


def test_sync_get_transaction_object_is_testnet_typed(
    sync_client: AleoNetworkClient, sample_tx_id: str
) -> None:
    """THE regression guard: parse a real testnet tx into a testnet Transaction.

    This is the exact call that caught the network-client bug — it used to parse
    into a mainnet ``Transaction`` on a testnet client.  Sourcing the id from a
    live block and asserting ``.transitions()`` works proves the object round-
    trips network-correctly.
    """
    tx = _with_retry(lambda: sync_client.get_transaction_object(sample_tx_id))
    transitions = list(tx.transitions())
    assert len(transitions) >= 1
    t0 = transitions[0]
    # A real transition exposes program/function/inputs/outputs.
    assert str(t0.program_id) != ""
    assert str(t0.function_name) != ""
    assert isinstance(list(t0.inputs()), list)
    assert isinstance(list(t0.outputs()), list)


# ─────────────────────────────────────────────────────────────────────────────
# Async AsyncAleoNetworkClient — same coverage, awaited
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_get_latest_height() -> None:
    client = _new_async_client()
    try:
        height = await _with_retry_async(client.get_latest_height)
        assert isinstance(height, int)
        assert height > 0
    finally:
        await client._client.aclose()  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_async_get_latest_block_and_state_root() -> None:
    client = _new_async_client()
    try:
        block = await _with_retry_async(client.get_latest_block)
        assert isinstance(block, dict)
        assert "header" in block
        root = await _with_retry_async(client.get_state_root)
        assert isinstance(root, str) and root.strip() != ""
    finally:
        await client._client.aclose()  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_async_get_program_parses_on_testnet() -> None:
    from aleo import testnet  # type: ignore[attr-defined]

    client = _new_async_client()
    try:
        source = await _with_retry_async(lambda: client.get_program("credits.aleo"))
        assert isinstance(source, str)
        program = testnet.Program.from_source(source)
        assert str(program.id) == "credits.aleo"
    finally:
        await client._client.aclose()  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_async_get_program_mapping_value(known_address: str) -> None:
    client = _new_async_client()
    try:
        value = await _with_retry_async(
            lambda: client.get_program_mapping_value(
                "credits.aleo", "account", known_address
            )
        )
        assert isinstance(value, str)
        assert value.strip().endswith("u64")
    finally:
        await client._client.aclose()  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_async_get_transaction_object_is_testnet_typed(
    sample_tx_id: str,
) -> None:
    """Async twin of the regression guard — parse a real testnet tx object."""
    client = _new_async_client()
    try:
        tx = await _with_retry_async(
            lambda: client.get_transaction_object(sample_tx_id)
        )
        transitions = list(tx.transitions())
        assert len(transitions) >= 1
        assert str(transitions[0].program_id) != ""
    finally:
        await client._client.aclose()  # pyright: ignore[reportPrivateUsage]


# ─────────────────────────────────────────────────────────────────────────────
# Sync Aleo facade — identity, programs, mappings, decode, account
# ─────────────────────────────────────────────────────────────────────────────


def test_facade_network_identity(sync_facade: Aleo) -> None:
    assert sync_facade.is_connected() is True
    assert sync_facade.network_id == 1  # testnet
    assert sync_facade.network_name == "testnet"


def test_facade_programs_get_and_functions(sync_facade: Aleo) -> None:
    program = _with_retry(lambda: sync_facade.programs.get("credits.aleo"))
    assert program.id == "credits.aleo"
    # `in` routes through ProgramFunctions.__contains__.
    assert "transfer_public" in program.functions


def test_facade_program_mapping_get(
    sync_facade: Aleo, known_address: str
) -> None:
    program = _with_retry(lambda: sync_facade.programs.get("credits.aleo"))
    value = _with_retry(lambda: program.mapping("account").get(known_address))
    assert isinstance(value, str)
    assert value.strip().endswith("u64")


def test_facade_decode_transition(sync_facade: Aleo, sample_tx_id: str) -> None:
    """aleo.decode_transition(<real tx id>) → {program, function, inputs, outputs}."""
    decoded = _with_retry(lambda: sync_facade.decode_transition(sample_tx_id))
    assert isinstance(decoded, dict)
    assert set(decoded.keys()) == {"program", "function", "inputs", "outputs"}
    assert decoded["program"] != ""
    assert decoded["function"] != ""
    assert isinstance(decoded["inputs"], list)
    assert isinstance(decoded["outputs"], list)


def test_facade_account_create_sign_verify(sync_facade: Aleo) -> None:
    """Account is local, but assert it on the testnet-configured client."""
    account = sync_facade.account.create()
    address = str(account.address)
    assert address.startswith("aleo1")
    assert sync_facade.is_valid_address(address)
    message = b"network-correctness guard"
    signature = sync_facade.account.sign(message, account)
    assert sync_facade.account.verify(account.address, message, signature) is True
    # Tampered message must fail.
    assert (
        sync_facade.account.verify(account.address, b"tampered", signature) is False
    )


# ─────────────────────────────────────────────────────────────────────────────
# Async AsyncAleo facade — identity, programs, mappings, decode
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_facade_network_identity() -> None:
    aleo = _new_async_facade()
    try:
        assert await aleo.is_connected() is True
        assert aleo.network_id == 1
        assert aleo.network_name == "testnet"
    finally:
        await aleo.network_client._client.aclose()  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_async_facade_programs_and_mapping(known_address: str) -> None:
    aleo = _new_async_facade()
    try:
        program = await _with_retry_async(
            lambda: aleo.programs.get("credits.aleo")
        )
        assert "transfer_public" in program.functions
        value = await _with_retry_async(
            lambda: program.mapping("account").get(known_address)
        )
        assert isinstance(value, str) and value.strip().endswith("u64")
    finally:
        await aleo.network_client._client.aclose()  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_async_facade_decode_transition(sample_tx_id: str) -> None:
    aleo = _new_async_facade()
    try:
        decoded = await _with_retry_async(
            lambda: aleo.decode_transition(sample_tx_id)
        )
        assert set(decoded.keys()) == {"program", "function", "inputs", "outputs"}
        assert decoded["program"] != ""
        assert decoded["function"] != ""
    finally:
        await aleo.network_client._client.aclose()  # pyright: ignore[reportPrivateUsage]
