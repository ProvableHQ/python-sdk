"""AsyncShieldSwap owned-position views, exercised rather than introspected.

The parity test asserts these methods *exist*; this one asserts they *run*.
That distinction matters: the first version of the async view failed to await
record_provider.find (which is `async def` on the async facade), so it returned
a coroutine and raised TypeError on iteration — invisible to a hasattr check.
"""
from __future__ import annotations

import pytest

from aleo_shield_swap.async_client import AsyncShieldSwap

from .test_owned_positions import (
    POSITION_ENTRY,
    POSITION_RECORD,
    TOKEN_RECORD,
    _tick,
)
from .conftest import POOL_TEXT, SLOT_TEXT

pytestmark = pytest.mark.asyncio


class _AsyncProvider:
    """Mirrors the async facade: find() is a coroutine."""

    def __init__(self, records):
        self._records = records
        self.calls = 0

    async def find(self, account=None, *, program=None, unspent=True, **_):
        self.calls += 1
        return list(self._records)


class _AsyncMapping:
    def __init__(self, values):
        self._values = values

    async def get(self, key):
        return self._values.get(str(key))


class _AsyncProgram:
    def __init__(self, mappings):
        self._mappings = mappings

    def mapping(self, name):
        return _AsyncMapping(self._mappings.get(name, {}))


class _AsyncPrograms:
    def __init__(self, mappings):
        self._mappings = mappings

    async def get(self, program_id, edition=None):
        return _AsyncProgram(self._mappings)


class _AsyncFacade:
    network_name = "testnet"

    def __init__(self, mappings, records):
        self.programs = _AsyncPrograms(mappings)
        self.record_provider = _AsyncProvider(records)
        self.default_account = object()


def _dex(*, positions=None, records=None):
    mappings = {
        "pools": {"5field": POOL_TEXT},
        "slots": {"5field": SLOT_TEXT},
        "positions": positions if positions is not None else {"42field": POSITION_ENTRY},
    }
    facade = _AsyncFacade(mappings, records if records is not None
                          else [{"record_plaintext": POSITION_RECORD}])
    dex = AsyncShieldSwap(facade)
    mappings["ticks"] = {
        dex.derive_tick_key("5field", -4080): _tick("5field", -4080),
        dex.derive_tick_key("5field", 4080): _tick("5field", 4080),
    }
    return dex


async def test_async_join_matches_the_sync_shape():
    owned = await _dex().get_owned_positions()
    assert len(owned) == 1
    p = owned[0]
    assert p.position_token_id == "42field"
    assert (p.pool_key, p.tick_lower, p.tick_upper) == ("5field", -4080, 4080)
    assert p.withdrawal == "aleo1payout"
    assert p.state is not None and p.state.liquidity == 500
    assert (p.state.tokens_owed0, p.state.tokens_owed1) == (11, 22)


async def test_async_state_none_while_finalizing():
    owned = await _dex(positions={}).get_owned_positions()
    assert owned[0].state is None


async def test_async_skips_non_position_records():
    dex = _dex(records=[{"record_plaintext": TOKEN_RECORD},
                        {"record_plaintext": POSITION_RECORD}])
    owned = await dex.get_owned_positions()
    assert [p.position_token_id for p in owned] == ["42field"]


async def test_async_pool_filter():
    dex = _dex()
    assert len(await dex.get_owned_positions(pool_key="5field")) == 1
    assert await dex.get_owned_positions(pool_key="9field") == []


async def test_async_get_owned_position_by_id():
    dex = _dex()
    got = await dex.get_owned_position("42field")
    assert got is not None and got.position_token_id == "42field"
    assert await dex.get_owned_position("999field") is None


async def test_async_no_records_is_empty():
    assert await _dex(records=[]).get_owned_positions() == []
