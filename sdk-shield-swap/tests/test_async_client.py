"""AsyncShieldSwap over an async stub facade — mirrors the sync assertions
for reads and the swap input order."""
import pytest

from aleo_shield_swap.async_client import AsyncShieldSwap
from aleo_shield_swap.errors import SwapOutputNotFinalizedError
from aleo_shield_swap.tick_math import MIN_SQRT_PRICE

from .conftest import (
    BLINDED_ADDRESS_0,
    BLINDING_FACTOR_0,
    POOL_TEXT,
    RECORD_TEXT,
    SLOT_TEXT,
    StubAccount,
)

pytestmark = pytest.mark.asyncio


class _AsyncMapping:
    def __init__(self, values):
        self.values = values

    async def get(self, key):
        return self.values.get(key)


class _Tx:
    id = "at1asynctx"
    raw = object()

    def outputs(self):
        return [[{"value": "88field"}]]


class _AsyncBoundCall:
    def __init__(self, recorder, fn, args):
        recorder.last_call = (fn, list(args))

    def simulate(self, account=None):
        return "simulated"

    async def build_transaction(self, account=None, **kw):
        return _Tx()

    async def delegate(self, account=None, **kw):
        return {"transaction_id": "at1delegated"}


class _AsyncFunctions:
    def __init__(self, recorder):
        self._recorder = recorder

    def __getattr__(self, fn):
        def call(*args):
            return _AsyncBoundCall(self._recorder, fn, args)
        return call


class _AsyncProgram:
    def __init__(self, recorder, mappings):
        self._mappings = mappings
        self.functions = _AsyncFunctions(recorder)

    def mapping(self, name):
        return _AsyncMapping(self._mappings.get(name, {}))


class _AsyncPrograms:
    def __init__(self, recorder, mappings):
        self._recorder = recorder
        self._mappings = mappings

    async def get(self, pid):
        return _AsyncProgram(self._recorder, self._mappings)


class _AsyncNetwork:
    def __init__(self, recorder):
        self._recorder = recorder

    async def get_latest_height(self):
        return 1000

    async def submit_transaction(self, raw):
        self._recorder.submitted.append(raw)
        return "at1asynctx"

    async def wait_for_transaction(self, tx_id, **kw):
        self._recorder.waited.append(tx_id)


class _AsyncProvider:
    def __init__(self, records):
        self._records = records

    async def find(self, account=None, *, program=None, unspent=True, **_):
        return self._records


class AsyncStubAleo:
    network_name = "testnet"

    def __init__(self, mappings=None, records=None):
        self.last_call = None
        self.submitted = []
        self.waited = []
        self.programs = _AsyncPrograms(self, mappings or {})
        self.record_provider = _AsyncProvider(
            records if records is not None else [{"record_plaintext": RECORD_TEXT}])
        self.network = _AsyncNetwork(self)
        self.default_account = StubAccount()

    async def decode_transition(self, tx_id):
        return {"outputs": [{"value": "88field"}]}


@pytest.fixture
def astub():
    return AsyncStubAleo(mappings={
        "pools": {"5field": POOL_TEXT},
        "slots": {"5field": SLOT_TEXT},
        "swap_outputs": {},
        "used_blinded_addresses": {},
    })


async def test_async_reads(astub):
    dex = AsyncShieldSwap(astub)
    slot = await dex.get_slot("5field")
    assert slot.tick == 4055
    pool = await dex.get_pool("5field")
    assert pool.fee == 3000
    with pytest.raises(SwapOutputNotFinalizedError):
        await dex.get_swap_output("9field")


async def test_async_swap_inputs_and_handle(astub):
    dex = AsyncShieldSwap(astub)
    call = await dex.swap(pool_key="5field", token_in_id="1field",
                          amount_in=10**9, nonce=123, expected_out=1_000_000,
                          token_in_program="tok.aleo")
    fn, args = astub.last_call
    assert fn == "swap" and len(args) == 12
    assert args[0] == RECORD_TEXT
    assert args[1] == BLINDING_FACTOR_0 and args[2] == BLINDED_ADDRESS_0
    assert args[7] == f"{MIN_SQRT_PRICE}u128"
    assert args[9] == "1100u32"

    handle = await call.transact()
    assert handle.swap_id == "88field"
    assert handle.transaction_id == "at1asynctx"
    assert astub.submitted


async def test_async_delegate_waits_and_recovers(astub):
    dex = AsyncShieldSwap(astub)
    call = await dex.swap(pool_key="5field", token_in_id="1field",
                          amount_in=10**9, expected_out=1_000_000,
                          token_in_program="tok.aleo")
    handle = await call.delegate()
    assert handle.transaction_id == "at1delegated"
    assert handle.swap_id == "88field"
    assert astub.waited == ["at1delegated"]
