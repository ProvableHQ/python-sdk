"""AsyncShieldSwap over an async stub facade — mirrors the sync assertions
for reads and the swap input order."""
import pytest

from aleo_shield_swap.async_client import AsyncShieldSwap
from aleo_shield_swap.errors import SwapOutputNotFinalizedError
from aleo_shield_swap.tick_math import MIN_SQRT_RATIO_X128, int_to_u256_plaintext

from .conftest import (
    BLINDED_ADDRESS_0,
    BLINDING_FACTOR_0,
    POOL_TEXT,
    RECORD_TEXT,
    SLOT_TEXT,
    StubAccount,
)

pytestmark = pytest.mark.asyncio


from .conftest import PROGRAM_ID, _Process, _StubTransition, _valid_source


class _AsyncMapping:
    def __init__(self, values):
        self.values = values

    async def get(self, key):
        return self.values.get(key)


class _Tx:
    """Shape-faithful async TransactionResult stub (root transition LAST)."""

    id = "at1asynctx"
    raw = object()

    def __init__(self, fn, pid=PROGRAM_ID):
        self._fn = fn
        self._pid = pid

    @property
    def outputs(self):
        return [[{"value": "999field"}], [{"value": "88field"}]]

    def decoded(self):
        return [{"program": "tok.aleo", "function": "transfer",
                 "outputs": [{"value": "999field"}]},
                {"program": self._pid, "function": self._fn,
                 "outputs": [{"value": "88field"}]}]

    def transitions(self):
        return [_StubTransition("tok.aleo", "transfer", ["999field"]),
                _StubTransition(self._pid, self._fn, ["88field"])]


class _AsyncBoundCall:
    def __init__(self, recorder, fn, args, pid=PROGRAM_ID):
        self.program_id = pid
        self.function_name = fn
        self._recorder = recorder
        recorder.last_call = (fn, list(args))
        recorder.last_program = pid

    def simulate(self, account=None):
        return "simulated"

    async def build_transaction(self, account=None, **kw):
        return _Tx(self.function_name, self.program_id)

    async def delegate(self, account=None, **kw):
        self._recorder.delegated_fn = self.function_name
        self._recorder.delegated_program = self.program_id
        return {"transaction_id": "at1delegated"}


class _AsyncFunctions:
    def __init__(self, recorder, pid):
        self._recorder = recorder
        self._pid = pid

    def __getattr__(self, fn):
        def call(*args):
            return _AsyncBoundCall(self._recorder, fn, args, self._pid)
        return call


class _AsyncProgram:
    def __init__(self, recorder, mappings, pid):
        self._mappings = mappings
        self.functions = _AsyncFunctions(recorder, pid)
        self.source = _valid_source(pid)

    def mapping(self, name):
        return _AsyncMapping(self._mappings.get(name, {}))


class _AsyncPrograms:
    def __init__(self, recorder, mappings):
        self._recorder = recorder
        self._mappings = mappings

    async def get(self, pid):
        return _AsyncProgram(self._recorder, self._mappings, pid)


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

    async def get_transaction_object(self, tx_id):
        return _Tx(self._recorder.delegated_fn,
                   self._recorder.delegated_program or PROGRAM_ID)

    async def get_program_mapping_value(self, program_id, mapping_name, key):
        self._recorder.mapping_reads.append((program_id, mapping_name, key))
        if program_id in self._recorder.missing_programs:
            from aleo import AleoNetworkError
            raise AleoNetworkError("404", status=404)
        return self._recorder._mappings.get(mapping_name, {}).get(key)


class _AsyncProvider:
    def __init__(self, records):
        self._records = records

    async def find(self, account=None, *, program=None, unspent=True, **_):
        return self._records


class AsyncStubAleo:
    network_name = "testnet"

    def __init__(self, mappings=None, records=None):
        self.last_call = None
        self.last_program = None
        self.delegated_fn = None
        self.delegated_program = None
        self.submitted = []
        self.waited = []
        self.registered_programs = []
        self.mapping_reads = []
        self.missing_programs = set()
        self._mappings = mappings or {}
        self.programs = _AsyncPrograms(self, self._mappings)
        self.record_provider = _AsyncProvider(
            records if records is not None else [{"record_plaintext": RECORD_TEXT}])
        self.network = _AsyncNetwork(self)
        self.process = _Process(self)
        self.default_account = StubAccount()


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
    assert args[7] == int_to_u256_plaintext(MIN_SQRT_RATIO_X128)
    assert args[9] == "11000u32"                  # height 1000 + the shared 10_000-block default

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


async def test_async_swap_wrapped_routes_through_router():
    from aleo_shield_swap._routing import ROUTER_ID
    astub = AsyncStubAleo(mappings={
        "pools": {"5field": POOL_TEXT},
        "slots": {"5field": SLOT_TEXT},
        "used_blinded_addresses": {},
        "from_wrapper_token_id": {"1field": "9field"},
    })
    dex = AsyncShieldSwap(astub)
    await dex.swap(pool_key="5field", token_in_id="1field", amount_in=10**9,
                   expected_out=1_000_000, token_in_program="credits.aleo")
    fn, args = astub.last_call
    assert (astub.last_program, fn) == (ROUTER_ID, "swap_from_wrapped")
    assert len(args) == 13
    assert args[1].startswith("[{ siblings: [0field")


async def test_async_claim_wrapped_output_routes_through_router():
    from aleo_shield_swap._routing import ROUTER_ID
    from aleo_shield_swap.types import SwapHandle
    out_text = ("{ recipient: 3field, caller: 4field, token_in: 1field, "
                "token_out: 2field, amount_out: 990000u128, amount_remaining: 5u128 }")
    astub = AsyncStubAleo(mappings={
        "pools": {"5field": POOL_TEXT},
        "slots": {"5field": SLOT_TEXT},
        "swap_outputs": {"77field": out_text},
        "from_wrapper_token_id": {"2field": "9field"},
    })
    handle = SwapHandle(swap_id="77field", blinding_factor="11field",
                        blinded_address="aleo1blinded", token_in_id="1field",
                        token_out_id="2field", pool_key="5field", amount_in=1,
                        transaction_id="at1req", program="shield_swap.aleo")
    dex = AsyncShieldSwap(astub)

    async def plain_program(token_id):          # the registry, stubbed
        return {"1field": "tok_in.aleo"}[token_id]

    async def wrapper_program(token_id):
        return {"2field": "wrapped_out.aleo"}[token_id]

    dex._token_program = plain_program
    dex._amm_token_program = wrapper_program
    await dex.claim_swap_output(handle)
    fn, args = astub.last_call
    assert (astub.last_program, fn) == (ROUTER_ID, "claim_to_wrapped_refund_arc20")
    assert len(args) == 9
    # Both legs' token programs are registered before authorization: the core
    # dispatches into them dynamically (payout AND refund), so a fresh async
    # process without them fails to authorize — the sync client already did this.
    assert {"tok_in.aleo", "wrapped_out.aleo", ROUTER_ID} <= set(astub.registered_programs)


async def test_async_swap_execution_and_pool_creator():
    from .test_client_reads import HEADER_TEXT, HOP0_TEXT, HOP1_TEXT
    astub = AsyncStubAleo(mappings={
        "swap_execution_headers": {"77field": HEADER_TEXT},
        "swap_execution_hops": {
            "{ swap_id: 77field, hop_index: 0u8 }": HOP0_TEXT,
            "{ swap_id: 77field, hop_index: 1u8 }": HOP1_TEXT,
        },
        "pool_creators": {"5field": "aleo1creator"},
    })
    dex = AsyncShieldSwap(astub)
    ex = await dex.get_swap_execution("77field")
    assert ex is not None and ex.executed_height == 4242
    assert [h.lp_fee for h in ex.hops] == [24, 22]
    assert await dex.get_swap_execution("99field") is None
    assert await dex.get_pool_creator("5field") == "aleo1creator"
    assert await dex.get_pool_creator("6field") is None


async def test_async_claim_no_refund_mirrors_sync_dispatch():
    from aleo_shield_swap._routing import ROUTER_ID
    from aleo_shield_swap.types import SwapHandle
    out_text = ("{ recipient: 3field, caller: 4field, token_in: 1field, "
                "token_out: 2field, amount_out: 990000u128, amount_remaining: 0u128 }")
    handle = SwapHandle(swap_id="77field", blinding_factor="11field",
                        blinded_address="aleo1blinded", token_in_id="1field",
                        token_out_id="2field", pool_key="5field", amount_in=1,
                        transaction_id="at1req", program="shield_swap.aleo")
    for wrapped, expected, count in (
        ({}, ("shield_swap.aleo", "claim_swap_output_no_refund"), 7),
        ({"2field": "9field"}, (ROUTER_ID, "claim_to_wrapped_no_refund"), 8),
        ({"1field": "9field"}, (ROUTER_ID, "claim_to_arc20_no_refund"), 7),
    ):
        astub = AsyncStubAleo(mappings={
            "pools": {"5field": POOL_TEXT}, "slots": {"5field": SLOT_TEXT},
            "swap_outputs": {"77field": out_text},
            "from_wrapper_token_id": wrapped,
        })
        dex = AsyncShieldSwap(astub)

        async def plain_program(token_id):       # the registry, stubbed
            return f"arc20_{token_id}.aleo"

        async def wrapper_program(token_id):
            return f"wrapper_{token_id}.aleo"

        dex._token_program = plain_program
        dex._amm_token_program = wrapper_program
        await dex.claim_swap_output(handle)
        fn, args = astub.last_call
        assert (astub.last_program, fn) == expected
        assert len(args) == count
        # Every leg's token program is registered: wrapper for wrapped legs,
        # the ARC-20 itself for plain ones (the core dispatches into both).
        for token_id in ("1field", "2field"):
            want = f"wrapper_{token_id}.aleo" if token_id in wrapped else f"arc20_{token_id}.aleo"
            assert want in astub.registered_programs, (wrapped, astub.registered_programs)
        assert "0u128" not in args           # amount_remaining is not an input


async def test_async_public_balances_are_chain_reads():
    aleo = AsyncStubAleo(mappings={"balances": {"aleo1x": "9u64"}})
    dex = AsyncShieldSwap(aleo)
    assert await dex.get_public_balances(["a.aleo", "a.aleo", "b.aleo"], address="aleo1x") == {
        "a.aleo": 9, "b.aleo": 9}
    assert await dex.get_public_balances(["a.aleo"]) == {"a.aleo": 0}   # bound account, absent
    aleo.default_account = None
    with pytest.raises(ValueError):
        await dex.get_public_balances(["a.aleo"])


async def test_async_identity_probe_gallops_past_a_long_used_run():
    """The async client shares the sync client's galloping search: an account
    with 70 swaps on chain gets counter 70 in a short gallop, not a
    ValueError after a fixed 64-counter window — and never downloads the
    program to do it."""
    from aleo_shield_swap.derivations import BlindedIdentityCache
    acct = StubAccount()
    dex_program = AsyncShieldSwap(AsyncStubAleo()).program
    cache = BlindedIdentityCache("testnet", acct, dex_program)
    used = {cache.at(c).blinded_address: "true" for c in range(70)}
    aleo = AsyncStubAleo(mappings={"used_blinded_addresses": used})
    ident = await AsyncShieldSwap(aleo)._next_blinded_identity(acct)
    assert ident.counter == 70 and ident.blinded_address == cache.at(70).blinded_address
    probes = [r for r in aleo.mapping_reads if r[1] == "used_blinded_addresses"]
    assert 64 < len(probes) < 80
    assert all(r[0] == dex_program for r in probes)


async def test_async_swap_takes_an_explicit_identity(astub):
    from aleo_shield_swap.derivations import BlindedIdentity
    dex = AsyncShieldSwap(astub)
    ident = BlindedIdentity(9, "bf9", BLINDED_ADDRESS_0)
    await dex.swap(pool_key="5field", token_in_id="1field", amount_in=10**9, nonce=1,
                   expected_out=1_000_000, token_in_program="tok.aleo", identity=ident)
    _, args = astub.last_call
    assert args[1] == "bf9"                          # the caller's identity, verbatim
    assert not [r for r in astub.mapping_reads if r[1] == "used_blinded_addresses"]


async def test_async_public_balances_isolate_one_bad_program(caplog):
    aleo = AsyncStubAleo(mappings={"balances": {"aleo1x": "9u64"}})
    aleo.missing_programs.add("gone.aleo")
    with caplog.at_level("WARNING"):
        out = await AsyncShieldSwap(aleo).get_public_balances(["a.aleo", "gone.aleo"], address="aleo1x")
    assert out == {"a.aleo": 9} and "gone.aleo" in caplog.text
