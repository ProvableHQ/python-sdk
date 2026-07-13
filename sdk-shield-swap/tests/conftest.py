"""Shared stubs: a facade client stub whose shapes mirror the REAL facade.

Shape fidelity matters — ``TransactionResult.outputs`` is a property,
``decoded()``/``transitions()`` are methods, executions order transitions
child-first with the root LAST, and writes require programs registered with
the process.  The stubs model all of that so shape bugs fail here instead
of on testnet.
"""
from __future__ import annotations

import pytest

from aleo_shield_swap.tick_math import Q64

SLOT_TEXT = ("{ tick: 4055i32, tick_spacing: 60i32, sqrt_price: " + str(Q64) + "u128, "
             "fee_protocol: 0u8, liquidity: 1000u128, fee_growth_global0_x_64: 0u128, "
             "fee_growth_global1_x_64: 0u128, fee_residual0_x_64: 0u128, "
             "fee_residual1_x_64: 0u128, max_liquidity_per_tick: 0u128, "
             "protocol_fees0: 0u128, protocol_fees1: 0u128, "
             "next_init_below: 3960i32, next_init_above: 4080i32 }")

POOL_TEXT = ("{ token0: 1field, token1: 2field, fee: 3000u16, enabled: true, "
             "scale0: 1000000000u128, scale1: 1u128 }")

RECORD_TEXT = ("{ owner: aleo1me.private, amount: 2000000000u128.private, "
               "_nonce: 7group.public }")

# Vector account from test_blinding.py — next_blinded_identity derives real values.
VIEW_KEY_SCALAR = "334926304971763782347498121479281870911723639068413954564748091722770623877scalar"
SIGNER = "aleo1rhgdu77hgyqd3xjj8ucu3jj9r2krwz6mnzyd80gncr5fxcwlh5rsvzp9px"
BLINDING_FACTOR_0 = "4588552248780721950435785476596782217652350429588181106944985529417784595808field"
BLINDED_ADDRESS_0 = "aleo1tucdl48jvu54emu9atq3vf0rslwtdpze83zcc2jrc8zxema0r5gq3zd76l"

PROGRAM_ID = "shield_swap_v3.aleo"

# A decoy child transition emitting a field output — root-scoped harvesting
# must NEVER pick this up.
CHILD_TRANSITION = {"program": "tok.aleo", "function": "transfer",
                    "outputs": [{"value": "999field"}]}


def _valid_source(pid: str) -> str:
    """Minimal REAL parseable program source (register_program_sources parses
    it with the actual snarkVM Program type)."""
    name = pid.removesuffix(".aleo").replace(".", "_")
    return (f"program {name}.aleo;\n"
            "function main:\n"
            "    input r0 as u64.public;\n"
            "    output r0 as u64.public;\n")


class StubAccount:
    class _VK:
        def to_scalar(self):
            return VIEW_KEY_SCALAR

    view_key = _VK()
    address = SIGNER


class _Mapping:
    def __init__(self, values):
        self.values = values

    def get(self, key):
        return self.values.get(key)


class _StubTransition:
    def __init__(self, program, function, output_values):
        self.program_id = program
        self.function_name = function
        self._outputs = output_values

    def outputs(self):
        return list(self._outputs)


class _Tx:
    """Shape-faithful TransactionResult stub: outputs is a PROPERTY,
    decoded()/transitions() are methods, root transition LAST."""

    id = "at1stubtx"
    raw = object()

    def __init__(self, fn):
        self._fn = fn

    @property
    def outputs(self):
        return [[{"value": "999field"}], [{"value": "77field"}]]

    def decoded(self):
        return [dict(CHILD_TRANSITION),
                {"program": PROGRAM_ID, "function": self._fn,
                 "outputs": [{"value": "77field"}]}]

    def transitions(self):
        return [_StubTransition("tok.aleo", "transfer", ["999field"]),
                _StubTransition(PROGRAM_ID, self._fn, ["77field"])]


class _BoundCall:
    def __init__(self, recorder, fn, args):
        self.program_id = PROGRAM_ID
        self.function_name = fn
        self._recorder = recorder
        self._recorder.last_call = (fn, list(args))

    def simulate(self, account=None):
        return "simulated"

    def build_transaction(self, account=None, **kw):
        return _Tx(self.function_name)

    def delegate(self, account=None, **kw):
        self._recorder.delegated_fn = self.function_name
        return {"transaction_id": "at1delegated"}


class _Functions:
    def __init__(self, recorder):
        self._recorder = recorder

    def __getattr__(self, fn):
        def call(*args):
            return _BoundCall(self._recorder, fn, args)
        return call


class _Program:
    def __init__(self, recorder, mappings, pid):
        self._recorder = recorder
        self._mappings = mappings
        self.functions = _Functions(recorder)
        self.source = _valid_source(pid)

    def mapping(self, name):
        return _Mapping(self._mappings.get(name, {}))


class _Programs:
    def __init__(self, recorder, mappings):
        self._recorder = recorder
        self._mappings = mappings

    def get(self, pid):
        self._recorder.fetched_programs.append(pid)
        return _Program(self._recorder, self._mappings, pid)


class _Process:
    def __init__(self, recorder):
        self._recorder = recorder

    def contains_program(self, program_id):
        return False

    def add_program(self, program):
        self._recorder.registered_programs.append(str(program.id))


class _NetworkClient:
    def get_latest_height(self):
        return 1000


class _Network:
    def __init__(self, recorder):
        self._recorder = recorder

    def submit_transaction(self, raw):
        self._recorder.submitted.append(raw)
        return "at1stubtx"

    def wait_for_transaction(self, tx_id, **kw):
        self._recorder.waited.append(tx_id)
        return {"status": "confirmed"}

    def get_transaction_object(self, tx_id):
        return _Tx(self._recorder.delegated_fn)


class _Provider:
    def __init__(self, records):
        self._records = records

    def find(self, account=None, *, program=None, unspent=True, **_):
        return self._records


class StubAleo:
    """Stubbed facade Aleo: mappings, functions, records, network, process."""

    network_name = "testnet"

    def __init__(self, mappings=None, records=None):
        self.last_call = None
        self.delegated_fn = None
        self.submitted = []
        self.waited = []
        self.fetched_programs = []
        self.registered_programs = []
        self.programs = _Programs(self, mappings or {})
        self.record_provider = _Provider(records if records is not None
                                         else [{"record_plaintext": RECORD_TEXT}])
        self.network = _Network(self)
        self.network_client = _NetworkClient()
        self.process = _Process(self)
        self.default_account = StubAccount()


@pytest.fixture
def stub_aleo():
    return StubAleo(mappings={
        "pools": {"5field": POOL_TEXT},
        "slots": {"5field": SLOT_TEXT},
        "swap_outputs": {},
        "used_blinded_addresses": {},
        "initialized_pools": {"5field": "true"},
    })
