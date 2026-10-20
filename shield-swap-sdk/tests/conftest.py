"""Shared stubs: a facade client stub whose shapes mirror the REAL facade.

Shape fidelity matters — ``TransactionResult.outputs`` is a property,
``decoded()``/``transitions()`` are methods, executions order transitions
child-first with the root LAST, and writes require programs registered with
the process.  The stubs model all of that so shape bugs fail here instead
of on testnet.
"""
from __future__ import annotations

import pytest

SLOT_TEXT = ("{ tick: 4055i32, tick_spacing: 60u32, "
             "sqrt_price: { hi: 1u128, lo: 0u128 }, "
             "fee_protocol: 0u8, liquidity: 1000u128, "
             "fee_growth_global0_x_128: { hi: 0u128, lo: 0u128 }, "
             "fee_growth_global1_x_128: { hi: 0u128, lo: 0u128 }, "
             "max_liquidity_per_tick: 0u128, "
             "protocol_fees0: 0u128, protocol_fees1: 0u128, "
             "next_init_below: 3960i32, next_init_above: 4080i32 }")

POOL_TEXT = "{ token0: 1field, token1: 2field, fee: 3000u16, enabled: true }"

RECORD_TEXT = ("{ owner: aleo1me.private, amount: 2000000000u128.private, "
               "_nonce: 7group.public }")

# Vector account from test_blinding.py — next_blinded_identity derives real
# values.  Counter-0 identity pinned for shield_swap.aleo.
VIEW_KEY_SCALAR = "334926304971763782347498121479281870911723639068413954564748091722770623877scalar"
SIGNER = "aleo1rhgdu77hgyqd3xjj8ucu3jj9r2krwz6mnzyd80gncr5fxcwlh5rsvzp9px"
BLINDING_FACTOR_0 = "1084832000575072863530983109046262857691989153364570676666410266416291033880field"
BLINDED_ADDRESS_0 = "aleo15mstsvdtzqf5nw8rfzx8mrllwxt907amfpt8nx3p8cskj4wd3uxq4uywn9"

PROGRAM_ID = "shield_swap.aleo"

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

    def __init__(self, fn, pid=PROGRAM_ID):
        self._fn = fn
        self._pid = pid

    @property
    def outputs(self):
        return [[{"value": "999field"}], [{"value": "77field"}]]

    def decoded(self):
        return [dict(CHILD_TRANSITION),
                {"program": self._pid, "function": self._fn,
                 "outputs": [{"value": "77field"}]}]

    def transitions(self):
        return [_StubTransition("tok.aleo", "transfer", ["999field"]),
                _StubTransition(self._pid, self._fn, ["77field"])]


class _BoundCall:
    def __init__(self, recorder, fn, args, pid=PROGRAM_ID):
        self.program_id = pid
        self.function_name = fn
        self._recorder = recorder
        self._recorder.last_call = (fn, list(args))
        self._recorder.last_program = pid

    def simulate(self, account=None):
        return "simulated"

    def build_transaction(self, account=None, **kw):
        return _Tx(self.function_name, self.program_id)

    def delegate(self, account=None, **kw):
        self._recorder.delegated_fn = self.function_name
        self._recorder.delegated_program = self.program_id
        return {"transaction_id": "at1delegated"}


class _Functions:
    def __init__(self, recorder, pid):
        self._recorder = recorder
        self._pid = pid

    def __getattr__(self, fn):
        def call(*args):
            return _BoundCall(self._recorder, fn, args, self._pid)
        return call


class _Program:
    def __init__(self, recorder, mappings, pid):
        self._recorder = recorder
        self._mappings = mappings
        self.functions = _Functions(recorder, pid)
        self.source = _valid_source(pid)
        # Every token program the stub does not describe is backed by an
        # (empty) freeze list named after it, so the automatic compliance
        # path resolves without a registry; a test overrides ``imports``.
        default = [] if pid.endswith("freezelist.aleo") else [f"{pid.removesuffix('.aleo')}_freezelist.aleo"]
        self.imports = list(recorder.imports.get(pid, default))

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
        return _Tx(self._recorder.delegated_fn,
                   self._recorder.delegated_program or PROGRAM_ID)

    def get_program_mapping_value(self, program_id, mapping_name, key):
        """The node's mapping endpoint: no program handle involved.  The
        stub keys mappings by name only — except ``freeze_list_root``, which
        is keyed by program (``StubAleo.roots``) because several lists are
        read in one call; a program listed in ``missing_programs`` answers
        the node's 404."""
        self._recorder.mapping_reads.append((program_id, mapping_name, key))
        if program_id in self._recorder.missing_programs:
            from aleo import AleoNetworkError
            raise AleoNetworkError(f"GET /program/{program_id}/... returned 404", status=404)
        if mapping_name == "freeze_list_root":
            from aleo.compliance import EMPTY_TREE_ROOT
            root = self._recorder.roots.get(program_id, EMPTY_TREE_ROOT if program_id.endswith("freezelist.aleo") else None)
            return None if root is None else f"{root}field"
        return self._recorder._mappings.get(mapping_name, {}).get(key)

    def get_program_freeze_list(self, program_id):
        """The compliance endpoint: ``StubAleo.freeze_list_trees[program]`` (None = no list)."""
        self._recorder.tree_reads.append(program_id)
        return self._recorder.freeze_list_trees.get(program_id)


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
        self.last_program = None
        self.delegated_fn = None
        self.delegated_program = None
        self.submitted = []
        self.waited = []
        self.fetched_programs = []
        self.registered_programs = []
        self.mapping_reads = []
        self.tree_reads = []
        self.missing_programs = set()
        # Freeze lists: every list the stub knows is empty unless a test says
        # otherwise — ``roots[program]`` is the current root (int), and
        # ``freeze_list_trees[program]`` the published tree for a populated one.
        # (``roots[program] = None`` makes that list's root unreadable.)
        self.roots = {}
        self.freeze_list_trees = {}
        # program id -> import list, for the compliance resolver's freeze-list lookup.
        self.imports = {"test_usdcx_stablecoin.aleo": ["test_usdcx_freezelist.aleo"],
                        "test_usad_stablecoin.aleo": ["test_usad_freezelist.aleo"]}
        self._mappings = mappings or {}
        self.programs = _Programs(self, self._mappings)
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
