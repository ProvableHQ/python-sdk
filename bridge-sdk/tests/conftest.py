"""Hermetic stand-ins for the aleo facade. Records every call so tests assert on exact inputs."""
from __future__ import annotations

import importlib
import json
from typing import Any

import pytest

SIGNER = "aleo1rhgdu77hgyqd3xjj8ucu3jj9r2krwz6mnzyd80gncr5fxcwlh5rsvzp9px"
IGP = "aleo194tz0jmyq8rd9htvnqppqw4jqerk2p2zd8plzn3sxl06wcgsm5pq9fka74"
IGP_KEY_ETH = f"{{ igp: {IGP}, destination: 1u32 }}"
IGP_KEY_SOL = f"{{ igp: {IGP}, destination: 1399811149u32 }}"
ETH_GAS_CONFIG = "{\n  gas_overhead: 159337u128,\n  exchange_rate: 402u128,\n  gas_price: 1000000000u128\n}"
SOL_GAS_CONFIG = "{ gas_overhead: 200000u128, exchange_rate: 1000u128, gas_price: 50000000u128 }"
USDCX_RECORD = f"{{ owner: {SIGNER}.private, amount: 5000000u128.private, _nonce: 7group.public }}"
USDCX_RECORD_SMALL = f"{{ owner: {SIGNER}.private, amount: 100u128.private, _nonce: 8group.public }}"
DELIVERED_KEY = "{ id: [262854447642257427123071959211115528903u128, 102980212169860384794748804418278302317u128] }"
NULLIFIED_NONCE = bytes.fromhex("aa" * 32)
CHILD = {"program": "usdcx_stablecoin.aleo", "function": "transfer_private_to_public", "outputs": [{"value": "1field"}]}


def _source(program_id: str) -> str:
    return f"program {program_id};\nfunction main:\n    input r0 as u64.public;\n    output r0 as u64.public;\n"


class FakeAccount:
    def __init__(self, address: str = SIGNER) -> None:
        self.address = address
        self.private_key = "APrivateKey1zkpFake"


class FakeMapping:
    def __init__(self, values: dict[str, Any]) -> None:
        self._values = values

    def get(self, key: Any) -> Any:
        return self._values.get(str(key))          # None ≙ absent/null mapping entry


class FakeTransition:
    def __init__(self, program: str, function: str, outputs: list) -> None:
        self.program_id, self.function_name, self._outputs = program, function, outputs

    def outputs(self) -> list:
        return list(self._outputs)


def _tx_json(tx_id: str, program: str, function: str, outputs: list) -> dict:
    return {"id": tx_id, "type": "execute", "execution": {"transitions": [
        dict(CHILD), {"program": program, "function": function, "outputs": outputs}]}}


class FakeTx:
    def __init__(self, tx_id: str, program: str, function: str, outputs: list) -> None:
        self.id = tx_id
        self._json = _tx_json(tx_id, program, function, outputs)
        self.raw = json.dumps(self._json)          # str(raw) is what the node accepts

    def transitions(self) -> list[FakeTransition]:
        return [FakeTransition(t["program"], t["function"], t["outputs"]) for t in self._json["execution"]["transitions"]]

    def decoded(self) -> list[dict]:
        return [{"program": t.program_id, "function": t.function_name, "outputs": t.outputs()} for t in self.transitions()]


class FakeBound:
    def __init__(self, aleo: "FakeAleo", program_id: str, function_name: str, args: tuple) -> None:
        self._aleo, self.program_id, self.function_name = aleo, program_id, function_name
        self.args = [str(a) for a in args]
        aleo.calls.append((program_id, function_name, list(self.args)))

    def simulate(self, account: Any = None) -> str:
        self._aleo.simulated.append((self.program_id, self.function_name))
        return "simulated"

    def build_transaction(self, account: Any = None, **fee: Any) -> FakeTx:
        self._aleo.fee_kwargs.append(dict(fee))
        return FakeTx("at1built", self.program_id, self.function_name, [{"value": "77field"}])

    def delegate(self, account: Any = None, *, broadcast: bool = True, **fee: Any) -> dict:
        self._aleo.delegated.append({"program": self.program_id, "function": self.function_name, "broadcast": broadcast, **fee})
        if self._aleo.delegate_returns_id_only:
            return {"transaction_id": "at1delegated"}
        return {"transaction": _tx_json("at1delegated", self.program_id, self.function_name, [{"value": "77field"}])}


class FakeFunctions:
    def __init__(self, aleo: "FakeAleo", program_id: str) -> None:
        self._aleo, self._program_id = aleo, program_id

    def __getitem__(self, name: str):
        return lambda *args: FakeBound(self._aleo, self._program_id, name, args)

    __getattr__ = __getitem__


class FakeProgram:
    def __init__(self, aleo: "FakeAleo", program_id: str) -> None:
        self.id = program_id
        self.source = _source(program_id)
        self.imports = list(aleo.imports.get(program_id, []))
        self.raw = ("raw", program_id)
        self.functions = FakeFunctions(aleo, program_id)
        self._mappings = aleo.mappings.get(program_id, {})

    def mapping(self, name: str) -> FakeMapping:
        return FakeMapping(self._mappings.get(name, {}))

    def mappings(self) -> list[str]:
        return sorted(self._mappings)


class FakePrograms:
    def __init__(self, aleo: "FakeAleo") -> None:
        self._aleo = aleo

    def get(self, program_id: str) -> FakeProgram:
        self._aleo.fetched.append(program_id)
        return FakeProgram(self._aleo, program_id)


class FakeRecords:
    def __init__(self, aleo: "FakeAleo") -> None:
        self._aleo = aleo

    def find(self, account: Any = None, *, program: str | None = None, record: str | None = None,
             unspent: bool = True, **_: Any) -> list[dict]:
        self._aleo.record_queries.append({"program": program, "record": record, "unspent": unspent})
        return [dict(r) for r in self._aleo.record_rows if program is None or r.get("program") == program]


class FakeNetwork:
    def __init__(self, aleo: "FakeAleo") -> None:
        self._aleo = aleo

    def submit_transaction(self, transaction: Any) -> str:
        self._aleo.submitted.append(transaction)
        if self._aleo.duplicate_on_submit:
            raise RuntimeError("Transaction 'at1prepared' already exists in the ledger")
        if isinstance(transaction, str):
            return str(json.loads(transaction)["id"])
        return str(getattr(transaction, "id", "at1built"))

    def wait_for_transaction(self, tx_id: str, *, timeout: float = 45.0, poll_interval: float = 2.0) -> dict:
        self._aleo.waited.append((tx_id, timeout))
        return {"status": "accepted"}

    def get_transaction_object(self, tx_id: str) -> FakeTx:
        return FakeTx(tx_id, "hyp_warp_token_wbtc_v2.aleo", "transfer_remote", [{"value": "99field"}])


class FakeProcess:
    def __init__(self, aleo: "FakeAleo") -> None:
        self._aleo = aleo

    def contains_program(self, program_id: Any) -> bool:
        return str(program_id) in self._aleo.registered

    def add_program(self, program: Any) -> None:
        self._aleo.registered.append(str(program[1]) if isinstance(program, tuple) else str(program))


class FakeAleo:
    """Facade stand-in: mappings keyed program → mapping → key; records; network; process; recorders."""

    def __init__(self, mappings: dict | None = None, records: list[dict] | None = None,
                 network_name: str = "mainnet", default_account: Any = None, imports: dict | None = None) -> None:
        self.network_name = network_name
        self.default_account = FakeAccount() if default_account is None else default_account
        self.mappings = mappings or {}
        # ``records`` is the module (aleo.records.find); the rows it returns live in ``record_rows``.
        self.record_rows = records if records is not None else [{"program": "usdcx_stablecoin.aleo", "record_plaintext": USDCX_RECORD}]
        self.imports = imports or {}
        self.calls: list = []
        self.simulated: list = []
        self.delegated: list = []
        self.fee_kwargs: list = []
        self.submitted: list = []
        self.waited: list = []
        self.fetched: list = []
        self.registered: list = []
        self.record_queries: list = []
        self.duplicate_on_submit = False
        self.delegate_returns_id_only = False
        self.programs = FakePrograms(self)
        self.records = FakeRecords(self)
        self.record_provider = self.records
        self.network = FakeNetwork(self)
        self.process = FakeProcess(self)


class FakeNetModule:
    """Stands in for aleo.<network> inside AleoCall's import registration (no real parsing)."""

    class Program:
        @staticmethod
        def from_source(source: str):
            return ("raw", source.split(";")[0].removeprefix("program "))

    class ProgramID:
        @staticmethod
        def from_string(value: str) -> str:
            return value


def default_mappings() -> dict:
    return {
        "hyp_hook_manager.aleo": {"destination_gas_configs": {IGP_KEY_ETH: ETH_GAS_CONFIG, IGP_KEY_SOL: SOL_GAS_CONFIG}},
        "hyp_mailbox.aleo": {"deliveries": {DELIVERED_KEY: "{ block_height: 1u32 }"}},
        "usdcx_bridge_v2.aleo": {"nullifier": {"[" + ",".join(f"{b}u8" for b in NULLIFIED_NONCE) + "]": "true"}},
        "credits.aleo": {"account": {SIGNER: "2392443u64"}},
        "usdcx_stablecoin.aleo": {"balances": {SIGNER: "1000000u128"}, "freeze_list": {}, "freeze_list_last_index": {}},
        "arc20_wbtc.aleo": {"balances": {SIGNER: "10000u128"}},
        "arc20_eth.aleo": {"balances": {}},
    }


@pytest.fixture
def fake_aleo(monkeypatch) -> FakeAleo:
    monkeypatch.setattr("aleo_bridge._calls._network_module", lambda aleo: FakeNetModule)
    return FakeAleo(mappings=default_mappings())


class _BridgeStub:
    """The five Bridge seams protocol modules use, until Task 11 wires the real Bridge into this fixture."""

    def __init__(self, aleo: FakeAleo) -> None:
        from aleo_bridge._calls import AleoCall
        from aleo_bridge.registry import DEFAULT_REGISTRY

        self.aleo = aleo
        self.registry = DEFAULT_REGISTRY
        self.environment = self.network = aleo.network_name
        self._AleoCall = AleoCall
        self._programs: dict = {}
        for attr, module, cls in (("hyperlane", "hyperlane", "HyperlaneModule"), ("xreserve", "xreserve", "XReserveModule"),
                                  ("freezelist", "freezelist", "FreezeList"), ("privacy", "privacy", "PrivacyModule")):
            try:
                setattr(self, attr, getattr(importlib.import_module(f"aleo_bridge.{module}"), cls)(self))
            except ImportError:
                setattr(self, attr, None)

    def aleo_address(self) -> str:
        return str(self.aleo.default_account.address)

    def program(self, program_id: str):
        if program_id not in self._programs:
            self._programs[program_id] = self.aleo.programs.get(program_id)
        return self._programs[program_id]

    def mapping_value(self, program_id: str, mapping: str, key: str) -> str | None:
        value = self.program(program_id).mapping(mapping).get(key)
        if value is None:
            return None
        text = str(value).strip().strip('"')
        return None if text in ("", "null", "None") else text

    def _call(self, program_id: str, function: str, inputs: list[str], build_result):
        program = self.program(program_id)
        bound = program.functions[function](*inputs)
        return self._AleoCall(self.aleo, bound, build_result, imports={program_id: program.source})


@pytest.fixture
def bridge(fake_aleo) -> Any:
    return _BridgeStub(fake_aleo)
