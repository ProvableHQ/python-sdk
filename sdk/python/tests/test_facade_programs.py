"""Tests for F4 facade: aleo.programs module.

All tests are mocked/offline.  ``get_program`` is mocked via the ``responses``
library to return a real vendored program source (``Program.credits().source``
for credits.aleo, or a small simple.aleo source), so no live network is needed.
"""
from __future__ import annotations

import pytest
import responses as resp_lib

from aleo import Aleo, HTTPProvider
from aleo.mainnet import Program as RawProgram
from aleo.facade.programs import (
    ProgramsModule,
    Program,
    ProgramFunctions,
    PreparedCall,
    Mapping,
)
from aleo.facade.errors import ProgramNotFound

# ---------------------------------------------------------------------------
# Constants / fixtures
# ---------------------------------------------------------------------------

BASE = "https://api.provable.com/v2"
NET = "mainnet"
HOST = f"{BASE}/{NET}"

CREDITS_SOURCE = RawProgram.credits().source

SIMPLE_SOURCE = """program simple.aleo;
function main:
    input r0 as u32.public;
    output r0 as u32.public;
"""

ADDR = "aleo1lqmly7ez2k48ajf5hs92ulphaqr05qm4n8qwzj8v0yprmasgpqgsez59gg"


def make_client(network: str = "mainnet") -> Aleo:
    return Aleo(HTTPProvider(BASE, network=network))


def _mock_credits() -> None:
    resp_lib.add(resp_lib.GET, f"{HOST}/program/credits.aleo", json=CREDITS_SOURCE)


# ---------------------------------------------------------------------------
# Module attachment
# ---------------------------------------------------------------------------


def test_programs_module_attached() -> None:
    a = make_client()
    assert isinstance(a.programs, ProgramsModule)


def test_programs_module_same_instance() -> None:
    a = make_client()
    assert a.programs is a.programs


def test_programs_module_repr() -> None:
    a = make_client()
    r = repr(a.programs)
    assert "ProgramsModule" in r
    assert "mainnet" in r


# ---------------------------------------------------------------------------
# get() → bound Program
# ---------------------------------------------------------------------------


@resp_lib.activate
def test_get_returns_bound_program() -> None:
    _mock_credits()
    a = make_client()
    p = a.programs.get("credits.aleo")
    assert isinstance(p, Program)
    assert p.id == "credits.aleo"
    assert "program credits.aleo;" in p.source
    assert isinstance(p.raw, type(RawProgram.credits()))
    assert p.imports == []


@resp_lib.activate
def test_get_missing_raises_program_not_found() -> None:
    resp_lib.add(resp_lib.GET, f"{HOST}/program/nope.aleo", status=404)
    a = make_client()
    with pytest.raises(ProgramNotFound) as exc_info:
        a.programs.get("nope.aleo")
    assert exc_info.value.program_id == "nope.aleo"


# ---------------------------------------------------------------------------
# Dynamic functions namespace
# ---------------------------------------------------------------------------


@resp_lib.activate
def test_functions_dir_lists_transitions() -> None:
    _mock_credits()
    a = make_client()
    p = a.programs.get("credits.aleo")
    names = dir(p.functions)
    assert "transfer_public" in names
    assert "transfer_private" in names


@resp_lib.activate
def test_functions_contains() -> None:
    _mock_credits()
    a = make_client()
    p = a.programs.get("credits.aleo")
    assert "transfer_public" in p.functions
    assert "does_not_exist" not in p.functions


@resp_lib.activate
def test_functions_iter() -> None:
    _mock_credits()
    a = make_client()
    p = a.programs.get("credits.aleo")
    fns = list(p.functions)
    assert "transfer_public" in fns
    assert "transfer_private" in fns
    assert len(fns) == len(p.functions)


@resp_lib.activate
def test_functions_attribute_access() -> None:
    _mock_credits()
    a = make_client()
    p = a.programs.get("credits.aleo")
    caller = p.functions.transfer_public
    assert "transfer_public" in caller.signature
    assert "address" in caller.signature
    assert "u64" in caller.signature


@resp_lib.activate
def test_functions_item_access() -> None:
    _mock_credits()
    a = make_client()
    p = a.programs.get("credits.aleo")
    caller = p.functions["transfer_public"]
    assert caller.function_name == "transfer_public"


@resp_lib.activate
def test_functions_unknown_attribute_names_available() -> None:
    _mock_credits()
    a = make_client()
    p = a.programs.get("credits.aleo")
    with pytest.raises(AttributeError) as exc_info:
        _ = p.functions.no_such_fn
    msg = str(exc_info.value)
    assert "no_such_fn" in msg
    assert "transfer_public" in msg  # lists available functions


@resp_lib.activate
def test_functions_unknown_item_names_available() -> None:
    _mock_credits()
    a = make_client()
    p = a.programs.get("credits.aleo")
    with pytest.raises(KeyError) as exc_info:
        _ = p.functions["no_such_fn"]
    msg = str(exc_info.value)
    assert "no_such_fn" in msg
    assert "transfer_public" in msg


# ---------------------------------------------------------------------------
# PreparedCall — signature + coercion
# ---------------------------------------------------------------------------


@resp_lib.activate
def test_prepared_call_signature_matches_inputs() -> None:
    _mock_credits()
    a = make_client()
    p = a.programs.get("credits.aleo")
    call = p.functions.transfer_public(ADDR, 10)
    assert isinstance(call, PreparedCall)
    # Declared inputs match get_function_inputs exactly.
    assert call.inputs == p.raw.get_function_inputs("transfer_public")
    assert call.signature == "transfer_public(address, u64)"
    assert call.program_id == "credits.aleo"
    assert call.function_name == "transfer_public"


@resp_lib.activate
def test_coercion_bare_int_gets_suffix() -> None:
    _mock_credits()
    a = make_client()
    p = a.programs.get("credits.aleo")
    call = p.functions.transfer_public(ADDR, 10)
    assert call.args == [ADDR, "10u64"]


@resp_lib.activate
def test_coercion_preformatted_passes_through() -> None:
    _mock_credits()
    a = make_client()
    p = a.programs.get("credits.aleo")
    call = p.functions.transfer_public(ADDR, "10u64")
    assert call.args == [ADDR, "10u64"]


@resp_lib.activate
def test_coercion_bad_type_cites_expected() -> None:
    _mock_credits()
    a = make_client()
    p = a.programs.get("credits.aleo")
    with pytest.raises(ValueError) as exc_info:
        p.functions.transfer_public(ADDR, 1.5)  # float — uncoercible to u64
    msg = str(exc_info.value)
    assert "u64" in msg
    assert "transfer_public(address, u64)" in msg


@resp_lib.activate
def test_coercion_wrong_arity() -> None:
    _mock_credits()
    a = make_client()
    p = a.programs.get("credits.aleo")
    with pytest.raises(ValueError) as exc_info:
        p.functions.transfer_public(ADDR)  # missing amount
    msg = str(exc_info.value)
    assert "2 argument" in msg
    assert "transfer_public(address, u64)" in msg


@resp_lib.activate
def test_coercion_typed_wrapper_stringified() -> None:
    _mock_credits()
    a = make_client()
    from aleo.mainnet import Address

    addr_obj = Address.from_string(ADDR)
    p = a.programs.get("credits.aleo")
    call = p.functions.transfer_public(addr_obj, 10)
    assert call.args[0] == ADDR


def test_coercion_boolean() -> None:
    # Build a tiny program with a boolean input to exercise the bool rule.
    src = """program boolprog.aleo;
function flip:
    input r0 as boolean.public;
    output r0 as boolean.public;
"""
    raw = RawProgram.from_source(src)
    a = make_client()
    p = Program(a, raw)
    call = p.functions.flip(True)
    assert call.args == ["true"]
    call2 = p.functions.flip(False)
    assert call2.args == ["false"]


# ---------------------------------------------------------------------------
# Mappings
# ---------------------------------------------------------------------------


@resp_lib.activate
def test_mapping_get_parses_value() -> None:
    _mock_credits()
    resp_lib.add(
        resp_lib.GET,
        f"{HOST}/program/credits.aleo/mapping/account/{ADDR}",
        json="500u64",
    )
    a = make_client()
    p = a.programs.get("credits.aleo")
    m = p.mapping("account")
    assert isinstance(m, Mapping)
    assert m.get(ADDR) == "500u64"


@resp_lib.activate
def test_mapping_names() -> None:
    _mock_credits()
    resp_lib.add(
        resp_lib.GET,
        f"{HOST}/program/credits.aleo/mappings",
        json=["account", "committee"],
    )
    a = make_client()
    p = a.programs.get("credits.aleo")
    assert "account" in p.mapping("account").names()


@resp_lib.activate
def test_program_mappings_list() -> None:
    _mock_credits()
    a = make_client()
    p = a.programs.get("credits.aleo")
    names = p.mappings()
    assert "account" in names
    assert "committee" in names


# ---------------------------------------------------------------------------
# ABI — the 3 entry points
# ---------------------------------------------------------------------------


def test_generate_abi_local_source() -> None:
    pytest.importorskip("aleo_abi")
    a = make_client()
    d = a.generate_abi(SIMPLE_SOURCE)
    assert isinstance(d, dict)
    assert d["program"] == "simple.aleo"


@resp_lib.activate
def test_generate_abi_local_from_facade_program() -> None:
    pytest.importorskip("aleo_abi")
    _mock_credits()
    a = make_client()
    p = a.programs.get("credits.aleo")
    d = a.generate_abi(p)  # facade Program → uses .raw
    assert d["program"] == "credits.aleo"


@resp_lib.activate
def test_programs_abi_web_path() -> None:
    pytest.importorskip("aleo_abi")
    _mock_credits()
    a = make_client()
    d = a.programs.abi("credits.aleo")
    assert isinstance(d, dict)
    assert d["program"] == "credits.aleo"


@resp_lib.activate
def test_program_abi_object_path() -> None:
    pytest.importorskip("aleo_abi")
    _mock_credits()
    a = make_client()
    p = a.programs.get("credits.aleo")
    d = p.abi()
    assert isinstance(d, dict)
    assert d["program"] == "credits.aleo"


@resp_lib.activate
def test_programs_abi_missing_raises_program_not_found() -> None:
    pytest.importorskip("aleo_abi")
    resp_lib.add(resp_lib.GET, f"{HOST}/program/nope.aleo", status=404)
    a = make_client()
    with pytest.raises(ProgramNotFound):
        a.programs.abi("nope.aleo")


# ---------------------------------------------------------------------------
# .functions works WITHOUT aleo_abi (built from get_function_inputs)
# ---------------------------------------------------------------------------


def test_functions_do_not_depend_on_aleo_abi() -> None:
    """The dynamic function namespace is built from get_function_inputs, never
    from aleo_abi — so it works even if aleo_abi were absent."""
    raw = RawProgram.from_source(SIMPLE_SOURCE)
    a = make_client()
    p = Program(a, raw)
    assert "main" in p.functions
    call = p.functions.main(7)
    assert call.args == ["7u32"]
