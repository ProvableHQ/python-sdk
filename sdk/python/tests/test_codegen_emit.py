"""Tests for aleo.codegen._emit — ABI type mapping and source emission."""
from aleo.codegen._emit import emit_struct, resolve_ty

SLOT_ABI = {
    "path": ["MiniSlot"],
    "fields": [
        {"name": "tick", "ty": {"Primitive": {"Int": "I32"}}},
        {"name": "sqrt_price", "ty": {"Primitive": {"UInt": "U128"}}},
        {"name": "pool", "ty": {"Primitive": "Field"}},
        {"name": "active", "ty": {"Primitive": "Boolean"}},
    ],
}

PREAMBLE = (
    "from dataclasses import dataclass\n"
    "from aleo.codegen.runtime import (parse_plaintext, fmt_int, fmt_bool,"
    " fmt_fieldlike, fmt_address)\n"
)


def test_resolve_uint():
    t = resolve_ty({"Primitive": {"UInt": "U128"}})
    assert t.annotation == "int"
    assert t.encode_expr("self.amount") == "fmt_int(self.amount, 'u128')"
    assert t.decode_expr("d['amount']") == "d['amount']"


def test_resolve_int_bool_field_address():
    assert resolve_ty({"Primitive": {"Int": "I32"}}).encode_expr("v") == "fmt_int(v, 'i32')"
    assert resolve_ty({"Primitive": "Boolean"}).encode_expr("v") == "fmt_bool(v)"
    assert resolve_ty({"Primitive": "Boolean"}).annotation == "bool"
    f = resolve_ty({"Primitive": "Field"})
    assert f.annotation == "str"
    assert f.encode_expr("v") == "fmt_fieldlike(v, 'field')"
    assert resolve_ty({"Primitive": "Address"}).encode_expr("v") == "fmt_address(v)"


def test_resolve_nested_struct():
    t = resolve_ty({"Struct": {"path": ["Slot"], "program": "x.aleo"}})
    assert t.annotation == "Slot"
    assert t.encode_expr("self.slot") == "self.slot.to_plaintext()"
    assert t.decode_expr("d['slot']") == "Slot.from_decoded(d['slot'])"


def test_emit_struct_roundtrip():
    ns: dict = {}
    exec(PREAMBLE + emit_struct(SLOT_ABI), ns)
    MiniSlot = ns["MiniSlot"]
    s = MiniSlot(tick=-4055, sqrt_price=22526123159817891330747538,
                 pool="4719field", active=True)
    text = s.to_plaintext()
    assert text == ("{ tick: -4055i32, sqrt_price: 22526123159817891330747538u128, "
                    "pool: 4719field, active: true }")
    assert MiniSlot.from_plaintext(text) == s
