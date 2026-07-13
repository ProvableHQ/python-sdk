"""Tests for aleo.codegen._emit — ABI type mapping and source emission."""
from aleo.codegen._emit import resolve_ty


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
