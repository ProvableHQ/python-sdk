"""ABI JSON → Python source emitter.

Build-time only; the emitted code imports :mod:`aleo.codegen.runtime` for
parsing and formatting.  The ABI shape this consumes is the ``aleo-abi``
output: struct = ``{path: [Name], fields: [{name, ty}]}``, record fields add
``mode``, mapping = ``{name, key: ty, value: ty}``, and ``ty`` is either
``{"Primitive": ...}`` or ``{"Struct": {"path": [...], "program": ...}}``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class PyType:
    """How one ABI type appears in emitted Python.

    ``annotation`` is the type annotation; ``encode_expr``/``decode_expr``
    map a value expression to the encoding/decoding expression emitted into
    ``to_plaintext``/``from_decoded`` bodies.
    """

    annotation: str
    encode_expr: Callable[[str], str]
    decode_expr: Callable[[str], str]


def resolve_ty(ty: Any) -> PyType:
    """Map an ABI ``ty`` tree to its emitted-Python representation."""
    if isinstance(ty, dict) and "Primitive" in ty:
        prim = ty["Primitive"]
        if isinstance(prim, dict):
            width = prim.get("UInt") or prim.get("Int")
            if width is None:
                raise ValueError(f"Unsupported primitive: {prim!r}")
            suffix = width.lower()
            return PyType("int", lambda e, s=suffix: f"fmt_int({e}, '{s}')", lambda e: e)
        if prim == "Boolean":
            return PyType("bool", lambda e: f"fmt_bool({e})", lambda e: e)
        if prim == "Address":
            return PyType("str", lambda e: f"fmt_address({e})", lambda e: e)
        if prim in ("Field", "Group", "Scalar"):
            suffix = prim.lower()
            return PyType("str", lambda e, s=suffix: f"fmt_fieldlike({e}, '{s}')", lambda e: e)
        raise ValueError(f"Unsupported primitive: {prim!r}")
    if isinstance(ty, dict) and "Struct" in ty:
        name = ty["Struct"]["path"][-1]
        return PyType(
            name,
            lambda e: f"{e}.to_plaintext()",
            lambda e, n=name: f"{n}.from_decoded({e})",
        )
    raise ValueError(f"Unsupported ABI type: {ty!r}")


def emit_struct(struct: dict[str, Any]) -> str:
    """Emit one struct as a frozen dataclass with encode/decode methods."""
    name = struct["path"][-1]
    fields = struct["fields"]
    lines: list[str] = ["@dataclass(frozen=True)", f"class {name}:"]
    for f in fields:
        lines.append(f"    {f['name']}: {resolve_ty(f['ty']).annotation}")

    # to_plaintext — emitted as a parts list + join (readable generated code).
    lines += ["", "    def to_plaintext(self) -> str:", "        parts = ["]
    for f in fields:
        enc = resolve_ty(f["ty"]).encode_expr("self." + f["name"])
        lines.append(f"            \"{f['name']}: \" + {enc},")
    lines += [
        "        ]",
        "        return \"{ \" + \", \".join(parts) + \" }\"",
    ]

    # from_decoded / from_plaintext.  Subscript expressions are precomputed
    # outside the f-string (no backslashes in f-string expressions on 3.10).
    kwarg_parts: list[str] = []
    for f in fields:
        subscript = "d['" + f["name"] + "']"
        kwarg_parts.append(f"{f['name']}={resolve_ty(f['ty']).decode_expr(subscript)}")
    kwargs = ", ".join(kwarg_parts)
    lines += [
        "",
        "    @classmethod",
        "    def from_decoded(cls, d: dict) -> \"" + name + "\":",
        f"        return cls({kwargs})",
        "",
        "    @classmethod",
        "    def from_plaintext(cls, text: str) -> \"" + name + "\":",
        "        return cls.from_decoded(parse_plaintext(text))",
        "",
    ]
    return "\n".join(lines) + "\n"
