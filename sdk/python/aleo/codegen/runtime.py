"""Runtime helpers imported by aleo.codegen-generated modules.

Pure Python, no PyO3 — generated modules must import cheaply and work in any
environment where the ``aleo`` package is installed.

``parse_plaintext`` parses an Aleo plaintext literal into Python values:
structs/records become dicts (record visibility suffixes ``.private`` /
``.public`` are stripped; ``_nonce`` is kept as a plain entry), suffixed
integers become ``int``, booleans become ``bool``, arrays become lists, and
field/group/scalar/address literals stay verbatim strings.

The ``fmt_*`` helpers are the inverse direction: they format Python values as
Aleo literals, validating range and shape so a bad value fails at encode time
with a clear message instead of on-chain.
"""
from __future__ import annotations

import re
from typing import Any

_INT_RE = re.compile(r"^(-?\d+)(u8|u16|u32|u64|u128|i8|i16|i32|i64|i128)$")
_MODE_RE = re.compile(r"\.(private|public|constant)$")
_ATOM_RE = re.compile(r"[^,}\]]+")


def parse_plaintext(text: str) -> Any:
    """Parse an Aleo plaintext literal into Python values."""
    value, rest = _parse_value(text.strip())
    if rest.strip():
        raise ValueError(f"Trailing content after plaintext value: {rest!r}")
    return value


def _parse_value(s: str) -> tuple[Any, str]:
    s = s.lstrip()
    if s.startswith("{"):
        return _parse_struct(s)
    if s.startswith("["):
        return _parse_array(s)
    return _parse_atom(s)


def _parse_struct(s: str) -> tuple[dict[str, Any], str]:
    s = s[1:].lstrip()  # consume "{"
    out: dict[str, Any] = {}
    while not s.startswith("}"):
        if not s:
            raise ValueError("Unterminated struct in plaintext")
        name, sep, s = s.partition(":")
        if not sep:
            raise ValueError(f"Expected 'name:' in struct, got {name!r}")
        value, s = _parse_value(s)
        out[name.strip()] = value
        s = s.lstrip()
        if s.startswith(","):
            s = s[1:].lstrip()
    return out, s[1:]


def _parse_array(s: str) -> tuple[list[Any], str]:
    s = s[1:].lstrip()  # consume "["
    out: list[Any] = []
    while not s.startswith("]"):
        if not s:
            raise ValueError("Unterminated array in plaintext")
        value, s = _parse_value(s)
        out.append(value)
        s = s.lstrip()
        if s.startswith(","):
            s = s[1:].lstrip()
    return out, s[1:]


def _parse_atom(s: str) -> tuple[Any, str]:
    m = _ATOM_RE.match(s)
    if m is None:
        raise ValueError(f"Expected a plaintext atom, got {s[:20]!r}")
    token = _MODE_RE.sub("", m.group(0).strip())
    rest = s[m.end():]
    if token == "true":
        return True, rest
    if token == "false":
        return False, rest
    im = _INT_RE.match(token)
    if im:
        return int(im.group(1)), rest
    # field/group/scalar literals, addresses, signatures — verbatim strings
    return token, rest


# ── Literal formatters (encode direction, used by generated to_plaintext) ────

_INT_BOUNDS = {
    "u8": (0, 2**8 - 1), "u16": (0, 2**16 - 1), "u32": (0, 2**32 - 1),
    "u64": (0, 2**64 - 1), "u128": (0, 2**128 - 1),
    "i8": (-(2**7), 2**7 - 1), "i16": (-(2**15), 2**15 - 1),
    "i32": (-(2**31), 2**31 - 1), "i64": (-(2**63), 2**63 - 1),
    "i128": (-(2**127), 2**127 - 1),
}


def fmt_int(v: int, suffix: str) -> str:
    """Format an int as a suffixed Aleo integer literal, validating range."""
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"Expected int for {suffix}, got {type(v).__name__}")
    lo, hi = _INT_BOUNDS[suffix]
    if not lo <= v <= hi:
        raise ValueError(f"{v} out of range for {suffix} [{lo}, {hi}]")
    return f"{v}{suffix}"


def fmt_bool(v: bool) -> str:
    """Format a bool as an Aleo boolean literal."""
    if not isinstance(v, bool):
        raise ValueError(f"Expected bool, got {type(v).__name__}")
    return "true" if v else "false"


def fmt_fieldlike(v: int | str, suffix: str) -> str:
    """Format an int or pre-suffixed literal as a field/group/scalar literal."""
    if isinstance(v, int) and not isinstance(v, bool):
        return f"{v}{suffix}"
    if isinstance(v, str) and re.fullmatch(rf"\d+{suffix}", v):
        return v
    raise ValueError(f"Expected int or '<digits>{suffix}' literal, got {v!r}")


def fmt_address(v: str) -> str:
    """Validate an aleo1… address literal (passes through unchanged)."""
    if not (isinstance(v, str) and v.startswith("aleo1")):
        raise ValueError(f"Expected an aleo1… address literal, got {v!r}")
    return v
