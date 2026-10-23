"""Read a program's record and function interface from its Aleo bytecode.

Inventory actions check a deployed program before spending anything: the
balance record exists, ``join`` takes two of it and returns one, ``split``
takes one plus an amount and returns two, and a join helper's ``join_N``
accepts the asset's record. The compiled :class:`~aleo.mainnet.Program` lists
function inputs but not outputs or record names, so this module reads the
declarations directly from the source text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

ParamKind = Literal["plaintext", "record", "external_record", "dynamic_record", "future", "dynamic_future"]

_PROGRAM = re.compile(r"^program\s+([a-z][a-z0-9_]*\.aleo);\s*$")
_IMPORT = re.compile(r"^import\s+([a-z][a-z0-9_]*\.aleo);\s*$")
_BLOCK = re.compile(r"^(function|closure|finalize|record|struct|mapping|constructor)\s+([A-Za-z_][A-Za-z0-9_]*):\s*$")
_PARAM = re.compile(r"^\s+(input|output)\s+r\d+\s+as\s+(.+?);\s*$")
_FIELD = re.compile(r"^\s+([A-Za-z_][A-Za-z0-9_]*)\s+as\s+(.+?)\.(constant|public|private);\s*$")


@dataclass(frozen=True)
class ProgramParam:
    """One function input or output.

    Attributes:
        kind: ``"record"`` for a local record, ``"external_record"`` for
            ``other.aleo/Name.record``, ``"dynamic_record"`` for
            ``dynamic.record``, ``"future"`` or ``"dynamic_future"`` for
            futures, otherwise ``"plaintext"``.
        type: The declared type without its visibility: a record name,
            ``program.aleo/Record``, ``identifier``, ``u64``, a struct name,
            or an array literal type.
        visibility: ``constant``, ``public``, or ``private`` for plaintext
            parameters; ``None`` for records and futures.
    """

    kind: ParamKind
    type: str
    visibility: str | None = None


@dataclass(frozen=True)
class ProgramFunction:
    """A function's declared inputs and outputs, in order."""

    name: str
    inputs: tuple[ProgramParam, ...] = ()
    outputs: tuple[ProgramParam, ...] = ()


@dataclass(frozen=True)
class RecordField:
    """One record entry with its declared type and visibility."""

    name: str
    type: str
    visibility: str


@dataclass(frozen=True)
class ProgramRecord:
    """A record declaration, including its ``owner`` entry."""

    name: str
    fields: tuple[RecordField, ...] = ()

    def field(self, name: str) -> RecordField | None:
        """Return the entry named *name*, or ``None``."""
        return next((item for item in self.fields if item.name == name), None)


@dataclass(frozen=True)
class ProgramInterface:
    """The parts of a program that inventory actions check.

    Attributes:
        id: The declared program id.
        imports: Program ids named by ``import`` lines, in order.
        functions: Functions by name with their inputs and outputs. Closures
            and finalize blocks are not included.
        records: Record declarations by name.
    """

    id: str
    imports: tuple[str, ...]
    functions: dict[str, ProgramFunction]
    records: dict[str, ProgramRecord]


def parse_param_type(declared: str) -> ProgramParam:
    """Classify a declared parameter type such as ``credits.record`` or ``u64.private``."""
    if declared.endswith(".record"):
        base = declared[: -len(".record")]
        if base == "dynamic":
            return ProgramParam("dynamic_record", base)
        return ProgramParam("external_record" if "/" in base else "record", base)
    if declared.endswith(".future"):
        base = declared[: -len(".future")]
        return ProgramParam("dynamic_future" if base == "dynamic" else "future", base)
    type_name, _, visibility = declared.rpartition(".")
    if not type_name or visibility not in ("constant", "public", "private"):
        return ProgramParam("plaintext", declared)
    return ProgramParam("plaintext", type_name, visibility)


def parse_program_interface(source: str) -> ProgramInterface:
    """Read the program id, imports, records, and function signatures from Aleo bytecode.

    Only declarations are read; instruction bodies are skipped. ``input`` and
    ``output`` lines inside closures and finalize blocks belong to those
    blocks and are not attributed to functions.

    Raises:
        ValueError: The text declares no ``program`` line.
    """
    program_id: str | None = None
    imports: list[str] = []
    functions: dict[str, ProgramFunction] = {}
    records: dict[str, ProgramRecord] = {}
    block: tuple[str, str] | None = None
    inputs: list[ProgramParam] = []
    outputs: list[ProgramParam] = []
    fields: list[RecordField] = []

    def close() -> None:
        if block is None:
            return
        kind, name = block
        if kind == "function":
            functions[name] = ProgramFunction(name, tuple(inputs), tuple(outputs))
        elif kind == "record":
            records[name] = ProgramRecord(name, tuple(fields))

    for raw in source.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        header = _BLOCK.match(line)
        if header:
            close()
            block = (header.group(1), header.group(2))
            inputs, outputs, fields = [], [], []
            continue
        if not line[0].isspace():
            close()
            block = None
            program = _PROGRAM.match(line)
            if program:
                program_id = program.group(1)
                continue
            imported = _IMPORT.match(line)
            if imported:
                imports.append(imported.group(1))
            continue
        if block is None:
            continue
        kind, _ = block
        if kind == "function":
            param = _PARAM.match(line)
            if param:
                (inputs if param.group(1) == "input" else outputs).append(parse_param_type(param.group(2)))
        elif kind == "record":
            entry = _FIELD.match(line)
            if entry:
                fields.append(RecordField(entry.group(1), entry.group(2), entry.group(3)))
    close()
    if program_id is None:
        raise ValueError("Program source declares no program id")
    return ProgramInterface(program_id, tuple(imports), functions, records)


__all__ = [
    "ParamKind", "ProgramFunction", "ProgramInterface", "ProgramParam", "ProgramRecord", "RecordField",
    "parse_param_type", "parse_program_interface",
]
