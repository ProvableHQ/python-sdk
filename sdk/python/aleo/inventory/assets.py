"""Resolve an asset against its deployed program and decode scanner records.

Before any record is spent, the asset's program is checked for the native
``join``/``split`` interface and the balance record and amount field are
fixed. Scanner records are then decoded against that resolution: a record
from another program, another owner, a compliance or registry record, or a
recipient-bound wrapper record is never treated as spendable inventory.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from .interface import ProgramFunction, ProgramInterface, ProgramParam, parse_program_interface
from .types import (
    AmountType, InventoryError, InventoryRecord, RecordAsset, ResolvedRecordAsset, Standard,
)

CREDITS_PROGRAM = "credits.aleo"
#: Microcredits a native ``credits.aleo/split`` removes from its outputs, separate from the fee.
CREDITS_SPLIT_DEDUCTION = 10_000
#: Record entries that mark a compliance, registry, or non-fungible record.
EXCLUDED_RECORD_FIELDS = ("token_id", "sender", "recipient")

_PROGRAM_ID = re.compile(r"^[a-z][a-z0-9_]*\.aleo$")
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_UNSIGNED = re.compile(r"^u(8|16|32|64|128)$")
_NONCE = re.compile(r"(?<![A-Za-z0-9_])_nonce:\s*(\d+group)")
_OWNER = re.compile(r"(?<![A-Za-z0-9_])owner:\s*([a-z0-9]+)")
_AMOUNT = re.compile(r"^(\d+)(u8|u16|u32|u64|u128)$")


def split_deduction(asset: RecordAsset) -> int:
    """Credits removed by one split of this asset, independent of network fees."""
    return CREDITS_SPLIT_DEDUCTION if asset.program == CREDITS_PROGRAM else 0


def amount_width(asset: RecordAsset) -> AmountType:
    """The declared width of the amount field, with the native defaults for unresolved assets."""
    if asset.amount_type is not None:
        return asset.amount_type
    return "u64" if asset.program == CREDITS_PROGRAM or asset.amount_field == "microcredits" else "u128"


def record_name(asset: RecordAsset) -> str:
    """The configured record name, or the native default (``credits`` / ``Token``)."""
    return asset.record_name or ("credits" if asset.program == CREDITS_PROGRAM else "Token")


def max_amount(asset: RecordAsset) -> int:
    """The largest amount the asset's declared width can hold."""
    return (1 << int(amount_width(asset)[1:])) - 1


def scope_for(chain_id: str, address: str) -> str:
    """The reservation namespace for one chain and account.

    Serialised exactly as Veil's ``JSON.stringify([chainId, address])`` so a
    journal can be shared with other Veil clients.
    """
    return json.dumps([chain_id, address], separators=(",", ":"))


@dataclass(frozen=True)
class AssetCapabilities:
    """What was learned about an asset's record when it was resolved.

    Attributes:
        asset: The resolved record name, amount field, and width.
        recipient_bound: The record declares ``recipient_bound``; bound
            records are excluded from inventory.
        excluded: The record declares a compliance or registry field, so no
            record of this type is spendable inventory.
    """

    asset: ResolvedRecordAsset
    recipient_bound: bool
    excluded: bool


def _is_record(param: ProgramParam, name: str) -> bool:
    return param.kind == "record" and param.type == name


def _detect_standard(interface: ProgramInterface) -> Standard | None:
    """Detect ARC-20 or ARC-22 from the token interface, ignoring missing view functions."""
    required = {"join", "split", "transfer_private", "transfer_public"}
    if not required.issubset(interface.functions):
        return None
    return "arc22" if "ComplianceRecord" in interface.records else "arc20"


def validate_asset(asset: RecordAsset, source: str | None) -> AssetCapabilities:
    """Resolve the asset's record defaults and check the program's join/split interface.

    Reads nothing from the network: pass the deployed program source (or
    ``None`` for ``credits.aleo``, whose interface is fixed).

    Raises:
        InventoryError: The asset names are malformed, the hint contradicts
            the program, the program is missing, or its ``join``/``split``
            does not take and return the selected record.
    """
    if not _PROGRAM_ID.match(asset.program):
        raise InventoryError("Invalid asset program")
    for name in (asset.record_name, asset.amount_field):
        if name is not None and not _IDENTIFIER.match(name):
            raise InventoryError("Invalid record or amount field name")
    if asset.standard is not None and asset.standard not in ("credits", "arc20", "arc22"):
        raise InventoryError("Unsupported token standard")
    if asset.standard == "credits" and asset.program != CREDITS_PROGRAM:
        raise InventoryError("Native credits require credits.aleo")
    if asset.program == CREDITS_PROGRAM and asset.standard not in (None, "credits"):
        raise InventoryError("credits.aleo requires the credits standard")
    if asset.program == CREDITS_PROGRAM:
        # Native credits have a fixed record interface; an override must match it.
        if (asset.record_name not in (None, "credits")) or (asset.amount_field not in (None, "microcredits")):
            raise InventoryError("credits.aleo requires the credits record and microcredits amount field")
        resolved = ResolvedRecordAsset(
            CREDITS_PROGRAM, standard="credits", record_name="credits", amount_field="microcredits", amount_type="u64",
        )
        return AssetCapabilities(resolved, recipient_bound=False, excluded=False)
    if source is None:
        raise InventoryError(f"{asset.program} is not deployed; its record interface cannot be resolved")
    interface = parse_program_interface(source)
    join = interface.functions.get("join")
    split = interface.functions.get("split")
    join_input = join.inputs[0] if join and join.inputs else None
    # An explicit record wins. Otherwise follow the native join input, which is
    # Token for ARC programs and can be another name for a custom program.
    name = asset.record_name or (join_input.type if join_input and join_input.kind == "record" else "Token")
    record = interface.records.get(name)
    if record is None or interface.id != asset.program:
        raise InventoryError(f"{asset.program} has an incompatible record join/split interface")
    field_name = asset.amount_field or next(
        (candidate for candidate in ("amount", "microcredits") if record.field(candidate) is not None), None,
    )
    entry = record.field(field_name) if field_name else None
    if entry is None or not _UNSIGNED.match(entry.type):
        raise InventoryError("Record amount field must be a declared unsigned integer")
    owner = record.field("owner")
    if (
        owner is None or owner.type != "address"
        or not _join_split_match(join, split, name, entry.type)
    ):
        raise InventoryError(f"{asset.program} has an incompatible token join/split interface")
    standard = asset.standard
    if standard is None and name == "Token" and field_name == "amount" and entry.type == "u128":
        # Detect a standard only for the default Token balance; a custom field
        # must never inherit eligibility for a helper that sums Token.amount.
        standard = _detect_standard(interface)
    resolved = ResolvedRecordAsset(
        asset.program, standard=standard, record_name=name, amount_field=str(field_name),
        amount_type=entry.type,  # type: ignore[arg-type]  # matched by _UNSIGNED above
    )
    return AssetCapabilities(
        resolved,
        recipient_bound=record.field("recipient_bound") is not None,
        excluded=any(record.field(item) is not None for item in EXCLUDED_RECORD_FIELDS),
    )


def _join_split_match(join: ProgramFunction | None, split: ProgramFunction | None, name: str, amount_type: str) -> bool:
    if join is None or split is None:
        return False
    return (
        len(join.inputs) == 2 and all(_is_record(param, name) for param in join.inputs)
        and len(join.outputs) == 1 and _is_record(join.outputs[0], name)
        and len(split.inputs) == 2 and _is_record(split.inputs[0], name)
        and split.inputs[1].kind == "plaintext" and split.inputs[1].type == amount_type
        and len(split.outputs) == 2 and all(_is_record(param, name) for param in split.outputs)
    )


def record_plaintext_of(record: Mapping[str, Any]) -> str | None:
    """The decrypted literal carried by a scanner record, or ``None``."""
    value = record.get("record_plaintext")
    return str(value) if value else None


def inventory_record_id(record: Mapping[str, Any]) -> str:
    """The record nonce (``…group``) that identifies a record in plans and journals.

    Raises:
        InventoryError: The record carries no decrypted plaintext.
    """
    plaintext = record_plaintext_of(record)
    match = _NONCE.search(plaintext) if plaintext else None
    if match is None:
        raise InventoryError("Inventory requires decrypted record plaintext; register the account for scanning")
    return match.group(1)


def plaintext_field(plaintext: str, name: str) -> str | None:
    """The literal value of entry *name* in a record plaintext, without its visibility suffix."""
    match = re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}:\s*([^,\s}}]+)", plaintext)
    if match is None:
        return None
    value = match.group(1)
    for suffix in (".private", ".public", ".constant"):
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def decode_record(
    address: str, asset: RecordAsset, record: Mapping[str, Any], capabilities: AssetCapabilities,
) -> InventoryRecord | None:
    """Decode one scanner record as spendable inventory, or ``None`` when it is not.

    A record is skipped when it belongs to another program or owner, is
    already spent, is a compliance or registry record, is recipient-bound,
    or holds an amount outside the asset's declared width.

    Raises:
        InventoryError: The record has no decrypted plaintext, so eligibility
            cannot be established.
    """
    resolved = capabilities.asset
    if capabilities.excluded or record.get("program_name") != asset.program or record.get("spent") is True:
        return None
    declared_name = record.get("record_name")
    if declared_name and declared_name != resolved.record_name:
        return None
    plaintext = record_plaintext_of(record)
    if plaintext is None:
        raise InventoryError("Inventory requires decrypted record plaintext; register the account for scanning")
    owner = _OWNER.search(plaintext)
    if owner is None or owner.group(1) != address:
        return None
    if any(plaintext_field(plaintext, name) is not None for name in EXCLUDED_RECORD_FIELDS):
        return None
    if plaintext_field(plaintext, "recipient_bound") == "true":
        return None
    raw = plaintext_field(plaintext, resolved.amount_field)
    match = _AMOUNT.match(raw) if raw else None
    if match is None or match.group(2) != resolved.amount_type:
        return None
    amount = int(match.group(1))
    if amount <= 0 or amount > max_amount(resolved):
        return None
    return InventoryRecord(inventory_record_id(record), amount, record)


__all__ = [
    "AssetCapabilities", "CREDITS_PROGRAM", "CREDITS_SPLIT_DEDUCTION", "EXCLUDED_RECORD_FIELDS", "amount_width",
    "decode_record", "inventory_record_id", "max_amount", "plaintext_field", "record_name", "record_plaintext_of",
    "scope_for", "split_deduction", "validate_asset",
]
