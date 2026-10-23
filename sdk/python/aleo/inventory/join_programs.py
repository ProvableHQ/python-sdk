"""Deployed helpers that join several records in one transaction.

A native ``join`` consumes two records. Consolidating many records pairwise
costs one transaction per record. The deployed helpers below join up to 16
credits or dedicated stablecoin records, or up to 15 records of any ARC-20 or
ARC-22 token, in a single transaction. The planner prefers them; execution
re-checks each helper's deployed signature before sending records to it.
"""
from __future__ import annotations

import re
from typing import Literal, Mapping

from .assets import CREDITS_PROGRAM
from .interface import ProgramFunction, ProgramInterface
from .types import (
    MAX_DYNAMIC_JOIN, MAX_STATIC_JOIN, InventoryError, RecordJoinProgram, ResolvedRecordAsset, TokenJoinRouter,
)

_PROGRAM_ID = re.compile(r"^[a-z][a-z0-9_]*\.aleo$")
Network = Literal["mainnet", "testnet"]


def _counts(low: int, high: int) -> tuple[int, ...]:
    return tuple(range(low, high + 1))


def _static_helpers(prefix: str) -> tuple[RecordJoinProgram, ...]:
    # Static helper deployments split their join functions into three size ranges.
    return tuple(
        RecordJoinProgram(f"{prefix}_{low}_{high}.aleo", "records", _counts(low, high))
        for low, high in ((2, 10), (11, 14), (15, 16))
    )


def _dynamic_helpers(network: Network) -> tuple[RecordJoinProgram, ...]:
    prefix = "main" if network == "mainnet" else "test"
    return (RecordJoinProgram(f"{prefix}_aj_arc20_2_15.aleo", "program_identifier_then_records", _counts(2, 15)),)


#: Known join helpers by network and source program, with ARC defaults.
#: An exact program entry takes precedence over the ``arc20`` / ``arc22``
#: entries. Selection still checks each helper's deployed signature; this
#: table alone does not establish that a helper exists on a node.
RECORD_JOIN_PROGRAMS: Mapping[str, Mapping[str, tuple[RecordJoinProgram, ...]]] = {
    "mainnet": {
        "credits.aleo": _static_helpers("autojoin_credits"),
        "usad_stablecoin.aleo": _static_helpers("aj_usad_stablecoin"),
        "usdcx_stablecoin.aleo": _static_helpers("aj_usdcx_stablecoin"),
        "arc20": _dynamic_helpers("mainnet"),
        "arc22": _dynamic_helpers("mainnet"),
    },
    "testnet": {
        "credits.aleo": _static_helpers("autojoin_credits"),
        "test_usad_stablecoin.aleo": _static_helpers("aj_test_usad_stablecoin"),
        "test_usdcx_stablecoin.aleo": _static_helpers("aj_test_usdcx_stablecoin"),
        "arc20": _dynamic_helpers("testnet"),
        "arc22": _dynamic_helpers("testnet"),
    },
}


def validate_record_join_program(helper: RecordJoinProgram) -> None:
    """Check helper metadata without contacting a node.

    Raises:
        InventoryError: The program id is malformed, a count repeats, or a
            count lies outside 2 through 16 (15 for dynamic routers).
    """
    limit = MAX_STATIC_JOIN if helper.input_mode == "records" else MAX_DYNAMIC_JOIN
    if (
        not _PROGRAM_ID.match(helper.program)
        or helper.input_mode not in ("records", "program_identifier_then_records")
        or not helper.counts or len(set(helper.counts)) != len(helper.counts)
        or any(count < 2 or count > limit for count in helper.counts)
    ):
        raise InventoryError("Invalid record join helper configuration")


def validate_token_join_router(router: TokenJoinRouter) -> int:
    """Check a configured router without contacting a node and return its input ceiling.

    Raises:
        InventoryError: The program id is malformed or ``max_records`` is
            outside 2 through 15.
    """
    if not _PROGRAM_ID.match(router.program):
        raise InventoryError("Invalid token join router program")
    limit = router.max_records
    if limit < 2 or limit > MAX_DYNAMIC_JOIN:
        raise InventoryError("Token join router max_records must be from 2 to 15")
    return limit


def router_as_helper(router: TokenJoinRouter) -> RecordJoinProgram:
    """Describe a configured dynamic router as a join helper."""
    return RecordJoinProgram(router.program, "program_identifier_then_records", _counts(2, validate_token_join_router(router)))


def join_program_candidates(
    network: str, asset: ResolvedRecordAsset, token_join: TokenJoinRouter | Literal[False] | None,
) -> tuple[RecordJoinProgram, ...]:
    """Helpers worth checking for *asset* on *network*, before reading any program.

    Returns nothing when helpers are disabled, when an explicit router covers
    the asset, when the network has no known helpers, or when the asset uses a
    custom record or amount field that no helper understands.
    """
    if token_join is False or (token_join is not None and asset.program != CREDITS_PROGRAM):
        return ()
    registry = RECORD_JOIN_PROGRAMS.get(network)
    if registry is None:
        return ()
    if asset.program == CREDITS_PROGRAM:
        defaults_match = asset.record_name == "credits" and asset.amount_field == "microcredits"
    else:
        defaults_match = asset.record_name == "Token" and asset.amount_field == "amount" and asset.amount_type == "u128"
    if not defaults_match:
        return ()
    candidates = registry.get(asset.program)
    if candidates is None and asset.standard is not None:
        candidates = registry.get(asset.standard)
    return candidates or ()


def join_function_compatible(
    function: ProgramFunction | None, asset: ResolvedRecordAsset, helper: RecordJoinProgram, count: int,
) -> bool:
    """Whether a deployed ``join_N`` accepts *count* of the asset's records and returns one."""
    if function is None or len(function.outputs) != 1:
        return False
    if helper.input_mode == "program_identifier_then_records":
        # Dynamic routers call Token.join and sum ``amount``; a custom record or
        # field must not be sent through that interface.
        if not (asset.record_name == "Token" and asset.amount_field == "amount" and asset.amount_type == "u128"):
            return False
        first = function.inputs[0] if function.inputs else None
        return (
            len(function.inputs) == count + 1
            and first is not None and first.kind == "plaintext" and first.type == "identifier" and first.visibility == "public"
            and all(param.kind == "dynamic_record" for param in function.inputs[1:])
            and function.outputs[0].kind == "dynamic_record"
        )
    external = f"{asset.program}/{asset.record_name}"
    return (
        len(function.inputs) == count
        and all(param.kind == "external_record" and param.type == external for param in function.inputs)
        and function.outputs[0].kind == "external_record" and function.outputs[0].type == external
    )


def supported_join_program(
    helper: RecordJoinProgram, interface: ProgramInterface, asset: ResolvedRecordAsset,
) -> RecordJoinProgram | None:
    """Restrict a helper to the counts its deployed code supports for *asset*; ``None`` when none do."""
    if interface.id != helper.program:
        return None
    counts = tuple(
        count for count in helper.counts
        if join_function_compatible(interface.functions.get(f"join_{count}"), asset, helper, count)
    )
    return RecordJoinProgram(helper.program, helper.input_mode, counts) if counts else None


def validate_record_join_function(
    helper: RecordJoinProgram, interface: ProgramInterface, asset: ResolvedRecordAsset, count: int,
) -> None:
    """Re-check a helper's deployed signature immediately before spending through it.

    Raises:
        InventoryError: The count is unsupported or the deployed function
            does not accept and return the asset's record.
    """
    validate_record_join_program(helper)
    if count not in helper.counts:
        raise InventoryError("Record count is outside the join helper limits")
    if interface.id != helper.program or not join_function_compatible(
        interface.functions.get(f"join_{count}"), asset, helper, count,
    ):
        raise InventoryError("Record join helper has an incompatible function signature")


def validate_token_join_function(router: TokenJoinRouter, interface: ProgramInterface, count: int) -> None:
    """Check a configured router's deployed ``join_N`` before sending it records.

    Raises:
        InventoryError: The count is outside the router's ceiling or the
            deployed function lacks the public identifier and dynamic record
            signature.
    """
    limit = validate_token_join_router(router)
    if count < 2 or count > limit:
        raise InventoryError(f"Token join requires 2 to {limit} records")
    function = interface.functions.get(f"join_{count}")
    first = function.inputs[0] if function and function.inputs else None
    if (
        interface.id != router.program or function is None or len(function.inputs) != count + 1
        or first is None or first.kind != "plaintext" or first.type != "identifier" or first.visibility != "public"
        or any(param.kind != "dynamic_record" for param in function.inputs[1:])
        or len(function.outputs) != 1 or function.outputs[0].kind != "dynamic_record"
    ):
        raise InventoryError("Token join router has an incompatible function signature")


__all__ = [
    "RECORD_JOIN_PROGRAMS", "join_function_compatible", "join_program_candidates", "router_as_helper",
    "supported_join_program", "validate_record_join_function", "validate_record_join_program",
    "validate_token_join_function", "validate_token_join_router",
]
