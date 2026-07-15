# Copyright (C) 2019-2026 Provable Inc.
# SPDX-License-Identifier: GPL-3.0-or-later
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Integration hook: ABI generation via the aleo-contract-abi-generator package."""
from __future__ import annotations

import json
import re
from typing import Any, Union


def generate_abi(
    program: object,
    network: str = "mainnet",
    imports: "list[tuple[str, str]] | None" = None,
) -> dict[str, Any]:
    """Generate an ABI dict for an Aleo program.

    Args:
        program: Either an aleo Program object (with .source and .id attributes)
                 or a raw bytecode string.
        network: Network name: "mainnet", "testnet", or "canary".
        imports: Optional ``(program_id, bytecode)`` dependencies in
                 topological order (dependencies before dependents).  snarkVM
                 validation is contextual, so a program that declares imports
                 is rejected unless they are supplied here.

    Returns:
        A dict containing the ABI for the program.

    Raises:
        ImportError: If the aleo-contract-abi-generator package is not installed.
        ValueError: If the program name cannot be determined.
    """
    try:
        import aleo_abi as _aleo_abi  # pyright: ignore[reportMissingImports]
    except ImportError:
        raise ImportError(
            "The aleo-contract-abi-generator package is required for ABI generation. "
            "Install it with: pip install aleo-contract-abi-generator"
        )

    # Duck-type: if it has .source and .id, treat as Program object
    if hasattr(program, "source") and hasattr(program, "id"):
        bytecode: str = getattr(program, "source")
        name: str = str(getattr(program, "id"))
    elif isinstance(program, str):
        bytecode = program
        # Parse name from `program X.aleo;` line
        m = re.search(r"program\s+(\S+\.aleo)\s*;", bytecode)
        if m is None:
            raise ValueError(
                "Could not determine program name from bytecode. "
                "Expected a line like: program foo.aleo;"
            )
        name = m.group(1)
    else:
        raise TypeError(
            f"Expected a Program object or str, got {type(program).__name__}"
        )

    json_str = _aleo_abi.generate_abi(name, bytecode, network, imports)
    return json.loads(json_str)


def check_compatibility(
    candidate: Union[dict[str, Any], str],
    standard: Union[dict[str, Any], str],
) -> list[str]:
    """Check whether a candidate ABI is compatible with a standard ABI.

    Args:
        candidate: The candidate program ABI as a dict or JSON string.
        standard: The standard/interface ABI as a dict or JSON string.

    Returns:
        A list of violation strings. Empty list means compatible.

    Raises:
        ImportError: If the aleo-contract-abi-generator package is not installed.
    """
    try:
        import aleo_abi as _aleo_abi  # pyright: ignore[reportMissingImports]
    except ImportError:
        raise ImportError(
            "The aleo-contract-abi-generator package is required for compatibility checking. "
            "Install it with: pip install aleo-contract-abi-generator"
        )

    if isinstance(candidate, dict):
        candidate_json = json.dumps(candidate)
    else:
        candidate_json = candidate

    if isinstance(standard, dict):
        standard_json = json.dumps(standard)
    else:
        standard_json = standard

    result: list[str] = _aleo_abi.check_compatibility(candidate_json, standard_json)
    return result
