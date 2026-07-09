# Copyright (C) 2024 Provable Inc.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the aleo-abi package."""

import json

import pytest

import aleo_abi


# ---------------------------------------------------------------------------
# Fixtures vendored from leo/tests/tests/cli/test_abi_from_aleo/contents/
# ---------------------------------------------------------------------------

SIMPLE_ALEO = """\
program simple.aleo;

struct Point:
    x as i32;
    y as i32;

record Token:
    owner as address.private;
    amount as u64.public;

mapping balances:
    key as address.public;
    value as u64.public;

function add_numbers:
    input r0 as u32.private;
    input r1 as u32.private;
    add r0 r1 into r2;
    output r2 as u32.private;

function get_point:
    input r0 as Point.private;
    output r0 as Point.private;

function get_token:
    input r0 as Token.record;
    output r0 as Token.record;"""

# Vendored from leo/tests/expectations/cli/test_abi_from_aleo/contents/simple_out/simple.aleo.abi.json
SIMPLE_EXPECTED = {
    "program": "simple.aleo",
    "structs": [
        {
            "path": ["Point"],
            "fields": [
                {"name": "x", "ty": {"Primitive": {"Int": "I32"}}},
                {"name": "y", "ty": {"Primitive": {"Int": "I32"}}},
            ],
        }
    ],
    "records": [
        {
            "path": ["Token"],
            "fields": [
                {
                    "name": "owner",
                    "ty": {"Primitive": "Address"},
                    "mode": "Public",
                },
                {
                    "name": "amount",
                    "ty": {"Primitive": {"UInt": "U64"}},
                    "mode": "Private",
                },
            ],
        }
    ],
    "mappings": [
        {
            "name": "balances",
            "key": {"Primitive": "Address"},
            "value": {"Primitive": {"UInt": "U64"}},
        }
    ],
    "storage_variables": [],
    "functions": [
        {
            "name": "add_numbers",
            "inputs": [
                {
                    "Plaintext": {
                        "ty": {"Primitive": {"UInt": "U32"}},
                        "mode": "Private",
                    }
                },
                {
                    "Plaintext": {
                        "ty": {"Primitive": {"UInt": "U32"}},
                        "mode": "Private",
                    }
                },
            ],
            "outputs": [
                {
                    "Plaintext": {
                        "ty": {"Primitive": {"UInt": "U32"}},
                        "mode": "Private",
                    }
                }
            ],
        },
        {
            "name": "get_point",
            "inputs": [
                {
                    "Plaintext": {
                        "ty": {
                            "Struct": {
                                "path": ["Point"],
                                "program": "simple.aleo",
                            }
                        },
                        "mode": "Private",
                    }
                }
            ],
            "outputs": [
                {
                    "Plaintext": {
                        "ty": {
                            "Struct": {
                                "path": ["Point"],
                                "program": "simple.aleo",
                            }
                        },
                        "mode": "Private",
                    }
                }
            ],
        },
        {
            "name": "get_token",
            "inputs": [
                {
                    "Record": {
                        "path": ["Token"],
                        "program": "simple.aleo",
                    }
                }
            ],
            "outputs": [
                {
                    "Record": {
                        "path": ["Token"],
                        "program": "simple.aleo",
                    }
                }
            ],
        },
    ],
}


# ---------------------------------------------------------------------------
# generate_abi tests
# ---------------------------------------------------------------------------


def test_generate_abi_simple():
    """generate_abi on simple.aleo produces the expected ABI."""
    result_json = aleo_abi.generate_abi("simple.aleo", SIMPLE_ALEO, "testnet")
    result = json.loads(result_json)
    assert result == SIMPLE_EXPECTED


def test_generate_abi_returns_string():
    """generate_abi returns a JSON string."""
    result = aleo_abi.generate_abi("simple.aleo", SIMPLE_ALEO, "testnet")
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert parsed["program"] == "simple.aleo"


def test_generate_abi_mainnet():
    """generate_abi works with 'mainnet' network."""
    result_json = aleo_abi.generate_abi("simple.aleo", SIMPLE_ALEO, "mainnet")
    result = json.loads(result_json)
    assert result["program"] == "simple.aleo"


def test_generate_abi_unknown_network():
    """generate_abi raises on unknown network."""
    with pytest.raises(Exception, match="unknown network"):
        aleo_abi.generate_abi("simple.aleo", SIMPLE_ALEO, "banana")


# ---------------------------------------------------------------------------
# check_compatibility tests
# ---------------------------------------------------------------------------


def test_check_compatibility_identical():
    """Identical ABIs produce no violations."""
    abi_json = json.dumps(SIMPLE_EXPECTED)
    violations = aleo_abi.check_compatibility(abi_json, abi_json)
    assert violations == []


def test_check_compatibility_missing_function():
    """Candidate missing a function that standard requires → violation."""
    # Standard has all functions; candidate is missing get_token
    candidate = dict(SIMPLE_EXPECTED)
    candidate["functions"] = [
        f for f in SIMPLE_EXPECTED["functions"] if f["name"] != "get_token"
    ]
    violations = aleo_abi.check_compatibility(
        json.dumps(candidate), json.dumps(SIMPLE_EXPECTED)
    )
    assert len(violations) > 0
    assert any("get_token" in v for v in violations)


def test_check_compatibility_different_signature():
    """Candidate function with different signature → violation."""
    # Mutate add_numbers to have a public output instead of private
    import copy

    candidate = copy.deepcopy(SIMPLE_EXPECTED)
    for fn in candidate["functions"]:
        if fn["name"] == "add_numbers":
            fn["outputs"][0]["Plaintext"]["mode"] = "Public"
    violations = aleo_abi.check_compatibility(
        json.dumps(candidate), json.dumps(SIMPLE_EXPECTED)
    )
    assert len(violations) > 0


def test_check_compatibility_returns_list():
    """check_compatibility always returns a list."""
    abi_json = json.dumps(SIMPLE_EXPECTED)
    result = aleo_abi.check_compatibility(abi_json, abi_json)
    assert isinstance(result, list)
