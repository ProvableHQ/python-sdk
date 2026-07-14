"""Tests for the aleo.abi integration hook."""
import importlib.util
import json
import sys

import pytest


SIMPLE_ALEO = """program simple.aleo;

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


_HAS_ALEO_ABI = importlib.util.find_spec("aleo_abi") is not None
requires_aleo_abi = pytest.mark.skipif(
    not _HAS_ALEO_ABI, reason="aleo-contract-abi-generator not installed (pip install aleo-contract-abi-generator)"
)


@requires_aleo_abi
def test_generate_abi_credits():
    """generate_abi(Program.credits()) returns dict with program=='credits.aleo'."""
    from aleo import abi
    from aleo.mainnet import Program
    credits = Program.credits()
    result = abi.generate_abi(credits)
    assert isinstance(result, dict)
    assert result["program"] == "credits.aleo"
    assert len(result.get("functions", [])) > 0


@requires_aleo_abi
def test_generate_abi_string_input():
    """generate_abi works with a raw bytecode string."""
    from aleo import abi
    result = abi.generate_abi(SIMPLE_ALEO, network="testnet")
    assert isinstance(result, dict)
    assert result["program"] == "simple.aleo"
    assert len(result.get("functions", [])) > 0


def test_generate_abi_lazy_import_error():
    """Helpful ImportError when aleo_abi is not installed."""
    # Hide the aleo_abi package from sys.modules
    saved = sys.modules.pop("aleo_abi", None)
    # Also block future imports
    sys.modules["aleo_abi"] = None  # type: ignore[assignment]
    try:
        from aleo import abi
        import importlib
        importlib.reload(abi)
        with pytest.raises(ImportError, match="pip install aleo-contract-abi-generator"):
            abi.generate_abi(SIMPLE_ALEO)
    finally:
        if saved is None:
            sys.modules.pop("aleo_abi", None)
        else:
            sys.modules["aleo_abi"] = saved
