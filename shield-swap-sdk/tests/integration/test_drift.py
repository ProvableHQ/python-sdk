"""Pinned-ABI drift test: a contract redeploy fails OUR CI, not a consumer."""
import json
import re
from pathlib import Path

import pytest

from .conftest import ENDPOINT

pytestmark = pytest.mark.live

ABI_PATH = Path(__file__).parents[2] / "codegen" / "shield_swap.abi.json"
PROGRAM = "shield_swap.aleo"


def _fetch(program_id: str) -> str:
    import requests

    r = requests.get(f"{ENDPOINT}/v2/testnet/program/{program_id}", timeout=30)
    r.raise_for_status()
    return r.json()


def _load_deps(program_id: str, seen: dict) -> None:
    """Post-order DFS: dependencies land in *seen* before their dependents
    (the freezelist itself imports the multisig)."""
    src = _fetch(program_id)
    for dep in re.findall(r"^import\s+(\S+?);\s*$", src, re.MULTILINE):
        if dep not in seen and dep != "credits.aleo":
            _load_deps(dep, seen)
    seen[program_id] = src


def test_pinned_abi_matches_deployed():
    import aleo.abi

    seen: dict = {}
    _load_deps(PROGRAM, seen)
    src = seen.pop(PROGRAM)
    live = aleo.abi.generate_abi(src, "testnet", imports=list(seen.items()))
    violations = aleo.abi.check_compatibility(live, json.loads(ABI_PATH.read_text()))
    assert violations == [], (
        f"deployed {PROGRAM} drifted from the pinned ABI — rerun "
        f"codegen/regen-abi.sh and review the diff: {violations}"
    )
