"""Pinned-ABI drift test: a contract redeploy fails OUR CI, not a consumer."""
import json
import re
from pathlib import Path

import pytest

from .conftest import ENDPOINT

pytestmark = pytest.mark.live

ABI_PATH = Path(__file__).parents[2] / "codegen" / "shield_swap.abi.json"
PROGRAM = "shield_swap_v3.aleo"


def _fetch(program_id: str) -> str:
    import requests

    r = requests.get(f"{ENDPOINT}/v2/testnet/program/{program_id}", timeout=30)
    r.raise_for_status()
    return r.json()


def test_pinned_abi_matches_deployed():
    import aleo.abi

    src = _fetch(PROGRAM)
    deps = []
    for dep in re.findall(r"^import\s+(\S+?);\s*$", src, re.MULTILINE):
        if dep != "credits.aleo":
            deps.append((dep, _fetch(dep)))
    live = aleo.abi.generate_abi(src, "testnet", imports=deps)
    violations = aleo.abi.check_compatibility(live, json.loads(ABI_PATH.read_text()))
    assert violations == [], (
        f"deployed {PROGRAM} drifted from the pinned ABI — rerun "
        f"codegen/regen-abi.sh and review the diff: {violations}"
    )
