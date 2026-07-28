"""Pinned-ABI drift test: a contract redeploy fails OUR CI, not a consumer."""
import json
import re
from pathlib import Path

import pytest

from .conftest import ENDPOINT

pytestmark = pytest.mark.live

CODEGEN = Path(__file__).parents[2] / "codegen"
# Core drives codegen; the routers are pinned as pure drift guards — the
# client assembles their inputs positionally from these exact signatures.
PROGRAMS = ["shield_swap.aleo", "shield_swap_router.aleo",
            "shield_swap_lp_router.aleo"]


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


@pytest.mark.parametrize("program", PROGRAMS)
def test_pinned_abi_matches_deployed(program):
    import aleo.abi

    seen: dict = {}
    _load_deps(program, seen)
    src = seen.pop(program)
    live = aleo.abi.generate_abi(src, "testnet", imports=list(seen.items()))
    pinned_path = CODEGEN / (program.removesuffix(".aleo") + ".abi.json")
    violations = aleo.abi.check_compatibility(live, json.loads(pinned_path.read_text()))
    assert violations == [], (
        f"deployed {program} drifted from the pinned ABI — rerun "
        f"codegen/regen-abi.sh and review the diff: {violations}"
    )
