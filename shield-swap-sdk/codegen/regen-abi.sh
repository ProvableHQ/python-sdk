#!/usr/bin/env bash
# Refetch the deployed program and regenerate the pinned ABI + bindings.
# Run from anywhere; requires the aleo + aleo-contract-abi-generator packages on python3's path.
set -euo pipefail
cd "$(dirname "$0")"
PROGRAM="${1:-shield_swap_v3.aleo}"
PYTHON="${PYTHON:-python3}"
"$PYTHON" - "$PROGRAM" <<'EOF'
import json
import re
import sys

import requests

import aleo.abi

NODE = "https://api.provable.com/v2/testnet/program/{}"


def fetch(program_id: str) -> str:
    r = requests.get(NODE.format(program_id), timeout=30)
    r.raise_for_status()
    return r.json()


def imports_of(src: str) -> list[str]:
    return re.findall(r"^import\s+(\S+?);\s*$", src, re.MULTILINE)


def load_deps(program_id: str, seen: dict) -> None:
    """Post-order DFS: dependencies land in `seen` before their dependents."""
    src = fetch(program_id)
    for dep in imports_of(src):
        if dep not in seen and dep != "credits.aleo":  # credits is preloaded
            load_deps(dep, seen)
    seen[program_id] = src


program = sys.argv[1]
seen: dict = {}
load_deps(program, seen)
src = seen.pop(program)
deps = list(seen.items())
print(f"deps (topological): {[d for d, _ in deps] or 'none'}")
abi = aleo.abi.generate_abi(src, "testnet", imports=deps)
json.dump(abi, open("shield_swap.abi.json", "w"), indent=1)
print(f"wrote shield_swap.abi.json ({len(abi['structs'])} structs, "
      f"{len(abi['records'])} records, {len(abi['mappings'])} mappings)")
EOF
"$PYTHON" -m aleo.codegen --abi shield_swap.abi.json \
    --out ../python/aleo_shield_swap/_generated.py
echo "regenerated: shield_swap.abi.json + _generated.py — review the git diff"
