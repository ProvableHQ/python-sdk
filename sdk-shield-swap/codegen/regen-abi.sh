#!/usr/bin/env bash
# Refetch the deployed program and regenerate the pinned ABI + bindings.
# Run from anywhere; requires the aleo + aleo-abi packages on python3's path.
set -euo pipefail
cd "$(dirname "$0")"
PROGRAM="${1:-shield_swap_v3.aleo}"
PYTHON="${PYTHON:-python3}"
curl -sf "https://api.provable.com/v2/testnet/program/${PROGRAM}" | "$PYTHON" -c \
  'import json,sys; sys.stdout.write(json.load(sys.stdin))' > shield_swap.aleo
"$PYTHON" - <<'EOF'
import json
import aleo.abi
src = open("shield_swap.aleo").read()
abi = aleo.abi.generate_abi(src, "testnet")
json.dump(abi, open("shield_swap.abi.json", "w"), indent=1)
EOF
"$PYTHON" -m aleo.codegen --abi shield_swap.abi.json \
    --out ../python/aleo_shield_swap/_generated.py
rm shield_swap.aleo   # ~600KB bytecode — pin the ABI, not the program
echo "regenerated: shield_swap.abi.json + _generated.py — review the git diff"
