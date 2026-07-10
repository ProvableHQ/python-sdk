#!/usr/bin/env bash
#
# build-both.sh — build+install BOTH network extension modules into the active
# venv for local development/testing.
#
# maturin builds exactly one cdylib per invocation, named by pyproject
# [tool.maturin] module-name (aleo._aleolib_mainnet). To get both networks in a
# single importable `aleo` package we:
#   1. `maturin develop --features mainnet` — installs the aleo package (editable
#      .pth -> sdk/python) with _aleolib_mainnet.abi3.so placed in python/aleo/.
#   2. Build a testnet wheel with the module-name overridden to
#      aleo._aleolib_testnet (via MATURIN_PEP517... no — via a CLI/env override),
#      then extract _aleolib_testnet.abi3.so out of that wheel and drop it next
#      to the mainnet .so in python/aleo/.
#
# The result: `import aleo.mainnet` and `import aleo.testnet` both resolve.
#
# Idempotent: re-running rebuilds and overwrites both .so files.
#
# Usage:  (from sdk/, with your venv active)   ./build-both.sh
set -euo pipefail

SDK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SDK_DIR"

PKG_DIR="$SDK_DIR/python/aleo"

if ! command -v maturin >/dev/null 2>&1; then
  echo "error: maturin not found on PATH. Activate your venv and 'pip install maturin'." >&2
  exit 1
fi

echo "==> [1/2] mainnet: maturin develop --features mainnet"
maturin develop --features mainnet

echo "==> [2/2] testnet: build wheel with module-name override, extract .so"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Override module-name so this build emits _aleolib_testnet (matching the
# cfg-named #[pymodule] under --features testnet). maturin reads module-name
# from pyproject; there is no CLI flag, so we build in a temp copy of the
# manifest dir with a patched pyproject and point the sources back here.
#
# Simpler + robust across maturin versions: patch pyproject.toml in place,
# build, then restore. We snapshot and restore on exit.
cp pyproject.toml "$WORK/pyproject.toml.bak"
restore_pyproject() { cp "$WORK/pyproject.toml.bak" pyproject.toml; }
trap 'restore_pyproject; rm -rf "$WORK"' EXIT

# Swap the module-name to the testnet extension.
python3 - "$SDK_DIR/pyproject.toml" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
s = s.replace('module-name = "aleo._aleolib_mainnet"',
              'module-name = "aleo._aleolib_testnet"')
open(p, "w").write(s)
PY

maturin build --no-default-features --features testnet --out "$WORK/dist"

restore_pyproject

WHEEL="$(ls -t "$WORK"/dist/*.whl | head -n1)"
echo "    built testnet wheel: $WHEEL"

# Extract just the testnet .so from the wheel and place it beside the mainnet one.
rm -rf "$WORK/unz"
python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$WHEEL" "$WORK/unz"
SO="$(find "$WORK/unz" -name '_aleolib_testnet*.so' | head -n1)"
if [ -z "$SO" ]; then
  echo "error: no _aleolib_testnet*.so found inside $WHEEL" >&2
  exit 1
fi
cp "$SO" "$PKG_DIR/"
echo "    installed $(basename "$SO") -> $PKG_DIR/"

echo "==> verifying both modules import"
python3 -c "
from aleo.mainnet import PrivateKey as M
from aleo.testnet import PrivateKey as T
print('mainnet + testnet import OK')
"
echo "==> done."
