#!/usr/bin/env bash
# Is the SDK's snarkvm pin behind the latest upstream release, and is an
# upgrade PR already open?
#
# Output (stdout `key=value`; also appended to $GITHUB_OUTPUT when set):
#   current=v4.8.1        SDK's pinned snarkvm version
#   latest=v4.8.1         newest bare vX.Y.Z tag upstream (no testnet-/canary-)
#   crates_io=true|false  is `latest` published on crates.io?
#   needs_upgrade=true|false
#
# Usage: snarkvm_check.sh [--current vX.Y.Z]
#   --current overrides pin detection (testing / workflow force_from_tag).
set -euo pipefail

REPO_URL="https://github.com/ProvableHQ/snarkVM"
CARGO_TOML="$(cd "$(dirname "$0")/../.." && pwd)/sdk/Cargo.toml"

current=""
if [[ "${1:-}" == "--current" ]]; then
  current="${2:?--current requires a tag argument}"
else
  dep_line=$(grep -m1 '^snarkvm = {' "$CARGO_TOML")
  if [[ $dep_line =~ tag\ =\ \"(v[0-9]+\.[0-9]+\.[0-9]+)\" ]]; then
    current="${BASH_REMATCH[1]}"
  elif [[ $dep_line =~ version\ =\ \"([0-9]+\.[0-9]+\.[0-9]+)\" ]]; then
    current="v${BASH_REMATCH[1]}"
  else
    echo "error: could not parse snarkvm pin from $CARGO_TOML" >&2
    exit 1
  fi
fi

latest=$(git ls-remote --tags "$REPO_URL" \
  | awk '{print $2}' \
  | sed 's|^refs/tags/||; s|\^{}$||' \
  | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' \
  | sort -uV | tail -1)
if [[ -z "$latest" ]]; then
  echo "error: no vX.Y.Z release tags found on $REPO_URL" >&2
  exit 1
fi

crates_io=false
if curl -fsS -o /dev/null -A "ProvableHQ/python-sdk snarkvm-upgrade check" \
     "https://crates.io/api/v1/crates/snarkvm/${latest#v}"; then
  crates_io=true
fi

# If an upgrade PR for `latest` is already open, don't re-attempt daily.
# Fail-soft when gh is unavailable/unauthenticated: worst case the skill's
# own `gh pr create` collides on the existing head branch and errors there.
pr_open=0
if command -v gh >/dev/null 2>&1; then
  pr_open=$(gh pr list --head "snarkvm-upgrade/${latest}" --state open \
              --json number --jq length 2>/dev/null || echo 0)
fi

needs_upgrade=false
if [[ "$current" != "$latest" && "$pr_open" == "0" ]]; then
  needs_upgrade=true
fi

out="current=$current
latest=$latest
crates_io=$crates_io
needs_upgrade=$needs_upgrade"
echo "$out"
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  echo "$out" >> "$GITHUB_OUTPUT"
fi
