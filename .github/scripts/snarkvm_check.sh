#!/usr/bin/env bash
# Are the SDK's two upstream pins — snarkvm (sdk/) and leo (sdk-abi/) — behind
# their latest releases, and is an upgrade PR already open?
#
# Output (stdout `key=value`; also appended to $GITHUB_OUTPUT when set):
#   current=v4.8.1              sdk/'s pinned snarkvm version
#   latest=v4.9.0               newest bare vX.Y.Z snarkvm tag (no testnet-/canary-)
#   crates_io=true|false        does crates.io publish a release equal to `latest`?
#   snarkvm_needs_upgrade=true|false
#   leo_current=4.3.2           leo version at sdk-abi's pinned rev
#   leo_latest=4.3.4            newest leo release version (max over leo-*-vX.Y.Z tags)
#   leo_current_rev=<sha>       sdk-abi's pinned leo rev
#   leo_latest_rev=<sha>        rev the newest leo release tags point at
#   leo_needs_upgrade=true|false
#   branch=deps-upgrade/...     branch the upgrade should use (empty if none needed)
#   needs_upgrade=true|false    either pin behind, and no PR open for `branch`
#   devnode_latest=v0.2.2       newest aleo-devnode release (informational —
#   devnode_snarkvm=testnet-v4.9.0   the SDK carries no devnode pin; this tells
#                               the upgrade PR whether the devnode will skew)
#
# Usage: snarkvm_check.sh [--current vX.Y.Z] [--leo-current X.Y.Z]
#   Overrides pin detection (testing / workflow force_from_tag).
set -euo pipefail

REPO_URL="https://github.com/ProvableHQ/snarkVM"
LEO_URL="https://github.com/ProvableHQ/leo"
DEVNODE_URL="https://github.com/ProvableHQ/aleo-devnode"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CARGO_TOML="$ROOT/sdk/Cargo.toml"
ABI_CARGO="$ROOT/sdk-abi/Cargo.toml"

force_current=""
force_leo=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --current) force_current="${2:?--current requires a tag argument}"; shift 2 ;;
    --leo-current) force_leo="${2:?--leo-current requires a version argument}"; shift 2 ;;
    *) echo "error: unknown argument '$1'" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------- snarkvm ----
if [[ -n "$force_current" ]]; then
  current="$force_current"
else
  # Fail-soft the grep so the parse error below reports the real problem
  # rather than set -e killing the script with a bare exit 1.
  dep_line=$(grep -m1 '^snarkvm = {' "$CARGO_TOML" || true)
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
  | sort -uV | tail -1 || true)
if [[ -z "$latest" ]]; then
  echo "error: no vX.Y.Z release tags found on $REPO_URL" >&2
  exit 1
fi

# Prefer the crates.io release when it matches `latest`; otherwise the tag is
# ahead of the registry and only the git pin can reach it.
crates_io=false
if curl -fsS -o /dev/null -A "ProvableHQ/python-sdk snarkvm-upgrade check" \
     "https://crates.io/api/v1/crates/snarkvm/${latest#v}"; then
  crates_io=true
fi

snarkvm_needs_upgrade=false
[[ "$current" != "$latest" ]] && snarkvm_needs_upgrade=true

# -------------------------------------------------------------------- leo ----
# sdk-abi pins leo by rev. Leo tags per crate (leo-abi-v4.3.4, leo-lang-v4.3.4,
# ...) and one release tags several crates at a single rev — but not every
# crate every time (leo-abi's newest tag lags leo-lang's). So take the highest
# version across ALL leo-*-v tags as the latest release, then resolve any tag
# at that version to its rev.
leo_current_rev="$(grep -m1 -oE 'rev = "[0-9a-f]{7,40}"' "$ABI_CARGO" \
  | grep -oE '[0-9a-f]{7,40}' || true)"

leo_tags=$(git ls-remote --tags "$LEO_URL" 2>/dev/null \
  | awk '{print $2}' \
  | sed 's|^refs/tags/||; s|\^{}$||' \
  | grep -E '^leo-[a-z0-9-]+-v[0-9]+\.[0-9]+\.[0-9]+$' | sort -u || true)
leo_latest=$(printf '%s\n' "$leo_tags" | sed -E 's/^leo-[a-z0-9-]+-v//' \
  | sort -uV | tail -1 || true)

leo_latest_rev=""
if [[ -n "$leo_latest" ]]; then
  leo_latest_tag=$(printf '%s\n' "$leo_tags" \
    | grep -E -- "-v${leo_latest//./\\.}\$" | head -1 || true)
  if [[ -n "${leo_latest_tag:-}" ]]; then
    leo_latest_rev=$(git ls-remote "$LEO_URL" "refs/tags/$leo_latest_tag" \
      2>/dev/null | awk '{print $1}' | head -1 || true)
  fi
fi

# Which leo version is the pinned rev? Read it from leo's own manifest at that
# rev — no extra bookkeeping in sdk-abi to drift out of date.
if [[ -n "$force_leo" ]]; then
  leo_current="$force_leo"
elif [[ -n "$leo_current_rev" ]]; then
  leo_current=$(curl -fsS \
    "https://raw.githubusercontent.com/ProvableHQ/leo/${leo_current_rev}/Cargo.toml" \
    2>/dev/null | grep -m1 '^leo-abi ' \
    | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)
  leo_current="${leo_current:-unknown}"
else
  leo_current=unknown
fi

# Only claim an upgrade when both sides are known — an unknown must never look
# like "behind" and kick off a 90-minute run on bad data.
leo_needs_upgrade=false
if [[ "$leo_current" != unknown && -n "$leo_latest" && -n "$leo_latest_rev" \
      && "$leo_current" != "$leo_latest" ]]; then
  leo_needs_upgrade=true
fi

# ----------------------------------------------------------------- branch ----
slug=""
[[ "$snarkvm_needs_upgrade" == true ]] && slug="snarkvm-$latest"
if [[ "$leo_needs_upgrade" == true ]]; then
  slug="${slug:+$slug-}leo-v$leo_latest"
fi
branch="${slug:+deps-upgrade/$slug}"

# If a PR for this exact branch is already open, don't re-attempt daily.
# Fail-soft when gh is unavailable/unauthenticated: worst case the skill's
# own `gh pr create` collides on the existing head branch and errors there.
pr_open=0
if [[ -n "$branch" ]] && command -v gh >/dev/null 2>&1; then
  pr_open=$(gh pr list --head "$branch" --state open \
              --json number --jq length 2>/dev/null || echo 0)
fi

needs_upgrade=false
if [[ -n "$branch" && "$pr_open" == "0" ]]; then
  needs_upgrade=true
fi

# ---------------------------------------------------------------- devnode ----
# Informational only — never gates the upgrade. Which snarkvm does the latest
# devnode release pin? Handles both git-tag and crates.io dep forms; fail-soft
# to "unknown" on network/parse trouble.
devnode_latest=$(git ls-remote --tags "$DEVNODE_URL" 2>/dev/null \
  | awk '{print $2}' \
  | sed 's|^refs/tags/||; s|\^{}$||' \
  | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' \
  | sort -uV | tail -1 || true)
devnode_snarkvm=unknown
if [[ -n "$devnode_latest" ]]; then
  devnode_dep=$(curl -fsS \
    "https://raw.githubusercontent.com/ProvableHQ/aleo-devnode/${devnode_latest}/Cargo.toml" \
    2>/dev/null | grep -m1 '^snarkvm ' || true)
  # The devnode tracks testnet tags as often as mainnet ones, so keep any
  # testnet-/canary- prefix verbatim — "testnet-v4.9.0" != "v4.9.0" and the
  # PR body should say which line the devnode is on.
  if [[ $devnode_dep =~ tag\ =\ \"((testnet-|canary-)?v[0-9]+\.[0-9]+\.[0-9]+)\" ]]; then
    devnode_snarkvm="${BASH_REMATCH[1]}"
  elif [[ $devnode_dep =~ version\ =\ \"([0-9]+\.[0-9]+\.[0-9]+)\" ]]; then
    devnode_snarkvm="v${BASH_REMATCH[1]}"
  fi
fi
devnode_latest="${devnode_latest:-unknown}"

out="current=$current
latest=$latest
crates_io=$crates_io
snarkvm_needs_upgrade=$snarkvm_needs_upgrade
leo_current=$leo_current
leo_latest=${leo_latest:-unknown}
leo_current_rev=${leo_current_rev:-unknown}
leo_latest_rev=${leo_latest_rev:-unknown}
leo_needs_upgrade=$leo_needs_upgrade
branch=$branch
needs_upgrade=$needs_upgrade
devnode_latest=$devnode_latest
devnode_snarkvm=$devnode_snarkvm"
echo "$out"
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  echo "$out" >> "$GITHUB_OUTPUT"
fi
