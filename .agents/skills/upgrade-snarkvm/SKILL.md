---
name: upgrade-snarkvm
description: Use when upgrading the SDK's pinned upstream Rust dependencies — snarkVM (sdk/) and leo (sdk-abi/) — to their latest releases. Compares tags, adapts the code to the changeset, runs the native validation loop, bumps PyPI package versions, and opens a PR. Invoked as /upgrade-snarkvm interactively or by the snarkvm-upgrade workflow.
---

# Upgrade snarkVM and leo

Upgrade the SDK's two upstream Rust pins to their latest releases and adapt
the code. Same procedure interactively and in CI.

Two independent pins, each with its own release cadence:

- **`sdk/`** pins `snarkvm` — tracks snarkVM's latest mainnet release tag.
- **`sdk-abi/`** pins four `leo` crates by rev (`leo-abi`, `leo-ast`,
  `leo-disassembler`, `leo-span`) plus a `snarkvm` dep that must match leo's
  — tracks leo's latest release.

Either can be behind independently, so a run may upgrade one, the other, or
both. The pre-check tells you which.

## Hard rules

- Plain commit messages — NO Co-Authored-By trailers, NO "Generated with"
  footers.
- Every version number comes from a command output. Never guess one.
- Dependency form follows one policy: prefer the crates.io release when its
  version equals the latest GitHub tag; otherwise pin the git tag. The
  pre-check's `crates_io` output already answers this — don't re-derive it.
- `sdk-abi` does NOT get to apply that policy. Its snarkvm dep must resolve
  to the **same source *and* version as leo's**, because cargo compiles one
  crate from two different sources twice: registry `4.8.1` and
  `tag = "v4.8.1"` are distinct crates, so `Process<N>`/`Program<N>` in
  leo-disassembler's signatures stop unifying with sdk-abi's. Mirror leo's
  snarkvm dep line verbatim — same source, same version/tag, same features —
  and never choose it independently. This means **sdk-abi's snarkvm can and
  will lag `sdk/`'s** whenever leo hasn't adopted the newest snarkVM; that is
  correct and expected, not a problem to fix. The two are separate build
  graphs (`sdk-abi/Cargo.toml` declares its own `[workspace]`), so skew
  between them is harmless.
- Proof the source identity held: after building sdk-abi,
  `grep -c '^name = "snarkvm"' sdk-abi/Cargo.lock` must be exactly **1**. A
  2 means leo and sdk-abi disagree on the source and the type errors you're
  about to see are that, not a real API break.
- Validate seriously before giving up: on failures, re-read the changeset,
  fix, and re-run. Only after several genuine fix attempts do you fall back
  to the draft-PR path in step 8.

## 1. Detect

```bash
bash .github/scripts/snarkvm_check.sh
```

If `needs_upgrade=false`, report why (both pins current, or an open PR exists
for `branch` — check `gh pr list --head "<branch>"`) and STOP.

Otherwise read the outputs into:

- `snarkvm_needs_upgrade` — do step 3. CUR=`current`, NEW=`latest`,
  CRATES_IO=`crates_io`.
- `leo_needs_upgrade` — do step 4. LEO_CUR=`leo_current`,
  LEO_NEW=`leo_latest`, LEO_CUR_REV=`leo_current_rev`,
  LEO_REV=`leo_latest_rev`.
- `branch` — the branch to push in step 8. Use it verbatim; it already
  encodes which pins moved.

Skip whichever step's flag is false, and skip that package's edits entirely —
a leo-only run must leave `sdk/` untouched, and vice versa.

Also note `devnode_latest`/`devnode_snarkvm`: which snarkvm the newest
aleo-devnode release pins. Purely informational — the SDK carries no devnode
pin (the binary comes from PATH and `-m devnode` tests skip in CI) — but the
PR body must say whether the devnode matches $NEW (step 8). The devnode often
tracks `testnet-v*` tags, so a mismatch there is normal.

## 2. Study the snarkVM changeset (only if snarkvm_needs_upgrade=true)

Use an existing local snarkVM clone if one is available (fetch its tags
first); otherwise clone to a scratch dir:

```bash
git clone --filter=blob:none https://github.com/ProvableHQ/snarkVM /tmp/snarkvm-upgrade-src
cd /tmp/snarkvm-upgrade-src
git log --oneline $CUR..$NEW
git diff --stat $CUR..$NEW -- console/ circuit/ synthesizer/ ledger/ algorithms/ utilities/ parameters/
```

Diff in detail the areas the SDK binds (what `sdk/src/*.rs` and
`sdk-abi/src/*.rs` `use`): console types (Address, PrivateKey, ViewKey,
Signature, Plaintext, Record, Literal), ledger (Block, Transaction,
Transition, queries), synthesizer (Process, Program, Authorization,
execution/deployment), and anything ARC-20/ARC-22-relevant. Write a short
breaking-change inventory (API renames, signature changes, semantic changes)
BEFORE editing anything.

## 3. Bump the snarkvm pin in sdk/ (only if snarkvm_needs_upgrade=true)

In `sdk/Cargo.toml`, replace the `snarkvm = {` line. CRATES_IO=true means
crates.io publishes a release whose version equals $NEW; false means the tag
is ahead of the registry, so the git tag is the only way to get $NEW.

- If CRATES_IO=true — switch to the crates.io form, keeping the feature list
  verbatim:
  ```toml
  snarkvm = { version = "X.Y.Z", default-features = false, features = [
      "console", "circuit", "synthesizer", "ledger", "utilities", "algorithms", "parameters",
  ] }
  ```
- If CRATES_IO=false — keep the git form, only changing the tag:
  ```toml
  snarkvm = { git = "https://github.com/ProvableHQ/snarkVM.git", tag = "vX.Y.Z", default-features = false, features = [ ... ] }
  ```

`sdk/Cargo.lock` refreshes on the next build — commit it with the change.

## 4. Bump leo in sdk-abi/ (only if leo_needs_upgrade=true)

sdk-abi tracks leo's releases. Leo tags per crate
(`leo-abi-vX.Y.Z`, `leo-lang-vX.Y.Z`, …) and one release tags several crates
at a single rev — but not every crate every time, so don't trust any single
crate's tag list. The pre-check already resolved the newest release for you:
LEO_NEW is the version, LEO_REV the rev its tags point at.

Study what changed first — sdk-abi binds leo's ABI surface, which moves more
than snarkVM's:

```bash
git clone --filter=blob:none https://github.com/ProvableHQ/leo /tmp/leo-upgrade-src
cd /tmp/leo-upgrade-src
git log --oneline $LEO_CUR_REV..$LEO_REV -- crates/abi/ crates/disassembler/ crates/ast/
git diff $LEO_CUR_REV..$LEO_REV -- crates/abi/src crates/disassembler/src
```

Then in `sdk-abi/Cargo.toml`:

1. Update EVERY leo crate's `rev = "..."` to LEO_REV — all four must move
   together (`leo-abi`, `leo-ast`, `leo-disassembler`, `leo-span`); a partial
   bump gives you two leo versions in one graph and unresolvable type errors.
2. Read leo's snarkvm dep line **at LEO_REV** and mirror it verbatim:

```bash
curl -fsS "https://raw.githubusercontent.com/ProvableHQ/leo/$LEO_REV/Cargo.toml" | grep -m1 '^snarkvm '
```

Copy its source form, version/tag, and feature list into sdk-abi's `snarkvm`
line exactly — see the hard rule above on why this is not a choice. If that
version differs from `sdk/`'s, say so in the PR body; the skew is expected.

## 5. Adapt the code

Fix `sdk/src/` (if step 3 ran) and `sdk-abi/src/` (if step 4 ran), guided by
the step-2 and step-4 inventories. Start with `cargo check` in each affected
package; compile errors first, then semantics. For sdk-abi, a wall of
`Process<N>`/`Program<N>` mismatches usually means the snarkvm source didn't
match leo's — re-check that before touching any code.

## 6. Validate (native loop, escalating — all from `sdk/`)

Run the whole loop regardless of which pin moved: sdk-abi ships an import
hook into the main package (`sdk/python/tests/test_abi_hook.py`), so a
leo-only upgrade can still break `sdk/`'s tests. The cost is cache-warm
rebuilds, not correctness risk.

Interactively, work inside `sdk/.env` (`source .env/bin/activate`); in CI use
the runner's python directly.

```bash
cd sdk
cargo fmt --check
cargo clippy --no-default-features --features mainnet -- -D warnings
cargo check --no-default-features --features testnet

pip install maturin
maturin build --release --features mainnet --out dist
sed -i.bak 's/module-name = "aleo._aleolib_mainnet"/module-name = "aleo._aleolib_testnet"/' pyproject.toml
maturin build --release --no-default-features --features testnet --out dist-testnet
mv pyproject.toml.bak pyproject.toml
python ../.github/scripts/merge_testnet_so.py dist dist-testnet
pip install --force-reinstall dist/aleo_sdk-*.whl
pip install pytest pytest-xdist httpx pynacl responses pytest-asyncio requests

python -m pytest python/tests -v -n auto -m "not slow and not live and not devnode"
python -m pytest python/tests -v -m slow
python -m pytest python/tests/test_testnet.py -v
```

Then the abi package (run these even if sdk-abi was held back — they prove
the version skew is harmless):

```bash
cd ..
maturin build --release --manifest-path sdk-abi/Cargo.toml --out dist
pip install --force-reinstall dist/aleo_contract_abi_generator-*.whl
( cd sdk-abi && python -m pytest python/tests -v )
( cd sdk && python -m pytest python/tests/test_abi_hook.py -v )
# Build-graph isolation canary — leo feature flags must not leak into sdk:
test "$(grep -c dev_skip_checks sdk/Cargo.lock || true)" = "0"
# Source-identity canary — one snarkvm copy, or leo's types won't unify:
test "$(grep -c '^name = "snarkvm"' sdk-abi/Cargo.lock || true)" = "1"
```

On any failure: diagnose against the changeset inventory, fix, re-run the
failing stage, then re-run the full loop once everything passes.

## 7. Bump PyPI package versions

For each of `aleo-sdk`, `aleo-contract-abi-generator`, `shield-swap-sdk`:

```bash
curl -fsS https://pypi.org/pypi/<pkg>/json | python3 -c "import json,sys; print(json.load(sys.stdin)['info']['version'])"
```

New version = live version with patch incremented (e.g. 0.3.0 → 0.3.1). If
any query fails, STOP the version-bump step and say so — never guess.
Apply to:
- `sdk/pyproject.toml` + `sdk/Cargo.toml` `[package] version` (aleo-sdk)
- `sdk-abi/pyproject.toml` + `sdk-abi/Cargo.toml` `[package] version`
- `shield-swap-sdk/pyproject.toml`

Rebuild lockfiles by running `cargo check` in `sdk/` and `sdk-abi/`; commit
lockfile changes.

## 8. PR

Use `branch` from step 1 verbatim — it already names what moved
(`deps-upgrade/snarkvm-v4.9.0`, `deps-upgrade/leo-v4.3.4`, or both joined).
Stage explicit paths, never `git add -A`: the validation loop leaves build
artifacts (`dist/`, `sdk/dist*/`) that are not gitignored.

```bash
git checkout -B "$branch"   # -B resets a stale branch with no open PR
git add sdk/Cargo.toml sdk/Cargo.lock sdk/pyproject.toml \
        sdk-abi/Cargo.toml sdk-abi/Cargo.lock sdk-abi/pyproject.toml \
        shield-swap-sdk/pyproject.toml
git add -u sdk/src sdk-abi/src   # any code adaptations from step 5
git commit -m "<title>"          # e.g. "deps: upgrade snarkvm to v4.9.0, leo to v4.3.4"
git push -u origin "$branch" --force-with-lease
```

Title names only the pins that actually moved. Body must cover:

- the breaking-change inventory per upgraded dep, and what was adapted
- `sdk/` dep-form decision (crates.io vs git tag, and why)
- `sdk-abi/`: leo rev old→new, and the snarkvm line mirrored from leo —
  state explicitly when it differs from `sdk/`'s, e.g. "sdk-abi stays on
  snarkvm 4.8.1 because leo v4.3.4 pins it; sdk/ is on 4.9.0. Expected —
  separate build graphs."
- validation evidence: which suites ran, pass counts, and both canaries
- the PyPI version bumps
- devnode: "aleo-devnode <devnode_latest> pins snarkvm <devnode_snarkvm> —
  matches $NEW" or "— differs from $NEW: local `-m devnode` e2e tests may
  skip on fee/consensus skew until a devnode release adopts it" (the devnode
  frequently tracks `testnet-v*`, so a mismatch is usually not actionable)

Then `gh pr create --title "<title>" --body ...`.
Genuinely stuck after repeated fix attempts → same, but `gh pr create
--draft`, with the body additionally listing exact failing tests, output
excerpts, and what was attempted.
