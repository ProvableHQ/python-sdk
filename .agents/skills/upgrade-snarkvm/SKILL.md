---
name: upgrade-snarkvm
description: Use when upgrading the SDK's pinned snarkVM to the latest upstream release — compares tags, adapts sdk/ and sdk-abi/ to the changeset, runs the native validation loop, bumps PyPI package versions, and opens a PR. Invoked as /upgrade-snarkvm interactively or by the snarkvm-upgrade workflow.
---

# Upgrade snarkVM

Upgrade the pinned `snarkvm` dependency to the latest upstream release and
adapt the SDK. Same procedure interactively and in CI.

## Hard rules

- Plain commit messages — NO Co-Authored-By trailers, NO "Generated with"
  footers.
- Every version number comes from a command output. Never guess one.
- `sdk-abi`'s snarkvm dep is type-coupled to leo: it stays in git-tag form
  with the exact tag and feature set leo's pinned rev uses, or
  `Process<N>`/`Program<N>` in leo-disassembler's signatures become distinct
  types. Never give sdk-abi a crates.io-form snarkvm dep.
- Validate seriously before giving up: on failures, re-read the changeset,
  fix, and re-run. Only after several genuine fix attempts do you fall back
  to the draft-PR path in step 8.

## 1. Detect

```bash
bash .github/scripts/snarkvm_check.sh
```

If `needs_upgrade=false`, report why (already at `latest`, or an open PR
exists for it — check `gh pr list --head "snarkvm-upgrade/<latest>"`) and
STOP. Otherwise set CUR=<current>, NEW=<latest>, CRATES_IO=<crates_io> and
continue.

## 2. Study the changeset

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

## 3. Bump the pin in sdk/

In `sdk/Cargo.toml`, replace the `snarkvm = {` line:

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

## 4. Bump the pin in sdk-abi/ (leo-coupled)

Check whether leo's default branch has picked up $NEW:

```bash
curl -s https://raw.githubusercontent.com/ProvableHQ/leo/HEAD/Cargo.toml | grep -m1 '^snarkvm '
```

- If leo's pin resolves to $NEW — either `tag = "$NEW"` or crates.io
  `version = "X.Y.Z"` matching $NEW without the `v` (leo migrated to the
  crates.io form around v4.8.1): get the rev
  (`git ls-remote https://github.com/ProvableHQ/leo HEAD`), then in
  `sdk-abi/Cargo.toml` update EVERY leo crate's `rev = "..."` to that rev
  and **mirror leo's snarkvm dep line exactly** — same form (tag or
  version), same feature list. Type-coupling means matching leo's source,
  whatever shape it takes.
- If leo still pins an older snarkvm: leave `sdk-abi/` entirely untouched
  (sdk and sdk-abi are separate build graphs; version skew is fine) and
  record "sdk-abi held back at $CUR — leo hasn't adopted $NEW" for the PR
  body.

## 5. Adapt the code

Fix `sdk/src/` (and `sdk-abi/src/` if bumped) guided by the step-2
inventory. Start with `cargo check`; compile errors first, then semantics.

## 6. Validate (native loop, escalating — all from `sdk/`)

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

```bash
git checkout -b "snarkvm-upgrade/$NEW"   # or reset it if a stale branch exists without an open PR
git add -A ':!docs/superpowers' && git commit -m "deps: upgrade snarkvm to $NEW"
git push -u origin "snarkvm-upgrade/$NEW" --force-with-lease
```

- All green → `gh pr create --title "deps: upgrade snarkvm to $NEW" --body ...`
  with: the breaking-change inventory, what was adapted, dep-form decision
  (crates.io vs git, sdk-abi held back or not), validation evidence (which
  suites ran, pass counts), and the version bumps.
- Genuinely stuck after repeated fix attempts → same but
  `gh pr create --draft`, body additionally lists exact failing tests with
  output excerpts and what was attempted.
