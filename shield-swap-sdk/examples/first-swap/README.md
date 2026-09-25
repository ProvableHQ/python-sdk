# First Shield Swap in Python

Create an account, request test tokens, trade 1.5 USDCx for ETH, and collect
the purchased ETH. This example uses the SDK's persistent profile and swap
journal. It runs only on testnet and requires a direct USDCx/ETH pool.

## Run from the SDK checkout

Use Python 3.10 or later on macOS or Linux (the SDK journal uses `fcntl`).
From the root of a checkout containing this example:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ./shield-swap-sdk
python shield-swap-sdk/examples/first-swap/swap.py
```

Installation requires `aleo-sdk>=0.5.0` with the testnet bindings. If a
compatible wheel is unavailable for the platform, build this checkout's
`sdk/` package using its build instructions, then install `shield-swap-sdk`.
The script calls `ShieldSwap.from_profile`, `onboard`, `swap_many(count=1)`,
and `collect_all`; the SDK handles authentication, funding, quoting,
record selection, delegated proving, and recovery handles.

The run can take several minutes. It writes no console logs. Exit code zero
means a positive output was recorded by a confirmed claim in the SDK journal.
Inspect `.shield-first-swap/result.json` in the invoking directory
for the swap and claim transaction IDs and the exact received ETH amount.
The result also retains base units and decimals for machine consumers.
A nonzero exit leaves a sanitized `error.json`; the journal contains further
local diagnostic information. A concurrent run exits nonzero without changing
another run's error file.

## Run from an installed release

The wheel also includes the same source and README. After installing a release
containing this example, run it from a private working directory:

```bash
python -m aleo_shield_swap.examples.first_swap.swap
python -m aleo_shield_swap.examples.first_swap.swap --claim
```

Both commands use `.shield-first-swap/` in the current directory. Run recovery
from that same directory. Source-checkout commands use the same state location.

## Account and recovery

On the first run, the SDK generates an account and saves it under `.shield-first-swap/` in the invoking directory. To import an account instead, set the optional
`SHIELD_SWAP_PRIVATE_KEY` in the invoking shell before the first run.
The SDK stores imported keys in the profile too. An existing profile always
wins over this variable; use a separate working directory for another account.
No `.env` file is loaded. The example takes no other environment configuration;
SDK API, key-file, and onboarding credential overrides are disabled.

Keep `.shield-first-swap/` private and retain its profile and journal: they contain the
private key, credentials, and secrets needed to collect purchased tokens.
The script creates owner-only state under a restrictive umask. The directory contains a `.gitignore` excluding all its contents; never force-add it to source control. Hosted scanning shares the view key with the scanner,
which can decrypt account records. Delegated proving shares transaction
authorizations with the prover, without sharing the private key.

Before submitting, the example saves `submission.json`. Any later normal run
refuses another trade, including after a lost submission response. Recover with:

```bash
python shield-swap-sdk/examples/first-swap/swap.py --claim
```

Recovery reuses the saved profile and journal; it neither requests funding nor
submits another swap. It also reconstructs `result.json` if the process stopped
after a successful claim. Do not delete the submission marker to retry.
An empty journal or missing claim is not proof that submission failed. A crash
between broadcast and journal persistence, an incomplete handle, or a rejected
transaction may require manual SDK/chain inspection. Preserve all state.
The process lock releases automatically when the process exits.

A funding failure before submission can be retried normally. The balance scan
may lag behind faucet delivery; one unspent record must cover 1.5 USDCx even
when several smaller records add up to enough. Missing pools or route quotes
stop the run; the example never relaxes slippage to force a swap.

## Offline checks

```bash
cd shield-swap-sdk/examples/first-swap
python -m unittest -v test_swap
```

These checks use no network, installed SDK, account, or funds. They exercise
lost-response protection, recovery after a confirmed claim, absent handles,
existing journals, and testnet enforcement. They do not establish live service
availability; only a completed live run verifies the funding-to-claim journey.
