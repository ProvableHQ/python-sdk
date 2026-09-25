# First Shield Swap in Python

Create a testnet account, request tokens, trade 1.5 USDCx for ETH, and claim
the output. [swap.py](./swap.py) calls the Python SDK directly and uses its
existing profile and journal.

## Run

Use Python 3.10 or later on macOS or Linux. From a Python SDK checkout:

```bash
python -m pip install ./shield-swap-sdk
python shield-swap-sdk/examples/first-swap/swap.py
```

Installation requires `aleo-sdk>=0.5.0` with testnet bindings. If no compatible
wheel is available, follow the `sdk/` package's build instructions first.

Releases containing the example also support:

```bash
python -m aleo_shield_swap.examples.first_swap.swap
```

## Account and funding

`ShieldSwap.from_profile(network="testnet")` creates or loads the SDK profile
at its default location, `~/.shield-swap/`. The SDK retains the account and
swap journal there. The example does not create a separate storage layout.

To import an existing account into a new profile, set `SHIELD_SWAP_PRIVATE_KEY`
before the first run. An existing profile keeps its saved account and network;
the example stops if that network is mainnet. Keep the profile private and
retain it for recovery.

`dex.api.authenticate()` signs the API challenge with the profile's account.
`request_airdrop()` starts the testnet faucet job; `get_airdrop_job()` polls it
until completion. `funding.results` contains each token's outcome and transaction
ID. Job completion does not guarantee every token transfer succeeded.

The faucet allows one request per address per 15 minutes. A rate-limit error
stops the example; inspect the existing funding before requesting again.
The example waits for the scanner to report at least 1.5 USDCx before trading.
It quotes a direct USDCx/ETH pool with `get_route()`, then calls
`swap(...).delegate(wait=True)` with that quote and a 0.5% slippage limit.
The SDK selects a token record and saves the swap handle in its journal.
One unspent record must cover 1.5 USDCx.

## Completion and recovery

After the swap confirms, `claim_swap_output(handle).delegate(wait=True)`
submits one claim and waits for confirmation. The example records the claim
in the SDK journal. `claim.transaction_id` identifies the claim and
`claim.amount_out` contains the received ETH in base units. The example writes no console logs
or additional result files.

Each run submits a new trade. To recover an interrupted run, load the same
profile and collect pending outputs instead of rerunning the swap:

```python
from aleo_shield_swap import ShieldSwap

dex = ShieldSwap.from_profile()
claims = dex.collect_all()
```

`claims.still_pending` lists swaps whose outputs remain pending. Check the
journal and transaction status before submitting another trade. `collect_all`
also collects owed fees from journaled liquidity positions.
