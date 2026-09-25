# First Shield Swap in Python

Create a testnet account, request tokens, trade 1.5 USDCx for ETH, and claim
the output. [swap.py](./swap.py) calls the Python SDK directly and uses its
in-memory account and swap handle.

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

`SHIELD_SWAP_PRIVATE_KEY` optionally supplies an existing testnet account.
Otherwise the example generates a private key in memory. `ShieldSwap(aleo)`
uses the configured Aleo client without creating a profile or swap journal.

`dex.api.authenticate()` signs the API challenge with the account's key.
`request_airdrop()` starts the testnet faucet job; `get_airdrop_job()` polls it
until completion. `funding.results` contains each token's outcome and transaction
ID. Job completion does not guarantee every token transfer succeeded.

The faucet allows one request per address per 15 minutes. A rate-limit error
stops the example; inspect the existing funding before requesting again.
The example waits for the scanner to report at least 1.5 USDCx before trading.
It quotes a direct USDCx/ETH pool with `get_route()`, then calls
`swap(...).delegate(wait=True)` with that quote and a 0.5% slippage limit.
The SDK selects a token record and returns the handle needed to claim.
One unspent record must cover 1.5 USDCx.

## Completion and recovery

After the swap confirms, `claim_swap_output(handle).delegate(wait=True)`
submits one claim and waits for confirmation. `claim.transaction_id` identifies
the claim and `claim.amount_out` contains the received ETH in base units.
The example writes no console logs or local account files.

The private key and swap handle remain in memory. **Retain both before ending
an interrupted session.** The handle contains the blinding information needed
to claim; `handle.to_json()` serializes it. Neither value is saved automatically.
A newly generated key is lost when the process exits unless retained separately.

Each run submits a new trade. To resume a pending claim, recreate the client
with the same private key, restore the retained handle with
`SwapHandle.from_json(...)`, and claim after confirming the swap transaction.
Do not rerun the whole script to recover a swap.

For automatic account storage and swap journaling, use
`ShieldSwap.from_profile(network="testnet")` instead of constructing
`ShieldSwap(aleo)`. That optional path stores the profile under `~/.shield-swap`;
it is not required for `swap()` or `claim_swap_output()`.
