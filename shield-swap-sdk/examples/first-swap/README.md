# First Shield Swap in Python

Create a testnet account, request tokens, trade 1.5 USDCx for ETH, and claim
the output. [swap.py](./swap.py) calls the Python SDK directly and uses its
account profile and swap handle.

## Run

Use Python 3.10 or later on macOS or Linux. From a Python SDK checkout:

```bash
python -m pip install ./shield-swap-sdk
python shield-swap-sdk/examples/first-swap/swap.py
```

Installation requires `aleo-sdk>=0.5.1` with testnet bindings. If no compatible
wheel is available, follow the `sdk/` package's build instructions first.

Releases containing the example also support:

```bash
python -m aleo_shield_swap.examples.first_swap.swap
```

## Account and funding

`SHIELD_SWAP_PRIVATE_KEY` optionally supplies an existing testnet account.
Otherwise `Profile.load_or_create()` saves a generated key in the SDK’s default
profile (`~/.shield-swap/profile.json`, owner-only) and reuses it on later runs.
An existing profile must use testnet. `ShieldSwap(aleo)`
uses the configured Aleo client. Both journal settings use the same account
and register it with the record scanner. The first `from_private_key()` call
sets the default account; later imports preserve an existing default.

`dex.api.authenticate()` signs the API challenge with the account's key.
`confirm_airdrop()` requests tokens and waits for the faucet job to settle.
It returns `funding.status == "settled"` with per-token outcomes in
`funding.job.results`, or `"rate_limited"` with the faucet's explanation in
`funding.message`. A settled job can contain failed token transfers.

The helper polls every 5 seconds and times out after 10 minutes by default.
`AirdropPendingError.job_id` identifies a timed-out job for further status reads.
A rate-limited account can continue if it already holds enough USDCx.
It looks up USDCx and ETH with `get_token(symbol)`, quotes a direct pool
with `get_route()`, then calls
`swap(...).delegate(wait=True)` with that quote and a 0.5% slippage limit.
The example passes `amount_in="1.5"` and the quote's decimal output directly
to `swap()`. The SDK converts both using token metadata; no unit conversion
is needed in the example. Strings and `Decimal` values represent token units;
integers retain their existing base-unit meaning. Excess precision is rejected.

The SDK selects a token record and returns the handle needed to claim.
One unspent record must cover 1.5 USDCx. The example sets `record_wait_seconds=120` to let the SDK poll for a covering
record every five seconds. If none appears within two minutes, preparation
raises `InsufficientRecordsError` before proving or submitting a swap. Scanner
errors propagate immediately. Other callers default to no wait.

## Completion and recovery

After the swap confirms, `claim_swap_output(handle).delegate(wait=True)`
submits one claim and waits for confirmation. `claim.transaction_id` identifies
the claim and `claim.amount_out` contains the received ETH in base units.
The example writes no console logs. Generated accounts are saved regardless
of the journal setting.

With journaling disabled, the swap handle remains in memory. Retain it with
`handle.to_json()` for later claims. The handle contains the blinding information
needed to claim. Enable the journal to retain that information if confirmation
times out or the process exits before the handle returns. An account supplied
through `SHIELD_SWAP_PRIVATE_KEY` remains the caller’s responsibility.

Each run submits a new trade. To resume a pending claim, recreate the client
with the same private key, restore the retained handle with
`SwapHandle.from_json(...)`, and claim after confirming the swap transaction.
Do not rerun the whole script to recover a swap.

Set `ENABLE_JOURNAL = True` in `swap.py` to attach the SDK's `Journal` at
`testnet-<account-address>.jsonl` in the working directory. The SDK retains swap
handles there, and the example records the confirmed claim. Keep the file
private: it contains the blinding information needed to claim.

The flag only controls journal attachment. It does not change the account,
client, network, record scanning, or profile persistence.
Use the same private key and journal to recover a pending swap.
With `ENABLE_JOURNAL = False` (the default), no journal is created.

The journal saves a swap handle as soon as delegated submission returns, before
waiting for confirmation. If the service returns only a transaction ID, the
first entry contains that ID and the claim secrets; a later entry adds the
swap ID after confirmation. If confirmation times out, retain the journal and
inspect the transaction instead of submitting again. An entry without a swap
ID is excluded from `pending_claims()` until recovered: read the confirmed
transaction’s swap transition, copy its public swap ID into the saved handle,
and record the completed handle with `Journal.record_swap(handle, counter)`
using the original entry’s counter. Confirm success before claiming.
