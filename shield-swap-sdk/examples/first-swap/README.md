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

Installation requires `aleo-sdk>=0.5.1` with testnet bindings. If no compatible
wheel is available, follow the `sdk/` package's build instructions first.

Releases containing the example also support:

```bash
python -m aleo_shield_swap.examples.first_swap.swap
```

## Account and funding

`SHIELD_SWAP_PRIVATE_KEY` optionally supplies an existing testnet account.
Otherwise the example generates a private key in memory. `ShieldSwap(aleo)`
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
One unspent record must cover 1.5 USDCx. If the scanner has not indexed a
covering record yet, preparation raises `InsufficientRecordsError` before
proving or submitting a swap.

## Completion and recovery

After the swap confirms, `claim_swap_output(handle).delegate(wait=True)`
submits one claim and waits for confirmation. `claim.transaction_id` identifies
the claim and `claim.amount_out` contains the received ETH in base units.
The example writes no console logs. Local account storage is disabled by default.

With journaling disabled, the private key and swap handle remain in memory. **Retain both before ending
an interrupted session.** The handle contains the blinding information needed
to claim; `handle.to_json()` serializes it. Neither value is saved automatically.
A newly generated key is lost when the process exits unless retained separately.

Each run submits a new trade. To resume a pending claim, recreate the client
with the same private key, restore the retained handle with
`SwapHandle.from_json(...)`, and claim after confirming the swap transaction.
Do not rerun the whole script to recover a swap.

Set `ENABLE_JOURNAL = True` in `swap.py` to attach the SDK's `Journal` at
`testnet-<account-address>.jsonl` in the working directory. The SDK retains swap
handles there, and the example records the confirmed claim. Keep the file
private: it contains the blinding information needed to claim.

The flag only controls journal attachment. It does not change the account,
client, network, or record scanning, and it does not save the private key.
Use the same private key and journal to recover a pending swap.
With `ENABLE_JOURNAL = False` (the default), no journal is created.
