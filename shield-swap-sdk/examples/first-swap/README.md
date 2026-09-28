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
The example stops if initial scanner registration fails. If a later records
query returns HTTP 422, the SDK re-registers the account and retries that query
once. A failed re-registration surfaces its error.

`dex.api.authenticate()` signs the API challenge with the account's key.
`has_swap_balance(source.address, "1.5")` checks for a covering private USDCx
record first. An account with enough funds skips the airdrop. Scanner errors
stop the example rather than triggering funding. Otherwise,
`shield_swap_client.confirm_airdrop()` requests tokens, waits for the faucet
job to settle, then waits for decrypted, unspent token records from the
airdrop's transaction IDs. Older records do not satisfy this check.
It returns `funding.status == "settled"` with per-token outcomes in
`funding.job.results`, or `"rate_limited"` with the faucet's explanation in
`funding.message`. A settled job can contain failed token transfers.
The example checks `funding.success` and raises `funding.error` on failure.
Success requires every transfer accepted and its records available; the error
includes the rate-limit reason or unsuccessful transfers and their transaction IDs.

The helper polls every 5 seconds; its default 10-minute timeout covers both
settlement and record scanning. If scanning times out, `funding.success` is
false and `funding.error` lists the pending transaction IDs. The lower-level
`api.confirm_airdrop(address)` only waits for faucet settlement.
`AirdropPendingError.job_id` identifies a timed-out job for further status reads.
The example stops on a rate-limit response, missing token results, or any
transfer that was not accepted on chain. For pending transfers, inspect the
reported transaction before requesting another airdrop. API errors and
confirmation timeouts also stop the example before it submits a swap.
`quote(token_in="USDCx", token_out="ETH", amount_in="1.5", slippage_bps=50)`
resolves the symbols and asks the API for its best route. The result includes
`estimated_amount_out`, `minimum_amount_out`, and ordered `hops`. Amounts on the
quote are decimal strings in token units. The slippage floor applies once to
the final output, including for two- and three-hop routes.

`swap(quote).delegate(wait=True)` executes that route in one transaction. The
SDK checks its network, token path, and live pool directions before preparing
it. A quote reserves no records and does not guarantee its estimated price;
refresh the quote before a later submission if prices have changed.
The existing `swap(pool_key=..., token_in_id=..., amount_in=...)` form remains
available. Its integer amounts use base units; strings/Decimal use token units.

The SDK selects a token record and returns the handle needed to claim.
One unspent record must cover 1.5 USDCx. If the scanner does not return a
covering record, preparation raises `InsufficientRecordsError` before proving
or submitting a swap. Allow newly funded records to become available before
trying again. Scanner errors propagate immediately.

## Completion and recovery

After the swap confirms, `claim_swap_output(handle, timeout=5)`
waits up to five seconds for its output mapping, polling missing-output reads
every two seconds. Then `.delegate(wait=True)` submits one claim and waits
for confirmation. Timeout stops before submission; preserve the handle and
inspect the existing swap instead of rerunning the example. Other read errors
propagate immediately. `claim.transaction_id` identifies
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
handles there and automatically records the claim after confirmation.
`delegate(wait=True)` and `transact(wait=True)` update the journal; non-waiting
submissions do not mark a claim confirmed. Keep the file
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
