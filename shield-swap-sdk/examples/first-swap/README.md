# First Shield Swap in Python

Create a testnet account, fund it when needed, swap 1.5 USDCx for ETH, and
claim the private output.

**Read [swap.py](./swap.py) for the complete flow.** Its comments explain each
step beside the code that performs it, including account setup, funding, quoting,
submission, and claiming.

## Run the example

From a Python SDK checkout:

```sh
python -m pip install ./shield-swap-sdk
python shield-swap-sdk/examples/first-swap/swap.py
```

From an installed release that includes the example:

```sh
python -m aleo_shield_swap.examples.first_swap.swap
```

`SHIELD_SWAP_PRIVATE_KEY` may provide an existing testnet account. Without it,
the example creates or reuses the SDK profile at `~/.shield-swap/profile.json`.
The script registers that account with the record scanner; records returned by
the scanner are decrypted locally by the SDK.

## What the script does

The script:

1. Creates or loads a testnet account.
2. Authenticates with Shield Swap.
3. Checks for enough private USDCx and requests an airdrop only when needed.
4. Quotes the best USDCx-to-ETH route, including multiple hops when beneficial.
5. Submits the quoted swap and waits for confirmation.
6. Waits for the output to become readable, then submits and confirms one claim.

Funding must report `funding.success` before the swap begins. A rate limit,
failed transfer, or record timeout raises the error returned by the funding
helper instead of submitting a swap.

## Recovery

`ENABLE_JOURNAL = False` keeps the swap handle in memory. Set it to `True` in
[swap.py](./swap.py) to save handles and confirmed claims in
`testnet-<account-address>.jsonl`.

Keep either the returned handle or the journal until the claim confirms. If
swap confirmation or output discovery times out, inspect and recover that
existing swap. Running the entire example again submits a new trade.
