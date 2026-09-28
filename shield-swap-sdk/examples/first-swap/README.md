# First Shield Swap in Python

Create a testnet account, fund it when needed, swap 1.5 USDCx for ETH, and
claim the private output. The complete runnable example is
[swap.py](./swap.py); the sections below follow that script in order.

## Run the example

From a Python SDK checkout:

```sh
python -m pip install ./shield-swap-sdk
python shield-swap-sdk/examples/first-swap/swap.py
```

An installed release that contains the example also supports:

```sh
python -m aleo_shield_swap.examples.first_swap.swap
```

## 1. Create an account

Start with an existing testnet private key or let the SDK create and retain one
in `~/.shield-swap/profile.json`.

```python
import os

from aleo import Aleo, HTTPProvider, testnet
from aleo_shield_swap import Journal, Profile, ShieldSwap


ENABLE_JOURNAL = False

key = os.environ.get("SHIELD_SWAP_PRIVATE_KEY")
if not key:
    profile = Profile.load_or_create(network="testnet")
    if profile.network != "testnet":
        raise RuntimeError("This example requires a testnet profile")
    key = profile.private_key

aleo = Aleo(HTTPProvider("https://edge.provable.com/api", network="testnet"))
private_key = testnet.PrivateKey.from_string(key)
account = aleo.account.from_private_key(private_key)
address = str(account.address)
```

Register the account so the scanner can find its records. The SDK decrypts the
returned records locally. A registration failure stops the example before any
funding or swap submission.

```python
registration = aleo.records.register(account)
if not registration["ok"]:
    raise RuntimeError(f"Record scanner registration failed: {registration['error']}")
```

## 2. Create and fund the client

Continue with the account above. Authentication authorizes Shield Swap API
requests without sending the private key. The optional journal retains swap
handles for recovery.

```python
shield_swap_client = ShieldSwap(aleo)
if ENABLE_JOURNAL:
    shield_swap_client.journal = Journal(f"testnet-{address}.jsonl")

shield_swap_client.api.authenticate(
    address,
    lambda message: str(private_key.sign(message.encode())),
)
```

Use an existing private USDCx record when one can cover `1.5 USDCx`. Otherwise,
request testnet tokens and wait until the resulting records are spendable.
`funding.error` reports a rate limit, failed transfer, or record timeout. The
example does not submit a swap when funding fails.

```python
source = shield_swap_client.api.get_token("USDCx")
amount_in = "1.5"
if not shield_swap_client.has_swap_balance(source.address, amount_in):
    funding = shield_swap_client.confirm_airdrop()
    if not funding.success:
        raise RuntimeError(funding.error)
```

The decimal string expresses token units. It avoids floating-point rounding;
`"1.5"` means 1.5 USDCx.

## 3. Quote and submit the swap

Request the best USDCx-to-ETH route. The route can contain intermediate pools,
and the 0.5% slippage limit applies once to its final output.

```python
quote = shield_swap_client.quote(
    token_in="USDCx",
    token_out="ETH",
    amount_in=amount_in,
    slippage_bps=50,
)
```

Submit that exact quote and wait for confirmation. This is the first on-chain
write in the swap flow and can pay a fee. The returned handle contains the
blinding information required to claim the output.

```python
handle = shield_swap_client.swap(quote).delegate(wait=True)
```

## 4. Claim the output

After the swap confirms, wait up to five seconds for its output mapping to
become readable. A timeout occurs before claim submission, so retain the handle
and inspect the confirmed swap instead of submitting another trade.

```python
claim = shield_swap_client.claim_swap_output(
    handle,
    timeout=5,
).delegate(wait=True)

if claim.amount_out <= 0:
    raise RuntimeError("The claim returned no ETH")
```

`delegate(wait=True)` submits one claim and waits for confirmation. The claimed
ETH arrives as a private record owned by the account.

## Recovery

With `ENABLE_JOURNAL = False`, the handle exists only in memory. Set it to
`True` in [swap.py](./swap.py) to save handles and confirmed claims in
`testnet-<account-address>.jsonl`.

Keep the handle or journal until the claim confirms. A confirmation or output
timeout does not show that the swap failed. Recover the existing swap; running
the entire example again submits a new trade.
