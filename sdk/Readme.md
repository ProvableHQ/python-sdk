# Aleo Python SDK (MainnetV0)

The Aleo Python SDK provides Python bindings to Aleo's zero-knowledge cryptographic primitives, built with snarkvm 4.8.1.

## Quick Start

```python
from aleo import Aleo

# Connect to mainnet
aleo = Aleo(Aleo.HTTPProvider("https://api.provable.com/v2"))

print(aleo.is_connected())   # True
print(aleo.network_name)     # "mainnet"
```

### Create or import an account

```python
# Create a fresh random account
account = aleo.account.create()
print(account.address)       # aleo1…
print(account.private_key)   # APrivateKey1zkp…

# Import from an existing private key string
account = aleo.account.from_private_key("APrivateKey1zkp...")
```

### Check balances and read mappings

```python
# Public balance in microcredits (1 credit == 1_000_000 microcredits)
balance = aleo.get_balance(str(account.address))
print(aleo.from_microcredits(balance), "credits")

# Read any on-chain mapping
credits_prog = aleo.programs.get("credits.aleo")
raw = credits_prog.mapping("account").get(str(account.address))
print(raw)    # e.g. "5000000u64"
```

### Sign and verify

```python
# Sign raw bytes
message = b"hello aleo"
signature = aleo.account.sign(message, account)
print(str(signature))    # sign1…

# Verify (address as string or Address object)
ok = aleo.account.verify(str(account.address), message, signature)
assert ok
```

### Call a program — the verb ladder

```python
# Fetch a live program
credits = aleo.programs.get("credits.aleo")

# Build a call (pure coercion — no network, no proof)
call = credits.functions.transfer_public(
    "aleo1recipient...",   # address (passed through)
    1_000_000,             # u64 (auto-coerced to "1000000u64")
)
print(call.signature)    # "transfer_public(address, u64)"

# Inspect outputs before proving (local, no proof, no network)
auth_result = call.simulate(account)
print(auth_result.decoded())

# Build + broadcast (proves locally; requires a live node + funded account)
# requires a live node + funded key
tx_id = call.transact(account)
confirmed = aleo.network.wait_for_transaction(tx_id)
```

### Delegate — the flagship frictionless path

```python
# The prover's fee master pays — no credits needed on your account.
# requires prover credentials configured on the DPS; fee master pays
result = aleo.programs.get("my_app.aleo") \
              .functions.my_function("arg", 42) \
              .delegate(account)
```

No fee record. No public balance. The Delegated Proving Service handles proving and fee payment.

### Async client

```python
import asyncio
from aleo import AsyncAleo

async def main():
    aleo = AsyncAleo(AsyncAleo.HTTPProvider("https://api.provable.com/v2"))
    connected = await aleo.is_connected()
    balance = await aleo.get_balance("aleo1...")
    print(connected, balance)

asyncio.run(main())
```

### Low-level primitives

For direct access to `PrivateKey`, `Signature`, `Program`, and the raw network client:

```python
from aleo.mainnet import PrivateKey, Signature, Account

pk = PrivateKey.random()
acct = Account.from_private_key(pk)
message = b"hello"
sig = Signature.sign(pk, message)
assert sig.verify(acct.address, message)
```

## Build & Install

```bash
bash install.sh
```

This creates a development environment and installs the `aleo` package with MainnetV0 bindings.

## Contributing
If you wish to contribute, please follow the contribution guidelines outlined on [GitHub](https://github.com/AleoHQ/python-sdk/blob/master/sdk/CONTRIBUTING.md).
