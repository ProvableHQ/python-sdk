# Aleo Python SDK (MainnetV0)

The Aleo Python SDK provides Python bindings to Aleo's zero-knowledge cryptographic primitives, built with snarkvm 4.8.1.

It ships two layers:

- A **Web3.py-style facade** (`aleo.Aleo` / `aleo.AsyncAleo`) — a high-level, batteries-included client for connecting to a node, managing accounts, reading state, and building/proving/broadcasting transactions.
- **Low-level primitives** (`aleo.mainnet`, `aleo.testnet`) — direct Python bindings to Aleo's cryptographic types, for when you need full control.

## Quick Start (facade)

```python
from aleo import Aleo

# Connect (construction is offline — no I/O until you make a call)
aleo = Aleo(Aleo.HTTPProvider("https://api.provable.com/v2"))
print(aleo.network_name)   # "mainnet"
print(aleo.network_id)     # 0

# Create an account (local)
account = aleo.account.create()
print(account.address)     # aleo1…

# Read a public balance  # requires a live node
balance = aleo.get_balance(str(account.address))
print(aleo.from_microcredits(balance), "credits")
```

`Aleo.HTTPProvider` is also importable directly as `from aleo import HTTPProvider`; the two are equivalent.

## The verb ladder (sync)

The facade follows a clean top-to-bottom narrative: **connect → account → read → build a call → inspect → send**. Each rung does a little more than the one above it.

### 1. Connect

```python
from aleo import Aleo

aleo = Aleo(Aleo.HTTPProvider("https://api.provable.com/v2"))

# Optional: check reachability  # requires a live node
if aleo.is_connected():
    print("connected to", aleo.network_name)
```

`HTTPProvider` is a *config object*, not a live connection — it is safe to construct without a network. It accepts `network=` (`"mainnet"` / `"testnet"`), `api_key=`, `prover_uri=` (the DPS endpoint), `headers=`, and a custom `transport=` callable.

### 2. Account — create or import (local)

```python
# Fresh random account
account = aleo.account.create()

# Import from a private-key string (or a PrivateKey object)
account = aleo.account.from_private_key("APrivateKey1zkp…")

# Derive deterministically from a field seed
account = aleo.account.from_seed("123field")

# Sign / verify (bytes)
sig = aleo.account.sign(b"hello aleo", account)
assert aleo.account.verify(str(account.address), b"hello aleo", sig)

# Sign / verify a typed Aleo value
sv = aleo.account.sign_value("100u64", account)
assert aleo.account.verify_value(str(account.address), "100u64", sv)
```

Set `aleo.default_account = account` and the verbs below will use it whenever you omit the signer.

> **Note:** the facade deliberately has no "recover signer from signature" verb. Aleo is a privacy chain — surfacing "which address signed this?" is a de-anonymisation vector. The low-level `Signature.to_address()` primitive remains available directly if you truly need it.

### 3. Read — balance and mappings

```python
# Public credits balance in microcredits (0 if the address is unfunded)  # requires a live node
micro = aleo.get_balance(str(account.address))
print(aleo.from_microcredits(micro), "credits")

# Read any on-chain mapping through a bound Program  # requires a live node
credits = aleo.programs.get("credits.aleo")
raw = credits.mapping("account").get(str(account.address))
print("account mapping value:", raw)
```

Unit helpers are local: `aleo.to_microcredits(1.5) == 1_500_000` and `aleo.from_microcredits(1_500_000) == 1.5`. Address validation is local too: `aleo.is_valid_address(s) -> bool`.

### 4. Build a call

`aleo.programs.get(...)` fetches a program and exposes its transitions as `program.functions.<name>`, mirroring web3.py's ABI-driven `contract.functions`. Calling one **coerces your Python arguments to Aleo values** and returns a `BoundCall`:

```python
# requires a live node (to fetch the program source)
credits = aleo.programs.get("credits.aleo")

call = credits.functions.transfer_public(str(account.address), 1_000_000)
print(call.signature)   # "transfer_public(address, u64)"
print(call.args)        # ['aleo1…', '1000000u64']  — coerced
```

You can also list/iterate the available functions: `list(credits.functions)`, `"transfer_public" in credits.functions`.

### 5. Inspect — `simulate` / `.decoded()`

Before proving or broadcasting anything, build the **authorization locally** and look at what the call will produce. This is a proof-free, network-free dry run:

```python
auth = call.simulate(account)          # alias of .authorize(); .call() also works
print(auth.outputs)                    # per-transition output lists
print(auth.decoded())                  # [{program, function, inputs, outputs}, …]
```

Both `AuthorizationResult` and `TransactionResult` expose the same `.outputs` / `.decoded()` surface, plus a `.raw` escape hatch to the underlying network object. You can also decode after the fact with `aleo.decode_transition(tx_id_or_transition)`.

### 6. Transact — full prove + broadcast

`build_transaction` (alias `prove`) runs the whole ladder locally: authorize → execute → prepare trace → prove execution → authorize+prove fee → assemble. `transact` does that **and** broadcasts, returning the transaction id:

```python
# requires a live node + funded private key
tx_id = credits.functions.transfer_public(
    str(account.address), 100
).transact(account)

confirmed = aleo.network.wait_for_transaction(tx_id, timeout=60.0)
```

Fees are **public by default** (base cost from the execution). Pass `priority_fee=` for a tip, or opt into a **private fee** with `private_fee=True` (auto-sourced from `aleo.record_provider`) or by passing an explicit `fee_record=`.

### 7. Delegate — the flagship path (fee master pays by default)

`delegate` hands proving to a **Delegated Proving Service (DPS)**: you build only the lightweight main authorization locally, and the DPS does the expensive SNARK proving. By default **the prover's fee master pays the fee** — no records, no public fee, no friction. That frictionlessness is the whole point.

```python
# requires prover credentials; fee master pays
result = credits.functions.transfer_public(
    str(account.address), 100
).delegate(account)
```

Want to pay your own fee instead? `delegate(account, pay_own_fee=True)` (public) or `delegate(account, fee_record=<credits record>)` (private). Both bind the fee to the real execution id, so they prove the execution locally first. `broadcast=False` returns the proven transaction without submitting it.

## Async (`AsyncAleo`)

The async facade mirrors the sync surface. Construction and the local, CPU-bound steps stay synchronous; everything that touches the network is awaitable.

```python
import asyncio
from aleo import AsyncAleo

async def main():
    aleo = AsyncAleo(AsyncAleo.HTTPProvider("https://api.provable.com/v2"))
    print(aleo.network_name)   # sync — no I/O

    # Account ops are sync (purely local), even on AsyncAleo
    account = aleo.account.create()
    sig = aleo.account.sign(b"hi", account)
    assert aleo.account.verify(str(account.address), b"hi", sig)

    # Reads are awaited  # requires a live node
    micro = await aleo.get_balance(str(account.address))
    print(aleo.from_microcredits(micro), "credits")

    # Build a call — fetching the program is awaited  # requires a live node
    credits = await aleo.programs.get("credits.aleo")
    call = credits.functions.transfer_public(str(account.address), 100)

    # authorize / simulate / call are SYNC (local proof-free build)
    auth = call.simulate(account)
    print(auth.decoded())

    # transact / delegate are awaited  # requires a live node / prover creds
    tx_id = await call.transact(account)
    result = await call.delegate(account)   # fee master pays by default

asyncio.run(main())
```

**Sync vs async on the async facade:**

- **Sync (local, no I/O):** `account.*` (create/import/sign/verify), `to_microcredits` / `from_microcredits` / `is_valid_address`, and `authorize` / `simulate` / `call` on a bound call.
- **Async (awaitable):** `is_connected`, `get_balance`, `programs.get`, mapping reads, `build_transaction` / `transact` / `delegate`. Heavy proving runs in `asyncio.to_thread` so it does not block the event loop.

## Testing utilities (`aleo.testing`)

The SDK ships an eth-tester-style harness for fast, deterministic local testing.

### `Devnode` — a local chain in a context manager

`Devnode` launches a local [`aleo-devnode`](https://github.com/ProvableHQ/aleo-devnode) with **manual block creation**, so tests control exactly when blocks are produced. It auto-picks a free port and tears the node down on exit.

```python
from aleo.testing import Devnode

with Devnode() as dn:
    aleo = dn.aleo                 # an Aleo client wired to the devnode
    alice = dn.accounts[0]         # 5 deterministic, pre-funded genesis accounts

    tx_id = aleo.programs.get("credits.aleo").functions.transfer_public(
        str(dn.accounts[1].address), 1_000_000
    ).transact(alice)

    dn.advance(1)                  # produce 1 block to confirm the tx
    snap = dn.snapshot()           # capture chain state
```

- `dn.aleo` — an `Aleo` client pointed at the devnode.
- `dn.accounts` — 5 deterministic, pre-funded genesis accounts.
- `dn.advance(n)` — produce `n` blocks (the node runs with manual block creation).
- `dn.snapshot()` — capture the current chain state.

Requires the `aleo-devnode` binary on your `PATH`, or set `$ALEO_DEVNODE_BIN` to its location.

### `LocalRecordScanner` — client-side record finding

`LocalRecordScanner` implements the `RecordProvider` protocol entirely on the client: it scans blocks and decrypts them with **your** view key. There is no hosted scanner and no view-key sharing.

```python
from aleo.testing import LocalRecordScanner

scanner = LocalRecordScanner(aleo, account)
rec = scanner.get_unspent(program="credits.aleo", record="credits")

# Plug it into the facade so private-fee transactions auto-source records:
aleo.record_provider = scanner
```

Any object satisfying the `RecordProvider` protocol can be assigned to `aleo.record_provider`, so you can keep your view key private instead of relying on a hosted scanning service.

### Live end-to-end tests

The `-m slow` live tests hit a real testnet and **skip automatically when their environment variables are unset**:

| Variable | Purpose |
| --- | --- |
| `ALEO_E2E_PRIVATE_KEY` | A funded testnet private key |
| `ALEO_E2E_ENDPOINT` | Node/API endpoint to test against |
| `ALEO_E2E_API_KEY` | Provable API key |
| `ALEO_E2E_CONSUMER_ID` | DPS consumer id for delegated proving |

The `-m devnode` tests additionally require the `aleo-devnode` binary (see `Devnode` above).

## Low-level primitives

When you need direct control, import the network module and use the cryptographic types directly:

```python
from aleo.mainnet import PrivateKey, Signature

# Generate a random private key
pk = PrivateKey.random()
address = pk.address
view_key = pk.view_key

# Sign a message
message = b"hello"
signature = Signature.sign(pk, message)
assert signature.verify(address, message)
```

`aleo.mainnet` (and `aleo.testnet`, when the extension is compiled) expose the full type set — `Account`, `Program`, `Process`, `Authorization`, `Transaction`, `RecordPlaintext`, `Field`, `Address`, and more.

## Build & Install

```bash
bash install.sh
```

This creates a development environment and installs the `aleo` package with MainnetV0 bindings.

## Contributing
If you wish to contribute, please follow the contribution guidelines outlined on [GitHub](https://github.com/AleoHQ/python-sdk/blob/master/sdk/CONTRIBUTING.md).
