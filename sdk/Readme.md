# Aleo Python SDK (MainnetV0)

The Aleo Python SDK provides Python bindings to Aleo's zero-knowledge cryptographic primitives, built with snarkvm 4.12.0.

It ships two layers:

- A **Web3.py-style facade** (`aleo.Aleo` / `aleo.AsyncAleo`) — a high-level, batteries-included client for connecting to a node, managing accounts, reading state, and building/proving/broadcasting transactions.
- **Low-level primitives** (`aleo.mainnet`, `aleo.testnet`) — direct Python bindings to Aleo's cryptographic types, for when you need full control.

## Quick Start (facade)

```python
from aleo import Aleo

# Connect (construction is offline — no I/O until you make a call)
aleo = Aleo(Aleo.HTTPProvider("https://edge.provable.com/api"))
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

aleo = Aleo(Aleo.HTTPProvider("https://edge.provable.com/api"))

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

# Recover a wallet account from its recovery phrase (offline; nothing is stored)
account = aleo.account.from_mnemonic(phrase)            # first account of the phrase
second = aleo.account.from_mnemonic(phrase, index=1)    # the next one

# Create a new account a wallet can import later (offline; nothing is stored)
phrase, account = aleo.account.create_with_mnemonic()   # show `phrase` to the user once

# Sign / verify (bytes)
sig = aleo.account.sign(b"hello aleo", account)
assert aleo.account.verify(str(account.address), b"hello aleo", sig)

# Sign / verify a typed Aleo value
sv = aleo.account.sign_value("100u64", account)
assert aleo.account.verify_value(str(account.address), "100u64", sv)
```

`create_with_mnemonic` makes a new 12-word phrase (24 with `strength=256`) and returns it with its first account; `generate_mnemonic` returns only the phrase. The phrase controls every account derived from it and is not stored anywhere, so give it to the person who will keep it and do not log it. `create` makes an account from a random key instead, which no phrase can recover. `from_mnemonic` takes a BIP-39 English phrase of 12 to 24 lower-case words separated by single spaces and uses the derivation of the Shield wallet and the other Aleo SDKs, so one phrase gives the same accounts everywhere. A mistyped word raises `aleo.keystore.KeystoreError` with `code == "INVALID_MNEMONIC"` instead of deriving a different account. Pass `derivation="legacy"` only to recover an account from a wallet that used the path `m/44'/0'` before Aleo's own path `m/44'/683'` was registered.

Set `aleo.default_account = account` and the verbs below will use it whenever you omit the signer.

> **Note:** the facade deliberately has no "recover signer from signature" verb. Aleo is a privacy chain — surfacing "which address signed this?" is a de-anonymisation vector. The low-level `Signature.to_address()` primitive remains available directly if you truly need it.

### Store an account — `aleo.keystore`

The functions above keep a key in memory. `aleo.keystore` stores an account on disk and finds it again on later runs, in an account file whose format is shared with the other Aleo SDKs, so an account set up once can serve every tool that reads it. The file is network-neutral: one account works on testnet and mainnet, and named accounts keep separate keys apart. Nothing here contacts a network, and nothing is written until `save`, `protect` or `migrate` is called.

```python
from aleo import Aleo, HTTPProvider, keystore

aleo = Aleo(HTTPProvider("https://edge.provable.com/api", network="testnet"))

# First run: create an account and store it under a name.
stored = keystore.save(aleo.account.create(), name="testnet")
print(stored.address, stored.storage, stored.path)   # never the key

# Later runs: resolve the key and hand it to the client.
loaded = keystore.load(name="testnet")
account = aleo.account.from_private_key(loaded.private_key)
```

The account file is `~/.aleo/accounts/<name>.json`, readable by its owner only. The name defaults to `default`; `ALEO_ACCOUNT` or the `name` argument selects another. `ALEO_ACCOUNTS_DIR` or the `dir` argument moves the directory, for example into a project or a test directory.

#### Choose how the key is stored

`save` takes a `storage` argument. Encryption and the Keychain protect a key at rest: a copied file, a backup, a synced folder. They do not stop a process running as the same user from reading it.

| `storage` | Where the key is | Choose it when |
| --- | --- | --- |
| `"plaintext"` (default) | In the account file, mode 0600 | The machine and its backups are trusted; no password is ever needed |
| `"keychain"` | In the macOS login keychain, under `aleo-account-key`. Each account file has an entry of its own, named after the address and an identifier the file records, and labelled `aleo-account-key:<name>`; the file holds no secret | Working on a Mac, and the file may be copied or backed up. The key has no other copy, so export it once as a backup |
| `"encrypted"` | In the account file, encrypted under a password | Any system where the file may leak. Each run needs the password |
| `"none"` | Nowhere; the file records the address | The key arrives through the environment on every run and the file only confirms it is the expected account |

An encrypted account takes its password from the `password` argument, then `ALEO_ACCOUNT_PASSWORD`, then the macOS Keychain when `save_password=True` put it there, then a hidden terminal prompt. A saved password belongs to one encrypted file: its Keychain entry is named after the address and the file's salt, so two files that hold one account under different passwords do not share it. An unattended job cannot answer a prompt, so it relies on the variable or the saved password; pass `prompt=False` to fail at once instead of waiting. A wrong password from any source but the prompt is an error and the later sources are not tried.

```python
keystore.save(account, name="mainnet", storage="encrypted", save_password=True)   # asks for a password twice
keystore.protect(name="testnet", storage="keychain")    # move an existing account; the old copy is removed after a check
print(keystore.info(name="testnet"))                    # name, path, address, storage; None when there is no file
key = keystore.export(name="testnet")                   # the private key itself: do not log it
```

#### How `load` finds the key

Every tool that uses the keystore resolves the key in the same order, and the first source that supplies one wins:

1. The `private_key` argument.
2. `ALEO_PRIVATE_KEY`, then its alias `SHIELD_SWAP_PRIVATE_KEY`.
3. The text file named by `ALEO_PRIVATE_KEY_FILE`, then its alias `SHIELD_SWAP_PRIVATE_KEY_FILE`.
4. The selected account file.
5. The `legacy_key` argument, which a tool passes when its own older state file still holds a key.

A key from the environment is used for that process and never written to disk. A variable that is set but empty is an error, as are two variables that hold different keys, and when the account file exists a key from sources 1 to 3 must belong to the address it records. `load` never generates a key: with no source it raises `ACCOUNT_NOT_FOUND`. `loaded.source` names the source used (`"argument"`, `"env"`, `"env-file"`, `"account-file"` or `"legacy"`).

`keystore.migrate(path)` copies the key out of a `profile.json` or `state.json` that keeps it in a top-level `private_key` or `privateKey` field. The old file keeps its plaintext copy unless `remove_legacy_key=True` is passed, which strips the field after the new account has been read back.

#### Handle failures

Every failure the keystore detects is an `aleo.keystore.KeystoreError` (a subclass of `aleo.AleoError`) with a stable `code`. Messages name addresses and paths, never a key or a password.

```python
from aleo.keystore import KeystoreError

try:
    loaded = keystore.load()
except KeystoreError as error:
    if error.code == "ACCOUNT_NOT_FOUND":
        ...   # no key in the environment and no account file: create or import one
    elif error.code == "PASSWORD_REQUIRED":
        ...   # encrypted account and no password source
    else:
        raise
```

The codes are `ACCOUNT_NOT_FOUND`, `KEY_NOT_STORED`, `ACCOUNT_EXISTS`, `ACCOUNT_BUSY`, `ADDRESS_MISMATCH`, `KEY_CONFLICT`, `EMPTY_KEY_SOURCE`, `INVALID_KEY`, `INVALID_MNEMONIC`, `INVALID_NAME`, `INVALID_ACCOUNT_FILE`, `UNSUPPORTED_VERSION`, `UNSUPPORTED_STORAGE`, `KEYCHAIN_UNAVAILABLE`, `KEYCHAIN_ITEM_NOT_FOUND`, `KEYCHAIN_FAILED`, `PASSWORD_REQUIRED`, `WRONG_PASSWORD`, `TERMINAL_REQUIRED` and `LEGACY_KEY_NOT_FOUND`; the `KeystoreError` docstring says when each is raised. A file system failure on the account file other than a missing file (permission denied, a directory in its place) is raised as `OSError`.

#### Set up an account from a terminal

`python -m aleo.keystore <command>` runs the same operations without writing code. Passwords are never taken from the command line: they come from `ALEO_ACCOUNT_PASSWORD`, the Keychain or a hidden prompt.

| Command | Does |
| --- | --- |
| `create` | Generates a key and stores it (`--storage plaintext` unless stated). With `--mnemonic` the key comes from a new recovery phrase, printed once and stored nowhere, so a wallet can import the account later; this needs a terminal |
| `import` | Stores an existing key from `--key-file`, the key variables or a hidden prompt, or the account of a recovery phrase from `--mnemonic-file` or the prompt (`--index`, `--derivation`). The derived key is stored, not the phrase |
| `show` | Prints the address, account name and storage type. Never the key |
| `protect` | Changes the storage with `--storage`; `--save-password` saves the password in the Keychain and `--forget-password` removes a saved one |
| `export` | Prints the key after a confirmation. Requires a terminal |
| `migrate` | Copies a key from `--from <profile.json or state.json>`; `--remove-legacy-key` strips it from the old file |

```sh
python -m aleo.keystore create --name testnet
python -m aleo.keystore import --key-file ./account.key --storage encrypted --save-password
python -m aleo.keystore show --json
```

Every command accepts `--name`, `--dir` and `--json`. With `--json` a result is one object with `name`, `address`, `storage` and `path`, and an error is `{"error": {"code": "…", "message": "…"}}` on standard output; without it, errors go to standard error as `<CODE>: <message>`. `export` and `create --storage none` show a key, so they run only with a terminal attached and never with `--json`: an agent or a script must not run them.

| Exit status | Meaning |
| --- | --- |
| 0 | Done |
| 1 | Failed |
| 2 | Needs a person: `ACCOUNT_NOT_FOUND`, `KEY_NOT_STORED`, `PASSWORD_REQUIRED` or `TERMINAL_REQUIRED` |
| 64 | Usage error |

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

### Private transfers — delegated proving + record discovery

A full private transfer combines the two trust-minimising paths: **delegated proving** (the prover's fee master pays) and a **record provider** to discover the private records you own. The default record provider (`aleo.records`) uses Provable's hosted scanner — registering shares your view key with that service so it can index your records. If you'd rather not share a view key, assign your own `RecordProvider` instead.

```python
# requires prover credentials + hosted-scanner registration
credits = aleo.programs.get("credits.aleo")

# 1) Move public credits into a private record (delegated — fee master pays)
credits.functions.transfer_public_to_private(str(account.address), 100_000) \
    .delegate(account)

# 2) Register with the record provider and find the new private record
aleo.records.register(account)                 # shares the view key with the scanner
record = aleo.records.get_unspent_credits_record()

# 3) Spend the private record with a private transfer (delegated)
credits.functions.transfer_private(record, str(recipient.address), 1) \
    .delegate(account)
```

`aleo.record_provider` is swappable: set it to your own object implementing the `RecordProvider` protocol (`get_unspent_credits_record` / `find`), and the whole facade (including private-fee auto-sourcing) uses it, with no view-key sharing.

### Record inventory — join, split, plan, reserve, recover

One transaction spends whole records, so a large payment needs one record that covers it and several payments submitted together need one record each. `aleo.inventory` reads an asset's spendable records, plans the joins and splits that reach a target layout, and executes that plan one confirmed transaction at a time. Build the client with `record_management` enabled and the journal follows the two halves of a write: `build_transaction` reserves the record inputs before proving and saves the proved transaction, `submit_transaction` records the broadcast (`transact` is both in sequence, `delegate` does the same through the prover), and `result.discard()` releases a built transaction that was never sent (after a broadcast attempt the reservation stays until reconciliation). Writers sharing the same store never spend the same record.

```python
from aleo.inventory import InventoryTarget, RecordAsset, RecordManagementConfig

aleo = Aleo(HTTPProvider(ENDPOINT, network="testnet"),
            record_management=RecordManagementConfig(enabled=True))
credits = RecordAsset("credits.aleo")
plan = aleo.inventory.plan_record_inventory(
    credits, InventoryTarget(records=4, min_record_amount=1_000_000, distribution="balanced"))
result = aleo.inventory.rebalance_record_inventory(plan, max_fee_microcredits=1_000_000)
```

`get_record_inventory` reads; `plan_record_inventory` plans without signing; `join_records`, `split_record`, and `auto_join` run single operations; `reconcile_record_inventory` settles pending journal entries against the chain and can rebroadcast a saved proved transaction. The journal is in memory by default; `JsonFileRecordInventoryStore(path)` and `SqliteRecordInventoryStore(path)` keep it across restarts and coordinate processes on one machine, and any object with `list`, `acquire`, and `update` works for another backend. The top-level [README](../README.md#manage-private-records) compares the stores and explains the recovery steps.

## Async (`AsyncAleo`)

The async facade mirrors the sync surface. Construction and the local, CPU-bound steps stay synchronous; everything that touches the network is awaitable.

```python
import asyncio
from aleo import AsyncAleo

async def main():
    aleo = AsyncAleo(AsyncAleo.HTTPProvider("https://edge.provable.com/api"))
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
