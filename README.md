# Aleo Python SDK

Build Python applications that read Aleo balances, send public or private
transfers, and interact with deployed programs. Use the same account to bridge
assets from Ethereum or Solana, trade on Shield Swap, or manage private records.

Start with `Aleo` for account and transaction workflows, or `AsyncAleo` for
applications that need concurrent network requests. Cryptographic types are
also available directly for signing, record handling, and custom integrations.

See the [SDK walkthrough](sdk/python/examples/facade_quickstart.py) for account
and program examples, and the [bridge tutorials](bridge-sdk/examples/README.md)
for transfers to and from Aleo, shielding, and recovery.

## Install

Install the core SDK to manage accounts, read the network, and submit Aleo
transactions. Delegated proving is included: a service generates proofs
instead of the application's machine.

```sh
python -m pip install aleo-sdk
```

The repository also includes packages for specific tasks. Install these when
the application needs the corresponding workflow:

| Task | Package | Guide |
| --- | --- | --- |
| Read balances, send transactions, and call programs | `aleo-sdk` | This README |
| Bridge assets between Aleo, Ethereum, and Solana | `aleo-bridge-sdk` | [Bridge SDK](bridge-sdk/README.md) |
| Swap assets and manage liquidity on Shield Swap | `shield-swap-sdk` | [Shield Swap SDK](shield-swap-sdk/README.md) |
| Generate program ABIs and check compatibility | `aleo-contract-abi-generator` | [ABI generator](sdk-abi/README.md) |

## Setup

To read account balances or submit transactions, create an Aleo client for the
intended network. Reading public data needs only an address. Sending funds also
requires the private key for the account holding those funds.

This setup connects to mainnet and imports an existing account from the
application's environment. Keep the key in the application's secret store;
never paste it into documentation, logs, or an agent conversation.

```python
import os
from aleo import Aleo, HTTPProvider

aleo = Aleo(HTTPProvider("https://edge.provable.com/api", network="mainnet"))
account = aleo.account.from_private_key(os.environ["ALEO_PRIVATE_KEY"])
aleo.default_account = account
address = str(account.address)
print(address)  # The public aleo1… address used to receive funds.
```

For a new account, use `aleo.account.create()` instead of importing a key.
Retain the new account's private key securely before receiving funds; creating
another account later does not restore access to the first one.

The following synchronous examples continue this setup. Each submission example
spends real funds on the configured network. Alternative proving methods are
separate choices, not consecutive steps for the same transfer.

## Read balances and program state

Check a public balance before deciding how much to send, leaving enough for any
fee paid by the account. Public credits are visible on-chain; this balance does
not include credits held in private records.

```python
microcredits = aleo.get_balance(address)
print(aleo.from_microcredits(microcredits), "credits")
# 1,500,000 microcredits are 1.5 credits; an unfunded address returns 0.
```

Programs can also store public values in mappings, which associate a key with a
value. For example, the credits program's `account` mapping holds public credit
balances:

```python
credits = aleo.programs.get("credits.aleo")
value = credits.mapping("account").get(address)
print(value)  # The mapping value returned by the node.
```

Use integer microcredits for credit amounts: **1 credit = 1,000,000
microcredits**. For conversion from display units, pass an exact decimal string,
such as `aleo.to_microcredits("1.5")`, instead of a floating-point amount.

## Send credits

A public transfer makes credits available at another Aleo address. The sender,
recipient, and amount are public. Start by preparing the recipient and amount,
inspect the intended operation, then choose how to prove and submit it.

### Prepare and inspect the transfer

A program call describes the operation to perform. Preparing it does not send
funds. This example prepares a transfer of one credit to an illustrative public
address; replace it with the intended recipient before submitting.

```python
recipient = "aleo1rs6fdxg703s3em27uhxsehhfd8znaly22jt6upggrxc77q8d9yfq33pk28"
amount = 1_000_000  # One credit, in microcredits.
credits = aleo.programs.get("credits.aleo")
call = credits.functions.transfer_public(recipient, amount)

print(call.signature)  # transfer_public(address, u64)
print(call.args)       # Recipient address and amount encoded for the program.
authorization = call.simulate(account)
print(authorization.decoded())  # Inspect the authorized inputs and outputs.
```

Simulation signs an authorization locally without generating a proof or
broadcasting a transaction. It does not reserve funds or guarantee that the
network will accept the eventual transaction.

Other deployed programs use the same pattern: load the program with
`aleo.programs.get("program_name.aleo")`, then select a function through
`program.functions`. `list(credits.functions)` lists the available functions.

### Choose where to generate the proof

Every Aleo transaction needs a cryptographic proof. Delegated proving avoids
performing that computation on the application's machine. The proving service
receives the transaction contents, while the private key stays with the
application. By default, `delegate` requests that the service's fee-paying
account cover the transaction fee.

To submit the prepared transfer through delegated proving:

```python
result = call.delegate(account)  # Requests proving and broadcast by the service.
# Retain the service response and its transaction ID to check confirmation.
```

The default `https://edge.provable.com/api` service does not require API
credentials. Credentialed deployments, such as `https://api.provable.com`,
require their own API key and consumer ID.

For transaction contents that should stay out of the proving service, generate
the proof locally instead. This uses the application's CPU and may download
proving parameters. The account pays a public transaction fee by default.

Run this **instead of** the delegated submission above:

```python
tx_id = call.transact(account)  # Proves locally and broadcasts once.
print(tx_id)  # Save this at1… transaction ID before waiting for confirmation.
```

To inspect a proven transaction before broadcasting, use
`call.build_transaction(account)`. Fee choices and private fee records are
covered under [Account, privacy, and proving options](#account-privacy-and-proving-options).

### Confirm delivery

Submission and confirmation are separate steps. Retain the transaction ID so
monitoring can continue if the application closes or a request times out.
This example continues the local submission above:

```python
from aleo import TransactionConfirmationTimeout

try:
    confirmed = aleo.network.wait_for_transaction(tx_id, timeout=60.0)
except TransactionConfirmationTimeout:
    print("Still unconfirmed; check this transaction again:", tx_id)
else:
    print("Confirmed:", tx_id)
```

**Do not send the transfer again because monitoring timed out.** Check the
existing transaction with `aleo.network.get_confirmed_transaction(tx_id)` or
continue waiting with the same ID. If the submission response was lost, inspect
account transaction history before trying another submission. Rejected
transactions can still incur fees.

## Use private credits

Private credits are held in encrypted records, analogous to Bitcoin's UTXOs.
A private transfer spends an unspent record and creates records for the
recipient and any change. The input record must cover the amount being sent.

Use private transfers when balances and transfer details should not be publicly
readable. Finding spendable records is a separate choice from proving a
transaction: the default hosted scanner needs the account's view key, which
lets that service decrypt its records. A view key cannot spend the funds.

The following example assumes the configured account already has a private
credits record. It registers the account for scanning, selects a record covering
one credit, and submits a private transfer through delegated proving:

```python
registration = aleo.records.register(account)  # Shares the view key with the scanner.
if not registration.get("ok"):
    raise RuntimeError("Record scanner registration failed; no transfer was submitted.")

record = aleo.records.get_unspent_credits_record(min_microcredits=1_000_000)
if record is None:
    raise RuntimeError("No suitable record found; check funds and scanner progress.")

credits = aleo.programs.get("credits.aleo")
private_call = credits.functions.transfer_private(record, recipient, 1_000_000)
result = private_call.delegate(account)  # The prover can read this transaction's contents.
```

A missing record can mean that scanning has not caught up, that funds are
already spent, or that no single record covers the amount. It does not establish
that the account has no private funds.

To create a private balance from public credits, use
`credits.functions.transfer_public_to_private(address, 1_000_000)` and submit
with one of the proving methods above. Wait for confirmation and scanner
indexing before trying to spend the new record. The source public balance change
remains visible.

To avoid sharing a view key with the hosted scanner, supply an unspent record
already held by the application, or assign a custom `RecordProvider` to
`aleo.record_provider`. Choose local proving separately if transaction contents
must also stay out of the proving service.

## Bridge assets

Bring assets from Ethereum or Solana to Aleo for use in Aleo applications, or
withdraw them back to their source chain. Hyperlane carries ETH, WBTC, USDT,
and SOL; Circle xReserve connects Ethereum USDC with Aleo USDCx.

The [Bridge SDK guide](bridge-sdk/README.md) covers supported pairs, costs,
public and private delivery, and recovery after an interruption. Its
[runnable tutorials](bridge-sdk/examples/README.md) show transfers in both
directions with explicit quotes, submission steps, and progress checks.

```sh
python -m pip install aleo-bridge-sdk
```

Keep a bridge journal when submitting transfers. It identifies the existing
transfer and completed work so an interruption can be recovered without sending
the funds again.

## Swap assets and manage liquidity

Use the Shield Swap SDK to trade assets privately on Aleo, provide liquidity,
and collect earnings. The [Shield Swap guide](shield-swap-sdk/README.md)
covers account setup, pool selection, quotes, and transaction workflows.

```sh
python -m pip install shield-swap-sdk
```

## Use an agent

An agent can help find supported assets, compare transfer costs, prepare a swap,
or check whether a transaction completed. The Bridge and Shield Swap packages
provide guides and tools for these tasks. Choose a guide for a coding agent
working in a project, or MCP for an application that connects to tool servers.

### Give a coding agent the package guide

Install the package needed for the task, then ask the agent to read its generated
guide. These commands print the available operations and how to use them; they
do not submit transactions:

```sh
python -m aleo_shield_swap  # Account, swap, and liquidity instructions.
python -m aleo_bridge      # Route, quote, transfer, and recovery instructions.
```

Keep existing project instructions when adding these guides; redirecting output
to an existing `AGENTS.md` would replace that file. Configure private keys through
the application's environment or secret store, rather than through chat.

Start with a concrete request, such as “Quote a transfer of 5 USDC from Ethereum
to USDCx on Aleo” or “Show the available Shield Swap pools.” Include the intended
network, accounts, and amount when relevant. Review the quote before authorizing
a transaction, and retain its transaction ID or bridge journal for recovery.

### Connect an MCP client

MCP lets an agent application call the package's tools through a local process.
Install the extra for the required package:

```sh
python -m pip install 'shield-swap-sdk[mcp]' 'aleo-bridge-sdk[mcp]'
```

Use `python -m aleo_shield_swap.mcp` or `python -m aleo_bridge.mcp` as the server
command. The package guides explain which tools read data and which submit
transactions, including their confirmation requirements.

### Limit access to read operations

For a bridge assistant that should only inspect routes, quotes, and progress,
expose the read-only tool definitions:

```python
from aleo_bridge import bridge_tools

tools = bridge_tools(include_writes=False)
# Supply these definitions to the application's agent integration.
```

When bridge submission tools are enabled, they require `confirm: true` before
moving funds. Without it, they return the quote or pending action for review.
The application should obtain the caller's authorization before supplying that
flag; a tool argument does not establish human approval by itself.

## Make concurrent network requests

Use `AsyncAleo` when an application needs to wait on network requests without
blocking other work. Install the async extra, then await network reads and
transaction submissions. Account creation, signing, and preparing a call remain
synchronous.

```sh
python -m pip install 'aleo-sdk[async]'
```

This standalone example reads two public balances concurrently without loading
keys or submitting transactions:

```python
import asyncio
from aleo import AsyncAleo, HTTPProvider

async def main():
    aleo = AsyncAleo(HTTPProvider("https://edge.provable.com/api", network="mainnet"))
    accounts = [aleo.account.create(), aleo.account.create()]
    balances = await asyncio.gather(
        *(aleo.get_balance(str(account.address)) for account in accounts)
    )
    for account, balance in zip(accounts, balances):
        print(account.address, aleo.from_microcredits(balance), "credits")
        # Newly created accounts normally have no funds.

asyncio.run(main())
```

Fetching a program uses `await aleo.programs.get(...)`. Preparing its call and
running `call.simulate(account)` stay synchronous. Choose either
`await call.delegate(account)` or `await call.transact(account)` to submit a
transfer; do not run both for the same intended payment.

## Test without spending mainnet funds

Use a local development node to test transactions with prefunded development
accounts. Tests control when blocks are created, so they can check behavior
before and after confirmation.

Install `aleo-devnode` and make its binary available on `PATH`, or set
`ALEO_DEVNODE_BIN` to its location. This standalone example sends one credit
between development accounts and produces a block:

```python
from aleo.testing import Devnode

with Devnode() as node:
    aleo = node.aleo
    sender, recipient = node.accounts[:2]
    credits = aleo.programs.get("credits.aleo")
    tx_id = credits.functions.transfer_public(
        str(recipient.address), 1_000_000
    ).transact(sender)
    node.advance(1)  # Include the transfer in a newly produced local block.
    confirmed = aleo.network.wait_for_transaction(tx_id, timeout=10.0)
```

Development accounts are deterministic test fixtures; never use their keys to
hold real funds. `node.snapshot()` captures the node's current chain state.

For SDK development, see the [build instructions](sdk/Readme.md). Run the fast
Python suite from `sdk/` so its pytest configuration applies:

```sh
cd sdk
python -m pytest python/tests -m 'not slow'
```

Live tests marked `slow` use configured network services and may submit
transactions. Review their account and endpoint requirements before enabling
them; `devnode` tests additionally require the local node binary.

## Account, privacy, and proving options

### Sign messages without sending a transaction

Use signatures to verify that a message was authorized by a specified account.
Signing and verification run locally and do not pay a transaction fee.
This example continues the account setup above:

```python
message = b"Approve this application session"
signature = aleo.account.sign(message, account)
assert aleo.account.verify(address, message, signature)

value_signature = aleo.account.sign_value("100u64", account)
assert aleo.account.verify_value(address, "100u64", value_signature)
```

For deterministic account generation, `aleo.account.from_seed("123field")`
accepts a field seed. That value is an illustrative, publicly known seed;
accounts holding funds need a securely generated secret seed or private key.

### Choose how to pay transaction fees

Local proving pays a public fee by default. Add `priority_fee=` in microcredits
for an additional fee, or use `private_fee=True` to select a private fee record
through `aleo.record_provider`. An explicit `fee_record=` uses the supplied
credits record instead.

Delegated proving requests service-paid fees by default. To pay from the
account's public balance, use `delegate(account, pay_own_fee=True)`; to use a
private credits record, pass `fee_record=`. These self-paid delegated options
also prove the execution locally to bind the fee, so they require local proving
resources. `broadcast=False` requests proving without submission.

### Use cryptographic types directly

Use the network-specific types when implementing signing, record processing,
or other operations below the account and program client. This standalone
example creates a key and verifies a signature without contacting the network:

```python
from aleo.mainnet import PrivateKey, Signature

key = PrivateKey.random()
signature = Signature.sign(key, b"hello")
assert signature.verify(key.address, b"hello")
```

`aleo.mainnet` exposes types including `Account`, `Program`, `Process`,
`Authorization`, `Transaction`, `RecordPlaintext`, `Field`, and `Address`.
`aleo.testnet` provides the corresponding types when the testnet extension is
built. Match the types and client to the intended network.

The SDK uses snarkVM 4.9.1. The repository also contains [zkML tooling](zkml/)
for translating Python machine-learning models into Leo and
[zkML research](zkml-research/) on model accuracy and constraint costs.
