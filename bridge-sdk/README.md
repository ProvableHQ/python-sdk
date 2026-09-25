# Aleo Bridge SDK

Transfer assets between Aleo, Ethereum, and Solana through Hyperlane and Circle
xReserve. The SDK quotes transfer costs, submits transactions, tracks delivery,
and recovers interrupted transfers from saved checkpoints.

A transfer starts with `quote`, continues with `execute`, and is followed with
`wait`. If another transaction is needed, the returned `progress.next` names
the action: `resume` for an unfinished source transfer or `complete` for a
private USDCx mint on Aleo.

## Install

Requires Python 3.10 or later. One installation includes support for all three
chains and delegated Aleo proving.

```sh
python -m pip install aleo-bridge-sdk
```

Import the package as `aleo_bridge`.

## Supported transfers

| Source chain | Source asset | Destination chain | Destination asset | Provider |
| --- | --- | --- | --- | --- |
| Ethereum | ETH | Aleo | ETH | Hyperlane |
| Aleo | ETH | Ethereum | ETH | Hyperlane |
| Ethereum | WBTC | Aleo | WBTC | Hyperlane |
| Aleo | WBTC | Ethereum | WBTC | Hyperlane |
| Ethereum | USDT | Aleo | USDT | Hyperlane |
| Aleo | USDT | Ethereum | USDT | Hyperlane |
| Solana | SOL | Aleo | SOL | Hyperlane |
| Aleo | SOL | Solana | SOL | Hyperlane |
| Ethereum | USDC | Aleo | USDCx | Circle xReserve |
| Aleo | USDCx | Ethereum | USDC | Circle xReserve |

## Setup

Create an Aleo client, attach the Ethereum connection, and configure a
checkpoint directory so the transfer can be recovered after a restart. Run the
following sections in the same Python session.

```python
import os
from aleo import Aleo, HTTPProvider
from aleo_bridge import Bridge, Ethereum, FileCheckpointStore

aleo = Aleo(HTTPProvider("https://edge.provable.com/api", network="mainnet"))
aleo.default_account = aleo.account.from_private_key(os.environ["ALEO_PRIVATE_KEY"])
ethereum = Ethereum(
    "https://ethereum-rpc.publicnode.com",
    private_key=os.environ["EVM_PRIVATE_KEY"],
)
store = FileCheckpointStore("~/.aleo-bridge/checkpoints")
bridge = Bridge(aleo, ethereum=ethereum, checkpoints=store)
recipient = bridge.aleo_address()
```

The account signs inside the application's process. Creating this client does
not submit a transfer. Network reads use the configured RPC and API endpoints;
transfer submission uses the configured signer.

For Solana transfers, add a connection to the same client:

```python
from aleo_bridge import Solana

solana = Solana(
    os.environ["SOLANA_RPC_URL"],
    private_key=os.environ["SOLANA_PRIVATE_KEY"],
)
bridge = Bridge(aleo, ethereum=ethereum, solana=solana, checkpoints=store)
```

`SOLANA_PRIVATE_KEY` accepts a base58 wallet export or the JSON array of 64
integers from a Solana CLI keypair file. Only configure the side chains needed
by the application.

Existing clients and signers can be supplied through
`Ethereum(w3=..., signer=...)` and `Solana(client=..., signer=...)`.
A side-chain connection without a signer supports reads.

Aleo transactions use delegated proving by default (`proving="delegate"`).
The proving service receives the transaction contents, but not the private
key. Pass `proving="local"` to `execute`, `resume`, or `complete` to prove on
the application's machine; local proving can download proving parameters.

## Find supported assets and routes

The registry is the reviewed catalog bundled with the package. Reading it does
not contact a network or request a signature.

```python
routes = bridge.routes(source_chain="ethereum", destination_chain="aleo")
for route in routes:
    print(route.id, route.protocol, route.availability)
```

Applications select routes by chain and asset names. They do not construct
route strings or copy contract addresses into transfer requests. Pass
`include_unavailable=True` to also list `metadata-required` entries, and check
`route.active` before presenting a route as executable.

## Move an asset across chains

Every transfer follows the same caller lifecycle:

1. `quote` checks that the requested transfer is supported and reports the
   costs that can be known before submission.
2. `execute` signs and submits the required source-chain transactions.
3. `wait` follows the submitted transfer until it finishes, fails, or requires
   another action from the caller.
4. `resume` or `complete` runs only when `progress.next` requests that action.

### 1. Get a bridge quote

The caller supplies the source chain and asset, the destination chain, the
amount in the asset's display units, and the recipient address on the
destination chain. Quoting can read networks and providers, but it does not
request a signature or move funds.

```python
quote = bridge.quote(
    source_chain="ethereum",
    source_asset="wbtc",
    destination_chain="aleo",
    amount="0.001",
    recipient=recipient,
)

for fee in quote.fees:
    print(fee.kind, fee.amount, fee.asset_id, "estimated" if fee.estimated else "")
print(quote.amount_out)
```

`destination_asset` and `bridge_protocol` (`"hyperlane"` or `"xreserve"`)
are required only when more than one route fits the three names. A `Route`
from `bridge.routes(...)`, or its id, can be passed as `route=` instead of the
names.

`quote.plan` identifies the exact route, amount, recipient, and reviewed
deployment that produced the quote. Keep this value unchanged for execution.

### 2. Submit a bridge transaction from the source chain

After reviewing the quote, call `execute` to sign and submit the source
transfer. This step commits funds and pays source-chain transaction fees. An
ERC-20 route can require an approval before its bridge deposit; USDT resets an
existing non-zero allowance when another approval is needed.

The returned `Progress` contains the transfer plan, the latest receipt, and
the next action to take. The configured store saves checkpoints as submission
advances, including after approvals and around Aleo transaction broadcast.

```python
progress = bridge.execute(quote.plan)
print(progress.receipt.source_tx_id)
```

Once a source transaction has been submitted, do not call `execute` again for
the same transfer. Use the returned progress while the process stays alive, or
recover from the latest checkpoint after an interruption.

### 3. Monitor bridge progress

`wait` polls source confirmation, provider processing, and destination
delivery where the route exposes verifiable evidence. It does not request
another signature or submit a transaction.

```python
progress = bridge.wait(progress)
```

The `next` field is the only value an application needs to select the next
lifecycle action:

| `progress.next` | Caller action |
| --- | --- |
| `done` | Show completion. No further action is required. |
| `failed` | Show `progress.error`. Do not repeat a transaction that already succeeded. |
| `wait` | Call `wait` again; polling stopped at an application-selected status. |
| `resume` | Submit the remaining source operation with `resume`. |
| `complete` | Authorize the private USDCx mint with `complete`. |

`wait` raises `PollingTimeoutError` if it reaches `timeout_seconds` (1200 by
default). The exception carries the latest progress. A timeout leaves the
transfer's outcome unresolved; continue polling that transfer instead of
submitting another one.

To retain the latest progress when polling times out, use this in place of the
`wait` call above:

```python
from aleo_bridge import PollingTimeoutError

try:
    progress = bridge.wait(progress, timeout_seconds=1200)
except PollingTimeoutError as exc:
    if exc.progress is None:
        raise
    progress = exc.progress

print(progress.next, progress.receipt.source_tx_id)
```

`resume` applies when a source operation remains unfinished, such as a deposit
following a confirmed ERC-20 approval. It can submit a transaction and does not
repeat confirmed work. After the caller elects to continue:

```python
if progress.next == "resume":
    progress = bridge.resume(progress)
    progress = bridge.wait(progress)
```

`complete` applies to private USDCx deposits and requires the recipient's Aleo
account and the secret nonce used for the deposit. See
[USDC Bridging Guide](#usdc-bridging-guide) before starting that flow.

## Recover Funds

Recover an existing bridge transaction to check whether it has completed or
needs another action. Recovery does not start a new transfer. A checkpoint
records the transfer details and transaction identifiers needed to reconstruct
its progress; a bridge journal stores those checkpoints between sessions.

### Recover with a bridge journal

The `FileCheckpointStore` configured in Setup acts as the bridge journal.
Restart with the same network, accounts, and checkpoint directory. Load the
checkpoint for the transfer by its saved checkpoint ID, then refresh its state:

```python
checkpoint_id = os.environ["BRIDGE_CHECKPOINT_ID"]
checkpoint = store.load(checkpoint_id)
if checkpoint is None:
    raise ValueError(f"No saved checkpoint for {checkpoint_id}")

progress = bridge.recover(checkpoint)
print(progress.next, progress.error)
```

`recover` checks chain and provider state where needed without signing or
submitting a transaction. Follow the returned `progress.next`:

| `progress.next` | Action |
| --- | --- |
| `wait` | Call `bridge.wait(progress)` to continue monitoring. |
| `resume` | Call `bridge.resume(progress)` to submit the unfinished source operation. |
| `complete` | Call `bridge.complete(progress, secret_nonce=secret_nonce)` to mint private USDCx on Aleo. |
| `done` | Delivery is complete; no further action is needed. |
| `failed` | Inspect `progress.error` before deciding what to do. |

`resume` and `complete` can submit transactions. A private USDCx mint requires
the original secret nonce, which must be stored separately from the journal.
**Do not call `execute` again after an ambiguous submission.** A timeout or lost
RPC response does not establish that the original transaction failed.

### Find saved bridge transactions

If the checkpoint ID is unknown, load the journal's checkpoints and inspect
their transfer details:

```python
result = store.load_checkpoints()

for checkpoint in result.checkpoints:
    print(checkpoint.id, checkpoint.intent)

for error in result.errors:
    print(error.path, error.error)
```

`result.checkpoints` contains readable checkpoints, ordered oldest first by
file modification time. `result.errors` identifies files that could not be
read or parsed; these are file-loading errors, not failed bridge transactions.
Select the checkpoint for the intended transfer and pass it to `recover`.
Loading the journal does not contact a network.

### Recover without saved files

A submitted bridge transaction remains on-chain even if the application loses
its journal. Find the original transaction in the source wallet's history or
a block explorer. Use the bridge deposit or dispatch transaction, not a token
approval. Check its status, asset, amount, sender, and destination recipient
before proceeding.

For a **confirmed Ethereum-to-Aleo Hyperlane transfer**, those details are
enough to reconstruct the recovery data in memory. The example below recovers
a WBTC transfer on mainnet. Enter the original transfer's details from the
explorer; the amount is in WBTC, not its smallest units.

Use the Aleo and Ethereum connections from Setup, but construct the bridge
without a checkpoint store. This example reads the network and does not load
or write a journal, sign a transaction, or send funds:

```python
bridge = Bridge(aleo, ethereum=ethereum)
source_tx_id = input("Confirmed Ethereum bridge transaction hash: ").strip()
sender = input("Original Ethereum sender address: ").strip()
recipient = input("Original Aleo recipient address: ").strip()
amount = input("Original amount in WBTC: ").strip()

route = bridge.routes(
    source_chain="ethereum",
    source_asset="wbtc",
    destination_chain="aleo",
    bridge_protocol="hyperlane",
)[0]

recovery_data = {
    "version": 1,
    "intent": {
        "source": {"chain": "ethereum", "asset": "wbtc"},
        "destination": {"chain": "aleo", "asset": "wbtc"},
        "bridgeProtocol": "hyperlane",
        "amount": amount,
        "sender": sender,
        "recipient": recipient,
    },
    "route": {"id": route.id, "registryVersion": bridge.registry.version},
    "source": {"transactionId": source_tx_id},
}
progress = bridge.recover(recovery_data)
print(progress.next, progress.error)
```

The dictionary uses the SDK's checkpoint format, but is constructed from the
original transfer details rather than loaded from a file. The installed
registry must describe the original route. Recovery reads the source receipt
to recover the Hyperlane message ID; `bridge.wait(progress)` then checks for
delivery on Aleo. Do not submit another deposit to restart monitoring.

Other routes may need additional information:

- **Private USDCx deposits:** recover the original commitment data from the
  Ethereum deposit and retain the original secret nonce for the claim. A lost
  nonce cannot be reconstructed from public chain data; the recipient cannot
  complete the claim through this flow without it.
- **Aleo-origin transfers:** recovery can track source confirmation, but some
  delivery checks depend on the destination balance recorded before submission.
  Without that information, confirm receipt on the destination chain; the SDK
  may continue reporting delivery as pending.
- **Transactions never broadcast:** an unsubmitted proof cannot be recovered
  from the chain. First establish that no source transfer was submitted before
  starting another one.

The SDK does not yet reconstruct every route from a transaction hash alone.
The example above applies specifically to confirmed Ethereum-to-Aleo Hyperlane
transfers; do not reuse its recovery data for a different route.

Checkpoints exclude private keys, record plaintext, Circle attestation bodies,
and private-mint secret nonces. They can contain a proved Aleo transaction
awaiting broadcast. `FileCheckpointStore` writes files with mode `600` through
an atomic rename; applications using another storage system can save the
`Checkpoint` received by `on_checkpoint` using its `to_json()` method.

## USDC Bridging Guide

USDC sent from Ethereum arrives on Aleo as USDCx. Before bridging, decide
whether the funds need to be publicly visible, whether the deposit can reveal
the recipient's Aleo address, and whether the recipient will be available to
claim them.

### Choose how the recipient will use the funds

**Public Bridge** delivers a public balance for payments and applications that use publicly visible
balances. Anyone can read the recipient's public USDCx balance. Delivery
requires no further action from the recipient.

**Public Bridge to Private Balance** delivers a private record for private payments and applications that
accept private funds. Its contents are encrypted rather than stored in a
public balance. The bridge can deliver this record without requiring the
recipient to return and claim it. However, the Ethereum deposit still reveals
the recipient's Aleo address: receiving funds privately does not, by itself,
hide who received the deposit.

**Private Bridge** suits transfers where the Ethereum
deposit should not reveal the recipient's Aleo address. The recipient must
return to claim the funds and keep a secret needed for that claim. Choose this
option only when the recipient can complete that extra step.

The Ethereum sender and deposited USDC amount remain public in all three
cases. Concealing the Aleo address does not conceal the Ethereum transaction.

| Recipient's needs | Delivery choice | Quote setting |
| --- | --- | --- |
| A public balance, with no claim step | Public Bridge | `mint_mode="public"` (default) |
| Funds for private use, with no claim step; the deposit may reveal the Aleo address | Public Bridge to Private Balance | `mint_mode="record"` |
| Funds for private use without publishing the Aleo address in the deposit; the recipient can claim them later | Private Bridge | `mint_mode="private"` |

### Bridge Privately: Hide the balance

The sender needs USDC for the transfer and ETH for Ethereum transaction fees.
Review the quoted fees and amount the recipient will receive before submitting
the deposit. Once delivery finishes, the recipient can use the USDCx without
signing a separate claim transaction.

Using the client from Setup, request a private record with `mint_mode="record"`.
Change it to `"public"` to receive a public balance instead:

```python
usdc_quote = bridge.quote(
    source_chain="ethereum",
    source_asset="usdc",
    destination_chain="aleo",
    destination_asset="usdcx",
    amount="2",
    recipient=bridge.aleo_address(),
    mint_mode="record",  # Use "public" to receive a public balance.
)
print(usdc_quote.fees, usdc_quote.amount_out)
```

Submit the deposit after accepting the quote, then monitor it until delivery
finishes. The recipient does not need to take part in this step:

```python
usdc_progress = bridge.execute(usdc_quote.plan)
usdc_progress = bridge.wait(usdc_progress)
print(usdc_progress.next, usdc_progress.error)
```

### Bridge Privately: Hide the balance and recipient

With Private Bridge, submitting the Ethereum deposit is only the
first step. The funds become available for use on Aleo after the recipient
claims them. The recipient needs both their Aleo account and the original
secret nonce—a random value that conceals the address in the deposit.

**Save the nonce before sending funds and retain it until the claim completes.**
Losing it prevents the recipient from completing the claim through this flow.
The bridge journal does not save it, so restoring the journal alone is not
enough. The default `0scalar` provides no secrecy; use a securely generated
nonce for this option.

This is an alternative to the automatic delivery example above. Store the
nonce as an Aleo scalar literal in `BRIDGE_MINT_SECRET_NONCE`. The example uses
the configured Aleo account as the recipient, so that account can claim the
funds later. Request a quote with `mint_mode="private"`:

```python
secret_nonce = os.environ["BRIDGE_MINT_SECRET_NONCE"]
private_quote = bridge.quote(
    source_chain="ethereum",
    source_asset="usdc",
    destination_chain="aleo",
    destination_asset="usdcx",
    amount="2",
    recipient=bridge.aleo_address(),
    mint_mode="private",
    secret_nonce=secret_nonce,
)
print(private_quote.fees, private_quote.amount_out)
```

After the deposit is confirmed and Circle has attested it, the recipient can
claim the USDCx. `progress.next == "complete"` indicates that the claim is
ready. Claiming requires the recipient to authorize an Aleo transaction and
can incur an Aleo fee:

```python
private_progress = bridge.execute(private_quote.plan, secret_nonce=secret_nonce)
private_progress = bridge.wait(private_progress)

if private_progress.next == "complete":
    private_progress = bridge.complete(private_progress, secret_nonce=secret_nonce)
    private_progress = bridge.wait(private_progress)

print(private_progress.next, private_progress.error)
```

If the application closes or monitoring times out, the deposit may still be
in progress. Recover the existing transfer rather than sending USDC again.
Keep the same nonce for any remaining deposit or claim step; see
[Recover Funds](#recover-funds).

## Shielding Assets

Shielding moves tokens from a public Aleo balance into an encrypted private
record for private payments and applications. Unshielding moves them back to
a public balance. Shielding does not erase the public history of a bridge
deposit.

- **Hyperlane** delivers assets to public balances and requires public funds
  for withdrawals. Shield after delivery for private use on Aleo; unshield
  before bridging back to Ethereum or Solana.
- **xReserve** can deliver USDCx as a public balance or a private record.
  Private delivery needs no additional shielding. A private USDCx withdrawal
  spends the record directly, so it needs no unshielding. Use `mode="public"`
  when withdrawing from a public balance.

### Shield a public balance

After bridge delivery is confirmed, shield the amount needed for private use.
This example requires at least 0.01 bridged SOL in the account's public Aleo
balance:

```python
receipt = bridge.shield("aleo/sol", amount="0.01").delegate()
print(receipt.transaction_id)
```

### Unshield for a Hyperlane withdrawal

Spending private funds requires an unspent record. Pass `record=` to select
one explicitly, or use the hosted scanner to find it. **Scanner registration
shares the account's view key with the service**, allowing it to decrypt the
account's records. The SDK never registers automatically. If that disclosure
is acceptable:

```python
registration = bridge.aleo.records.register(bridge.aleo.default_account)
if not registration.get("ok"):
    raise RuntimeError(f"Scanner registration failed: {registration}")
```

Once the scanner has indexed a sufficient unspent record, unshield the amount
needed for the withdrawal:

```python
receipt = bridge.unshield("aleo/sol", amount="0.01").delegate()
print(receipt.transaction_id)
```

Wait for the unshielding transaction to confirm before bridging those funds.
The same record-discovery requirement applies to private xReserve withdrawals;
supplying `record=` avoids the hosted scanner.

Shielding and unshielding each submit a separate Aleo transaction with a fee.
A failed conversion does not reverse or repeat the bridge transfer.

## Understand transfer costs

Read `quote.fees` before submitting. Each fee names the asset and chain in
which it is paid, its amount in display units, and whether it is estimated.
`quote.amount_out` reports the destination amount when it can be determined.
Estimates are not a guarantee of the final amount received.

- Ethereum transfers need ETH for gas in addition to the asset being sent.
  An approval can add a transaction. `Ethereum` applies a minimum EIP-1559
  priority fee of 0.1 gwei by default; `min_priority_fee_wei` changes it.
- Aleo-origin Hyperlane transfers pay the relayer in credits as well as paying
  an Aleo transaction fee. Execution refreshes the relayer payment before
  proving unless `gas_payment_microcredits` is supplied explicitly. That
  override is in microcredits: 1,000,000 microcredits equal one credit.
- Solana transfers need SOL for transaction fees, relay costs, and creation of
  Hyperlane message accounts, in addition to the amount being transferred.
- xReserve withdrawal quotes use the registry's configured fee of 2 USDCx and
  mark it as estimated. The burn must exceed that amount. The actual delivery
  depends on the provider's fee.

Use a reliable Ethereum RPC endpoint and serialize transfers from the same
Ethereum account so concurrent submissions do not compete for a nonce.

## Agents and MCP

The lifecycle is also exposed as tool definitions for an agent runtime.

```python
from aleo_bridge import bridge_tools, dispatch_tool

tools = bridge_tools()
result = dispatch_tool(bridge, "bridge_quote", {
    "source_chain": "ethereum", "source_asset": "usdc", "destination_chain": "aleo",
    "amount": "2", "recipient": bridge.aleo_address(),
})
```

Reads (`bridge_status`, `bridge_list_assets`, `bridge_list_routes`,
`bridge_quote`, `bridge_get_progress`, `bridge_pending`) move nothing. Writes
(`bridge_execute`, `bridge_resume`, `bridge_complete`, `bridge_shield`,
`bridge_unshield`) require `confirm: true`; without it they return the quote
or the recovered progress and a `how_to_confirm` field, and move nothing.
`bridge_tools(include_writes=False)` omits the writes entirely.

`python -m aleo_bridge` prints the generated agent guide, and
`python -m aleo_bridge.mcp` serves the same tools over stdio
(`pip install 'aleo-bridge-sdk[mcp]'`).

## Development

```sh
cd bridge-sdk && python -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q -m "not live"          # hermetic suite
.venv/bin/python codegen/gen_context.py --check      # AGENTS.md is generated from docstrings
```
