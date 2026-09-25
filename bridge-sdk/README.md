# Aleo Bridge SDK

Bring assets from Ethereum or Solana to Aleo for payments and applications,
or withdraw them back to their source chain. Hyperlane carries ETH, WBTC,
USDT, and SOL; Circle xReserve connects Ethereum USDC with Aleo USDCx.

Review the cost before sending, monitor whether the recipient received the
funds, and recover an interrupted transfer without making another deposit.
For USDC, choose between public delivery, a private balance, and a private
balance that also conceals the Aleo recipient in the Ethereum deposit.

## Install

The bridge package includes the clients needed for Ethereum, Solana, and
Aleo, along with delegated proving for Aleo transactions. Install it in a
Python 3.10 or later environment before following the examples:

```sh
python -m pip install aleo-bridge-sdk
```

Import the package as `aleo_bridge`.

## Supported pairs

The bridge supports transfers to and from Aleo, connecting assets on Ethereum
and Solana with their corresponding assets on Aleo. Funds can be brought to
Aleo for use in payments and applications, then bridged back to the supported
external chain.

The table below shows the supported mainnet pairs in both directions, including
the asset sent, the asset received, and the provider handling the transfer.
Choose the source chain and asset first, then find the corresponding destination:

| Source chain | Source asset | Destination chain | Destination asset | Provider |
| --- | --- | --- | --- | --- |
| Aleo | ETH | Ethereum | ETH | Hyperlane |
| Aleo | WBTC | Ethereum | WBTC | Hyperlane |
| Aleo | USDT | Ethereum | USDT | Hyperlane |
| Aleo | SOL | Solana | SOL | Hyperlane |
| Aleo | USDCx | Ethereum | USDC | Circle xReserve |
| Ethereum | ETH | Aleo | ETH | Hyperlane |
| Ethereum | WBTC | Aleo | WBTC | Hyperlane |
| Ethereum | USDT | Aleo | USDT | Hyperlane |
| Ethereum | USDC | Aleo | USDCx | Circle xReserve |
| Solana | SOL | Aleo | SOL | Hyperlane |

## Setup

To bridge assets to and from Aleo, create a `Bridge` client with the accounts
and network connections needed for the transfer. The client uses those
connections to check balances, quote costs, submit transactions, and monitor
whether the recipient has received the funds.

A bridge transfer can still be in progress when the application closes or
loses its connection. A journal is a local record of the transfer's details
and submitted transaction IDs, saved as checkpoints while the transfer
advances. After a restart, these checkpoints let the application identify the
original transfer, check what completed, and continue unfinished steps without
starting another transfer.

To retain this recovery information between sessions, attach a
`FileCheckpointStore` when creating the `Bridge` client, as shown below.

The setup below creates an Aleo account connection, adds Ethereum, and attaches
a journal. Solana can be added afterward for transfers involving SOL.

Run connected examples in the same Python session. Unless marked as observed,
output comments are illustrative; addresses, transaction IDs, and fees vary
by transfer.

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
recipient = bridge.aleo_address()  # The configured account's aleo1… address.
```

Setup does not move funds. Keys stay in the application's process for
signing; RPC endpoints provide balances, fees, and confirmation status.

To send or receive SOL through the bridge, include the Solana connection:

```python
from aleo_bridge import Solana

solana = Solana(
    os.environ["SOLANA_RPC_URL"],
    private_key=os.environ["SOLANA_PRIVATE_KEY"],
)
bridge = Bridge(aleo, ethereum=ethereum, solana=solana, checkpoints=store)
```

Solana wallets and command-line tools export account keys in different
formats. A wallet commonly exports a base58 string, while a Solana CLI keypair
file contains a JSON array of 64 integers. Both represent signing keys; use
the key for the account that holds the SOL needed for the transfer and fees,
rather than its public address. The bridge accepts either format without
manual conversion.

Applications already using Web3.py or solana-py can reuse their clients and
signers through `Ethereum(w3=..., signer=...)` or
`Solana(client=..., signer=...)`. Omit the side-chain signer for an application
that only reads balances, quotes, or status.

Aleo transactions require a cryptographic proof before they can be submitted.
The bridge uses Aleo's delegated proving by default: a proving service generates
the proof, so the application does not need to perform that computation. The
service receives the transaction contents, but the private key stays with the
application.

Applications that need to keep transaction contents out of the proving service
can configure local proving with `proving="local"` on `execute`, `resume`, or
`complete`. The application's machine then generates the proof and may need
to download proving parameters.

## Find supported assets and routes

A route identifies an asset pair that can be bridged between two chains.
Finding routes first helps an application offer transfers supported by the
configured network.

The example below lists routes from Ethereum to Aleo. It reads the package's
catalog without contacting a network or requesting a signature.

```python
routes = bridge.routes(source_chain="ethereum", destination_chain="aleo")
for route in routes:
    print(route.id, route.protocol, route.availability)
    # Example: hyperlane:ethereum/wbtc->aleo/wbtc hyperlane active
```

Use the chain and asset names from these results when requesting a quote.
For deployment inspection, `include_unavailable=True` also returns entries
that cannot yet be used. Offer a transfer only when `route.active` is true.

## Move an asset across chains

The following walkthrough sends 0.001 WBTC from Ethereum to the Aleo
recipient configured in Setup. It covers the full transfer: reviewing the
cost, submitting the deposit, and monitoring delivery.

The sender needs WBTC for the transfer and ETH for Ethereum transaction fees.
Complete the steps in order, retaining the quote and progress returned along
the way.

1. Get a quote with `quote` and review fees and the expected amount received.
2. Accept the quote by submitting its plan with `execute`.
3. Monitor delivery with `wait`.
4. If another action is requested, use `resume` for unfinished source work or
   `complete` for a private USDCx claim.

### 1. Get a bridge quote

A bridge quote shows the expected cost and amount received so the sender can
decide whether to proceed. Requesting one reads current network information
without signing or sending funds.

For this transfer, specify WBTC on both chains, the amount to send, and the
Aleo recipient. Then inspect the fees and expected output. Amounts use display
units: `"0.001"` means 0.001 WBTC.

```python
quote = bridge.quote(
    source_chain="ethereum",
    source_asset="wbtc",
    destination_chain="aleo",
    destination_asset="wbtc",
    amount="0.001",  # 0.001 WBTC, not atomic units.
    recipient=recipient,  # Aleo address receiving the WBTC.
)

for fee in quote.fees:
    print(fee.kind, fee.amount, fee.asset_id, "estimated" if fee.estimated else "")
    # Example: network 0.0001 ethereum/eth estimated (illustrative fee).
print(quote.amount_out)  # "0.001" WBTC for this Hyperlane quote.
```

Use `bridge_protocol="hyperlane"` or `"xreserve"` if more than one provider
matches the transfer. A route selected from `bridge.routes(...)` can also be
passed as `route=`. The API can infer a destination asset when only one route
fits; naming both assets keeps the intended transfer clear.

Keep `quote.plan` unchanged when accepting the quote so the submitted transfer
uses the reviewed asset, amount, and recipient. Request a new quote to change
those details.

### 2. Submit a bridge transaction from the source chain

The source transaction deposits the sender's funds into the bridge so they
can be delivered on the destination chain. This is the step that commits funds
and pays source-chain transaction fees.

After accepting the quote, submit its plan. A token transfer may first require
an approval, so one bridge transfer can involve several transactions. USDT may
also need an existing allowance reset before approval.

Call `execute` once and retain the returned progress to monitor the transfer.
The journal saves checkpoints as submission advances, allowing the application
to recover if it closes between approval and deposit.

```python
def show_checkpoint(checkpoint):
    print("Checkpoint ID:", checkpoint.id)  # Copy this ID for store.load(...).

progress = bridge.execute(quote.plan, on_checkpoint=show_checkpoint)
print(progress.receipt.source_tx_id)  # Ethereum transaction hash: 0x… (64 hex digits).
```

Once a source transaction has been submitted, do not call `execute` again for
the same transfer. Use the returned progress while the process stays alive, or
recover from the latest checkpoint after an interruption.

### 3. Monitor bridge progress

After submission, the bridge still needs to confirm the deposit and deliver
the funds. Monitoring establishes whether the recipient can use them or needs
to take another action, such as claiming private USDCx.

Pass the progress returned by submission to `wait`, then inspect the next
action. Monitoring does not sign or send another transaction.

```python
progress = bridge.wait(progress)
print(progress.next)  # "done" when delivered; otherwise inspect the next action.
```

Read `progress.next` to decide whether the recipient can use the funds or
another action is needed:

| `progress.next` | Caller action |
| --- | --- |
| `done` | Show completion. No further action is required. |
| `failed` | Show `progress.error`. Do not repeat a transaction that already succeeded. |
| `wait` | Call `wait` again; polling stopped at an application-selected status. |
| `resume` | Submit the remaining source operation with `resume`. |
| `complete` | Authorize the private USDCx mint with `complete`. |

A monitoring timeout leaves delivery unresolved; it does not cancel the
deposit. Keep checking the existing transfer instead of sending again.
`wait` allows 20 minutes by default and raises `PollingTimeoutError` with the
latest progress when that time expires.

Use this in place of the `wait` call above to keep that progress for another
check:

```python
from aleo_bridge import PollingTimeoutError

try:
    progress = bridge.wait(progress, timeout_seconds=1200)
except PollingTimeoutError as exc:
    if exc.progress is None:
        raise
    progress = exc.progress

print(progress.next, progress.receipt.source_tx_id)  # Example while pending: wait 0x…
```

If an approval succeeded but the deposit remains unfinished, continue the
transfer with `resume` when requested. It can submit the missing transaction
without repeating confirmed work. After deciding to continue:

```python
if progress.next == "resume":
    progress = bridge.resume(progress)
    progress = bridge.wait(progress)
```

A Private Bridge USDCx deposit needs the recipient to claim the funds with
their Aleo account and original secret nonce. The [USDC Bridging Guide](#usdc-bridging-guide)
explains that choice and the `complete` call used to claim.

## Recover Funds

An application can lose its connection or close while a bridge transfer is
still in progress. Recovery finds that existing transfer, checks whether the
funds arrived, and identifies any remaining action. It does not refund or
repeat the deposit.

Start with the saved journal if one is available. The examples below show how
to recover a known checkpoint, find a transfer in the journal, or reconstruct
a supported transfer from chain history when no files remain.

A bridge journal saves checkpoints—records identifying the transfer and its
submitted transactions—so monitoring can continue after a restart. If no files
remain, some transfers can be recovered using their on-chain history.

### Recover with a bridge journal

A saved checkpoint identifies the transfer and the work already submitted,
allowing monitoring to continue after a restart. This path loads that record
and asks the network for the transfer's current progress.

Reopen the journal with the same network and connections. Select the latest
checkpoint ID from the submission callback or journal listing; the ID can
change as the transfer advances.

For a concrete example, the repository includes a [checkpoint from a confirmed
Solana-to-Aleo transfer](examples/checkpoints/cWFKiumuvVuvrxM8xtunZxNM4FNUppSdyNm7HEqKjV3ZmENebD4DAf44kbyvq9fKJ61VzNrH3tYpLJgUrY8MEGW.json).
It was reconstructed from the original transaction and checked against mainnet.
Its ID is the source transaction's Solana signature. Run this example from the
`bridge-sdk` directory; it reads the saved file and network without sending
funds:

```python
from aleo import Aleo, HTTPProvider
from aleo_bridge import Bridge, FileCheckpointStore, Solana

example_store = FileCheckpointStore("examples/checkpoints")
example_bridge = Bridge(Aleo(HTTPProvider()), solana=Solana())
checkpoint_id = (
    "cWFKiumuvVuvrxM8xtunZxNM4FNUppSdyNm7HEqKjV3ZmENebD4DAf44kbyvq9fKJ61VzNrH3tYpLJgUrY8MEGW"
)
checkpoint = example_store.load(checkpoint_id)
if checkpoint is None:
    raise ValueError(f"No saved checkpoint for {checkpoint_id}")

progress = example_bridge.recover(checkpoint)
print(progress.next, progress.error)  # Observed on 2026-09-25: wait None.
```

This checkpoint belongs to an existing transfer, not the account configured in
Setup. Recovery reported a confirmed source transaction with delivery still
pending when checked. To recover an application's own transfer, use its
checkpoint store and the ID returned by that checkpoint. Ethereum checkpoint
IDs can be `0x` transaction hashes or Hyperlane message IDs; Aleo transaction
IDs begin with `at1`.

Recovery reports whether to keep waiting, finish a submission, or claim the
funds. It does not authorize those actions. Follow `progress.next`:

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

The journal can contain several transfers, and the checkpoint ID may not be
at hand. Listing the saved entries helps identify the intended transfer from
its asset, amount, and recipient.

Load the entries, inspect their details, and select one for recovery. This
step reads local files without contacting a network:

```python
result = store.load_checkpoints()

for checkpoint in result.checkpoints:
    print(checkpoint.id)  # Example format: 0x followed by 64 hex digits.
    print(checkpoint.intent["source"])  # {"chain": "ethereum", "asset": "wbtc"}
    print(checkpoint.intent["amount"])  # "0.001" for the WBTC example.

for error in result.errors:
    print(error.path, error.error)  # Identifies an unreadable file and the reason.
```

Select the intended transfer from `result.checkpoints` and pass it to
`recover` to check its current status. Entries are ordered by file modification
time, oldest first. An entry in `result.errors` means a file could not be
loaded; it does not mean the associated transfer failed.

### Recover without saved files

Losing the journal does not remove a submitted transfer from the chain. For
supported routes, the original transaction and its transfer details can be
used to restore monitoring without any saved files.

First locate the bridge deposit or dispatch in the source wallet's history or
a block explorer. Check its status, asset, amount, sender, and destination
recipient. A token approval alone does not identify a completed deposit.

For a **confirmed Ethereum-to-Aleo Hyperlane transfer**, those details are
enough to reconstruct the recovery data in memory. The example below recovers
a WBTC transfer on mainnet. Enter the original transfer's details from the
explorer; the amount is in WBTC, not its smallest units.

Use the Aleo and Ethereum connections from Setup, but construct the bridge
without a checkpoint store. This example reads the network and does not load
or write a journal, sign a transaction, or send funds:

```python
bridge = Bridge(aleo, ethereum=ethereum)
source_tx_id = input("Confirmed Ethereum bridge transaction hash: ").strip()  # 0x + 64 hex digits.
sender = input("Original Ethereum sender address: ").strip()  # 0x + 40 hex digits.
recipient = input("Original Aleo recipient address: ").strip()  # aleo1… address from the deposit.
amount = input("Original amount in WBTC: ").strip()  # Example: "0.001".

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
print(progress.next, progress.error)  # Example: wait None; done None after delivery.
```

This restores monitoring from the original transaction without a saved file.
Use `bridge.wait(progress)` to check delivery on Aleo; no new deposit is needed.
The installed SDK must still support the original route. The recovery data
above supplies the same transfer details that a journal would have retained.

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

Keep any private-mint nonce separately: restoring a checkpoint cannot replace
that claim secret. Checkpoints exclude private keys and private record
plaintext, but can include an authorized Aleo transaction awaiting broadcast.
Protect the journal accordingly. Applications with existing storage can save
checkpoints from `on_checkpoint` using `checkpoint.to_json()`.

## USDC Bridging Guide

USDC sent from Ethereum arrives on Aleo as USDCx, where it can be used in
payments and applications. The bridge offers three delivery options with
different visibility and claim requirements.

First choose whether the recipient needs a public or private balance and
whether the deposit may reveal their Aleo address. The examples then show
private delivery with and without a separate recipient claim.

### Choose how the recipient will use the funds

The choice affects both how the funds can be used on Aleo and what the
Ethereum deposit reveals. Compare the options below before requesting a
quote; the table maps each choice to its `mint_mode` setting.

**Public Bridge** delivers a public balance for payments and applications
that use publicly visible balances. Anyone can read the recipient's public USDCx balance. Delivery
requires no further action from the recipient.

**Public Bridge to Private Balance** delivers a private record for payments
and applications that accept private funds. Its contents are encrypted rather than stored in a
public balance. The bridge can deliver this record without requiring the
recipient to return and claim it. However, the Ethereum deposit still reveals
the recipient's Aleo address: receiving funds privately does not, by itself,
hide who received the deposit.

**Private Bridge** suits transfers where the Ethereum deposit should not
reveal the recipient's Aleo address. The recipient must
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

Public Bridge to Private Balance delivers an encrypted USDCx record ready
for private payments. The recipient does not need to return for a separate
claim, but the Ethereum deposit still identifies their Aleo address.

This example requests a quote for that delivery option, submits the deposit,
and monitors it until the funds arrive.

The sender needs USDC for the transfer and ETH for transaction fees. Review
the cost and expected amount received before submitting the deposit.

Using the client from Setup, request a private record with `mint_mode="record"`.
Change it to `"public"` to receive a public balance instead:

```python
usdc_quote = bridge.quote(
    source_chain="ethereum",
    source_asset="usdc",
    destination_chain="aleo",
    destination_asset="usdcx",
    amount="2",  # 2 USDC, not 2 atomic units.
    recipient=bridge.aleo_address(),
    mint_mode="record",  # Use "public" to receive a public balance.
)
print(usdc_quote.fees)  # Fee entries include asset_id, amount, and estimated.
print(usdc_quote.amount_out)  # Expected USDCx amount in display units.
```

Submit the deposit after accepting the quote, then monitor it until delivery
finishes. The recipient does not need to take part in this step:

```python
usdc_progress = bridge.execute(usdc_quote.plan)
usdc_progress = bridge.wait(usdc_progress)
print(usdc_progress.next, usdc_progress.error)  # done None when delivery completes.
```

### Bridge Privately: Hide the balance and recipient

Private Bridge conceals the recipient's Aleo address in the Ethereum deposit
and delivers funds as a private balance. Unlike automatic delivery, it requires
the recipient to claim the funds before spending them.

The flow has three stages: retain a secret nonce, submit the deposit using
that nonce, and claim the USDCx when it is ready. The nonce is a random value
used to conceal the address; the claim requires both the original nonce and
the recipient's Aleo account.

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
secret_nonce = os.environ["BRIDGE_MINT_SECRET_NONCE"]  # Aleo scalar: decimal digits + "scalar".
private_quote = bridge.quote(
    source_chain="ethereum",
    source_asset="usdc",
    destination_chain="aleo",
    destination_asset="usdcx",
    amount="2",  # 2 USDC, not 2 atomic units.
    recipient=bridge.aleo_address(),
    mint_mode="private",
    secret_nonce=secret_nonce,
)
print(private_quote.fees)  # Review Ethereum fees before submitting.
print(private_quote.amount_out)  # Expected USDCx amount in display units.
```

After the deposit is confirmed and Circle has attested it, the recipient can
claim the USDCx. `progress.next == "complete"` indicates that the claim is
ready. Claiming requires the recipient to authorize an Aleo transaction and
can incur an Aleo fee:

```python
private_progress = bridge.execute(private_quote.plan, secret_nonce=secret_nonce)
private_progress = bridge.wait(private_progress)

if private_progress.next == "complete":  # The deposit is ready for the recipient to claim.
    private_progress = bridge.complete(private_progress, secret_nonce=secret_nonce)
    private_progress = bridge.wait(private_progress)

print(private_progress.next, private_progress.error)  # done None after the claim confirms.
```

If the application closes or monitoring times out, the deposit may still be
in progress. Recover the existing transfer rather than sending USDC again.
Keep the same nonce for any remaining deposit or claim step; see
[Recover Funds](#recover-funds).

## Shielding Assets

Assets already held on Aleo can be moved between a public balance and an
encrypted private record. Shielding prepares funds for private payments;
unshielding makes them public when a bridge withdrawal requires it. Earlier
public deposits remain visible.

Whether either step is needed depends on the bridge provider. Check the
requirements below, then shield or unshield only the amount needed.

- **Hyperlane** delivers assets to public balances and requires public funds
  for withdrawals. Shield after delivery for private use on Aleo; unshield
  before bridging back to Ethereum or Solana.
- **xReserve** can deliver USDCx as a public balance or a private record.
  Private delivery needs no additional shielding. A private USDCx withdrawal
  spends the record directly, so it needs no unshielding. Use `mode="public"`
  when withdrawing from a public balance.

### Shield a public balance

Shielding lets the account use publicly held tokens in private payments and
applications. The funds must already be available in its public Aleo balance.

After bridge delivery confirms, select the amount to shield and submit the
conversion. This example shields 0.01 bridged SOL:

```python
receipt = bridge.shield("aleo/sol", amount="0.01").delegate()  # Shield 0.01 SOL on Aleo.
print(receipt.transaction_id)  # Aleo transaction ID, beginning with at1.
```

### Unshield for a Hyperlane withdrawal

Hyperlane withdrawals spend public balances, so funds held in a private
record must be unshielded first. This step makes the withdrawal amount public
on Aleo; it does not yet send funds to another chain.

First select an unspent record with `record=`, or use the hosted scanner to
find one. Then unshield the required amount and wait for confirmation before
starting the bridge withdrawal.

**Scanner registration shares the account's view key with the service**, which
can then decrypt the account's records. Supply the record explicitly to avoid
that disclosure. If hosted discovery is acceptable, register the account:

```python
registration = bridge.aleo.records.register(bridge.aleo.default_account)
if not registration.get("ok"):
    raise RuntimeError(f"Scanner registration failed: {registration}")
```

Once the scanner has indexed a sufficient unspent record, unshield the amount
needed for the withdrawal:

```python
receipt = bridge.unshield("aleo/sol", amount="0.01").delegate()  # Return 0.01 SOL to a public balance.
print(receipt.transaction_id)  # Aleo transaction ID, beginning with at1.
```

Wait for the unshielding transaction to confirm before bridging those funds.
The same record-discovery requirement applies to private xReserve withdrawals;
supplying `record=` avoids the hosted scanner.

Shielding and unshielding each submit a separate Aleo transaction with a fee.
A failed conversion does not reverse or repeat the bridge transfer.

## Understand transfer costs

A bridge transfer can incur transaction fees, relay costs, and provider fees
in addition to the amount sent. Some fees require a different asset—for
example, sending WBTC from Ethereum still requires ETH for gas.

Before accepting a quote, check each fee's currency, keep enough to cover it,
and review the expected amount the recipient will receive. The list below
explains the costs associated with each chain and provider.

`quote.fees` names each fee's asset, chain, amount, and whether it is estimated.
`quote.amount_out` gives the expected destination amount when available.
Estimated costs and received amounts can change before the transfer completes.

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

An agent can help a caller find a route, review its costs, and monitor a
transfer. An application can limit the agent to those read operations or also
allow it to submit transactions after explicit confirmation.

The example below exposes the bridge tools and requests a USDC quote. It does
not submit a deposit. Afterward, choose whether the application should expose
submission tools or connect through MCP.

```python
from aleo_bridge import bridge_tools, dispatch_tool

tools = bridge_tools()
result = dispatch_tool(bridge, "bridge_quote", {
    "source_chain": "ethereum", "source_asset": "usdc", "destination_chain": "aleo",
    "amount": "2", "recipient": bridge.aleo_address(),
})
```

For an agent that only advises or monitors, use
`bridge_tools(include_writes=False)`. Status, route, quote, and recovery tools
can then inspect transfers without moving funds.

To permit a deposit, claim, resumed submission, shielding, or unshielding,
the corresponding write tool requires `confirm: true`. Without confirmation,
it returns the quote or recovered progress and `how_to_confirm` instructions
instead of submitting a transaction.

Use `python -m aleo_bridge` to obtain the package's agent instructions.
For an MCP client, install `aleo-bridge-sdk[mcp]` and configure
`python -m aleo_bridge.mcp` as its stdio server.

## Development

The development setup supports checking SDK changes without submitting live
transactions. It includes the offline test suite and a check that the generated
agent guide matches the SDK.

Create an environment, install the development dependencies, then run both
checks:

```sh
cd bridge-sdk && python -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q -m "not live"          # hermetic suite
.venv/bin/python codegen/gen_context.py --check      # AGENTS.md is generated from docstrings
```
