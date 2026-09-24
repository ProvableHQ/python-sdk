# aleo-bridge-sdk

Moves assets between Aleo, Ethereum, and Solana through reviewed Hyperlane and
Circle xReserve deployments.

The package signs with local keys or with signers the caller already holds. It
does not choose a key, store transfer progress unless the caller binds a
checkpoint store, or submit a second transaction after an interruption without
the caller asking for it.

```sh
pip install aleo-bridge-sdk
```

Imports as `aleo_bridge`.

## Supported transfers

| Source | Destination | Asset received | Provider |
| --- | --- | --- | --- |
| Ethereum ETH | Aleo | ETH | Hyperlane |
| Aleo ETH | Ethereum | ETH | Hyperlane |
| Ethereum WBTC | Aleo | WBTC | Hyperlane |
| Aleo WBTC | Ethereum | WBTC | Hyperlane |
| Ethereum USDT | Aleo | USDT | Hyperlane |
| Aleo USDT | Ethereum | USDT | Hyperlane |
| Solana SOL | Aleo | SOL | Hyperlane |
| Aleo SOL | Solana | SOL | Hyperlane |
| Ethereum USDC | Aleo | USDCx | Circle xReserve |
| Aleo USDCx | Ethereum | USDC | Circle xReserve |

Hyperlane routes accept any positive amount. The xReserve deposit requires at
least 2 USDC, and the xReserve withdrawal must burn more USDCx than its fee.
Sepolia USDC to Aleo-testnet USDCx and its reverse are available on testnet.

The registry also lists ALEO and USAD entries for deployment discovery. Those
entries are marked `metadata-required` and cannot be quoted or executed. Solana
routes support native SOL, not USDC or other SPL tokens.

## Create a client

A client holds one Aleo account and, for each side chain it will touch, a
transport and a signer. Reads use the transports only. A signature is requested
only when a fund-moving action runs.

```python
from aleo import Aleo, HTTPProvider
from aleo_bridge import Bridge, Ethereum, Solana

aleo = Aleo(HTTPProvider("https://edge.provable.com/api", network="mainnet"))
aleo.default_account = aleo.account.from_private_key(aleo_private_key)

bridge = Bridge(
    aleo,
    ethereum=Ethereum(ethereum_rpc_url, private_key=evm_private_key),
    solana=Solana(solana_rpc_url, private_key=solana_private_key),
)
```

Local EVM and Solana keys sign inside the caller's process and broadcast
through their configured transports. The bridge client never writes a key to
disk. A side chain the application does not use can be omitted; a transfer
that needs it then fails at `quote` with a `ConfigurationError`.

An existing `web3.Web3` instance or `eth_account` signer can be supplied as
`Ethereum(w3=..., signer=...)`, and an existing solders `Keypair` or solana-py
client as `Solana(client=..., signer=...)`. A transport passed without a
signer gives a read-only connection. A Solana `private_key` accepts the base58
export of a browser wallet or the 64-integer array of a `solana-cli` `id.json`.

Aleo transactions prove through the delegated proving service by default
(`proving="delegate"`), which sees the transaction contents but not the private
key. Pass `proving="local"` to prove on the caller's machine.

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

### 1. Quote the transfer

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
    recipient=aleo_address,
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

### 2. Submit the source transfer

Execution may submit more than one transaction. An ERC-20 route sends an
approval before its bridge deposit, and USDT first resets a non-zero allowance
to zero. The result carries the receipt of the latest submission and the
identifiers of every transaction already sent.

```python
progress = bridge.execute(quote.plan)
print(progress.receipt.source_tx_id)
```

Once a source transaction has been submitted, do not call `execute` again for
the same transfer. Use the returned progress while the process stays alive, or
recover from the latest checkpoint after an interruption.

### 3. Follow the transfer

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

`wait` gives up after `timeout_seconds` (default 1200) by raising
`PollingTimeoutError`. A timeout is not a failure: the transfer is still in
flight, the exception carries the latest `progress`, and `wait` can be called
again. A source transaction that was dropped or replaced by the network before
it mined is reported as `failed` with an error stating that no funds moved;
`recover` then confirms from chain history that nothing landed and returns a
`resume` step for it.

`resume` applies when work such as an ERC-20 approval succeeded but the
deposit itself was not submitted. It does not repeat the confirmed approval.

```python
if progress.next == "resume":
    progress = bridge.wait(bridge.resume(progress))
```

`complete` applies only to an Ethereum USDC deposit that selected a private
USDCx mint. Circle first attests the deposit. The Aleo recipient then submits
one destination transaction that creates the private record.

```python
if progress.next == "complete":
    progress = bridge.wait(bridge.complete(progress, secret_nonce=secret_nonce))
```

## Recover after an interruption

A checkpoint contains the public transfer intent and the transaction
identifiers needed to find the transfer again. It excludes private keys, Aleo
record plaintext, proofs, Circle attestations, and private-mint secret nonces.

Binding a `FileCheckpointStore` to the client saves a checkpoint at every
submission boundary: after each approval, after an Aleo transaction is proved
but before it is broadcast, and after each broadcast. Files are written with
mode 600 through an atomic rename.

```python
from aleo_bridge import Bridge, FileCheckpointStore

store = FileCheckpointStore("~/.aleo-bridge/checkpoints")
bridge = Bridge(aleo, ethereum=ethereum, solana=solana, checkpoints=store)
```

An application that prefers its own storage passes `on_checkpoint=` to
`execute`, `resume`, and `complete` instead; the callback receives a
`Checkpoint` with `to_json()` and `from_json()`.

After a restart, `pending()` lists every unfinished transfer in the store
without contacting a network, and `recover` re-reads chain and provider state
for one of them. Neither signs, submits, or repeats a transaction.

```python
for progress in bridge.pending():
    if progress.next == "wait":
        progress = bridge.wait(progress)
    if progress.next == "resume":
        progress = bridge.wait(bridge.resume(progress))
    if progress.next == "complete":
        progress = bridge.wait(bridge.complete(progress, secret_nonce=secret_nonce))
```

The application then handles `progress.next` by the same table above. A
private-mint secret nonce must be stored separately because it is
intentionally absent from the checkpoint. A finished transfer is removed from
the store; a transfer whose source transaction was dropped stays listed until
`resume` replaces it or the application calls `store.delete(progress.receipt.id)`.

## Choose how USDCx arrives

An Ethereum USDC deposit selects, at quote time, how the USDCx is delivered on
Aleo:

| `mint_mode` | Delivery | Caller action after the deposit |
| --- | --- | --- |
| `"public"` (default) | public USDCx balance of the recipient | none; `wait` ends at `done` |
| `"record"` | a private record minted by the relayer | none; `wait` ends at `done` |
| `"private"` | a private record only the recipient can mint | `complete` with the same `secret_nonce` |

A private mint commits to the recipient and a caller-chosen `secret_nonce` on
Ethereum. Only the recipient's Aleo key, presenting the same nonce, can
complete it. The package never stores the nonce; the default value `0scalar`
offers no secrecy, so a caller who wants the commitment to hide the recipient
must choose and keep a nonce of their own.

## Use private assets on Aleo

Hyperlane routes mint wrapped assets into public Aleo balances and spend public
balances when bridging out of Aleo. Shielding and unshielding move the same
asset between that public balance and a private record owned by the account.

### Unshield before bridging out through Hyperlane

An outbound Hyperlane transfer cannot spend a private record directly. Convert
the amount into the account's public balance before quoting and executing the
bridge transfer.

```python
receipt = bridge.unshield("aleo/sol", amount="0.01").delegate()
print(receipt.transaction_id)
```

Private USDCx is burned directly by the xReserve withdrawal. It does not need
to be unshielded first.

### Shield an asset for private use on Aleo

After a Hyperlane transfer arrives, its Aleo balance is public. Convert any
amount that should be held or spent privately into a record owned by the
account.

```python
receipt = bridge.shield("aleo/sol", amount="0.01").delegate()
print(receipt.transaction_id)
```

Shielding and unshielding each submit one Aleo transaction and pay an Aleo
transaction fee. They are separate from bridge delivery: a failed conversion
does not repeat or reverse the completed cross-chain transfer.

### Record selection shares the account's view key

Any action that spends a private record the caller did not name explicitly
(`unshield`, and the default private xReserve withdrawal) finds the record
through the hosted record scanner. The scanner returns nothing for an account
that has not been registered, and registering shares that account's view key
with the scanning service, which can then decrypt every record the account
owns. The package never registers an account on its own. Register once,
explicitly:

```python
bridge.aleo.records.register(bridge.aleo.default_account)
```

Until then, record selection raises a `ConfigurationError` that says so. Pass
`record=` to spend a specific record without the scanner, or use a public
withdrawal (`mode="public"`) when privacy of the burn is not required.

## Costs to plan for

- **xReserve withdrawal fee.** The registry states 2 USDCx. The fee Circle
  actually charges has been observed at about 1.0035 USDC on mainnet and
  testnet, so the quote marks this fee as `estimated` and `amount_out` is a
  lower bound: the recipient may receive more, never less. The burn minimum
  still uses the registry value.
- **Aleo-origin Hyperlane transfers** pay the destination gas in credits, about
  8 to 9 credits per transfer at current rates, on top of the Aleo transaction
  fee. The quote reports the exact amount.
- **Solana-origin transfers** cost about 0.0056 SOL on the source side, of
  which about 0.0037 SOL is rent for Hyperlane's per-message accounts, so a
  small transfer still needs that balance.
- **Ethereum tips.** Public RPC endpoints often suggest a zero priority fee,
  and a zero-tip transaction can sit unmined for hours. Every EIP-1559 tip is
  raised to a floor of 0.1 gwei. `Ethereum(..., min_priority_fee_wei=...)`
  changes the floor. Prefer a dedicated RPC endpoint over a load-balanced
  public one for transfers, and never run two fund-moving Ethereum transfers
  from the same key at the same time.

## Agents and MCP

The lifecycle is also exposed as tool definitions for an agent runtime.

```python
from aleo_bridge import bridge_tools, dispatch_tool

tools = bridge_tools()
result = dispatch_tool(bridge, "bridge_quote", {
    "source_chain": "ethereum", "source_asset": "usdc", "destination_chain": "aleo",
    "amount": "2", "recipient": aleo_address,
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
