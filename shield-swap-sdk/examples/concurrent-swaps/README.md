# Concurrent swaps in Python

Prepare six balanced USDCx records, submit six USDCx → ETH swaps at the same
time, and claim their outputs. The complete runnable example is
[concurrent_swaps.py](./concurrent_swaps.py); the sections below follow it in
order.

A private trade spends one record. Six trades started together therefore need
six records that each cover a trade, and the six swaps must not pick the same
record. The example reshapes the account's available USDCx records into six
equal lots, then lets each swap select and reserve its own record.

## Run the example

From a Python SDK checkout, with `SHIELD_SWAP_PRIVATE_KEY` set to a funded
**testnet** account with DEX access:

```sh
python -m pip install ./shield-swap-sdk

# Reads inventory, lists the maintenance operations, and quotes a trade. No submissions.
python shield-swap-sdk/examples/concurrent-swaps/concurrent_swaps.py

# Prepares the records, submits six swaps, and claims each successful output.
python shield-swap-sdk/examples/concurrent-swaps/concurrent_swaps.py --execute
```

Preparing the records can join and split **the account's entire available
USDCx inventory**; records held by other operations are left out of the plan.
At least 0.6 USDCx must be available. Execution uses delegated proving with
the prover's fee master paying network fees. `--count` and `--amount` change
the batch size and the amount per trade.

## 1. Bind one journal to the client

Record coordination lives on the facade. Building the `Aleo` client with
`record_management` enabled makes every swap reserve its input record before
proving, save the proved transaction before broadcast, and record the
submission. Trades and inventory maintenance then share one journal.

```python
from aleo import Aleo, HTTPProvider
from aleo.inventory import RecordManagementConfig, SqliteRecordInventoryStore

store = SqliteRecordInventoryStore(directory / "inventory.sqlite")
aleo = Aleo(HTTPProvider("https://edge.provable.com/api", network="testnet"),
            record_management=RecordManagementConfig(enabled=True, store=store, chain_id="aleo:testnet"))
```

The SDK's default store is in memory and lasts for one process. The example
uses the SDK's SQLite store so reservations and proved transactions survive a
restart and several processes on the machine share one journal; its layout
matches Veil's store, so a Veil client can read the same file.
`JsonFileRecordInventoryStore` is the alternative when a journal that can be
read in an editor matters more than throughput. A custom store, such as a
database with a uniqueness constraint on held record nonces, extends the same
guarantees across machines.

## 2. Refuse to start over unresolved work

Before planning, the example reconciles the journal and checks the swap
journal for unclaimed outputs. A transaction the node has not confirmed keeps
its reservation; a missing confirmation is not evidence that it failed.

```python
journal = aleo.inventory.reconcile_record_inventory()
pending = [entry for entry in journal if entry.status in ("reserved", "prepared", "submitted")]
```

## 3. Plan and review the record layout

`plan_inventory` reads the token's underlying record program, the scanner, and
the deployed join helpers, and returns the joins and splits that reach the
target. Nothing is reserved or signed until the plan is executed.

```python
from aleo.inventory import InventoryTarget

plan = dex.plan_inventory("USDCx", InventoryTarget(records=6, min_record_amount=100_000,
                                                   distribution="balanced", tolerance_bps=0),
                          max_transactions=24)
for step in plan.steps:
    print(step.kind, len(step.inputs), [output.amount for output in step.outputs])
```

`plan.deduction` reports the credits a native split removes from its outputs.
Network fees are separate.

## 4. Execute the plan

Each step reserves its inputs, proves, broadcasts, waits for confirmation, and
waits until the scanner shows the exact outputs before the next step uses
them. An interrupted result names the reason and every submitted transaction;
no accepted step is rolled back.

```python
result = aleo.inventory.rebalance_record_inventory(plan, timeout=600.0, poll_interval=3.0)
if result.status != "complete":
    raise SystemExit(result.reason)
```

## 5. Submit the swaps together

One quote fixes the output floor for every trade; it reserves no pool
liquidity, so a trade can still fail when the price moves. Each `dex.swap`
selects an input record. When another swap reserved that record first, the
swap moves to the next eligible record from the same scan before proving.
An explicit `token_record` is never replaced.

```python
quote = dex.quote(token_in="USDCx", token_out="ETH", amount_in="0.1", slippage_bps=50)
outcomes = submit_concurrent_swaps(dex, quote, count=6)
```

`submit_concurrent_swaps` runs the swaps on a thread pool, waits for each
successful swap's output, and claims it. One failed trade never discards a
sibling's handle, and a failed claim keeps the handle for a later attempt.

## Keep recovery data

The default directory is `.shield-swap/concurrent-swaps/testnet/<address>/`.
Reuse it for the same account.

| File | Purpose |
| --- | --- |
| `inventory.sqlite` | Record reservations and proved transactions awaiting broadcast, for reconciliation. |
| `swaps.jsonl` | Blinded identities and swap handles needed to claim outputs. Treat it as key material. |
| `batch.pending` | Remains after an interrupted batch so a later run cannot repeat it unnoticed. |

Keep these files private and out of version control. **Do not run another
trader for the same account while this example is running.** Several clients
in one process must share the same store object.

After an interruption:

1. Stop the previous process.
2. Rebuild the client with the same account, network, and files, and call
   `aleo.inventory.reconcile_record_inventory()`. A transaction the node has
   not seen keeps its reservation; pass `rebroadcast=True` to resubmit a saved
   proved transaction with its original id.
3. Call `dex.collect_all()` to claim every finalized swap recorded in
   `swaps.jsonl`.
4. Remove `batch.pending` only after every submitted transaction and claim is
   resolved, then preview the inventory again before a new batch.

The example never rebuilds or repeats a swap after an uncertain submission.
