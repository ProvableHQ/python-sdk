"""Prepare six balanced USDCx records and swap them for ETH at the same time.

Preview (reads, plans, and quotes; submits nothing):

    python shield-swap-sdk/examples/concurrent-swaps/concurrent_swaps.py

Execute (reshapes the account's available USDCx records, submits six swaps,
claims each output):

    python shield-swap-sdk/examples/concurrent-swaps/concurrent_swaps.py --execute

Requires ``SHIELD_SWAP_PRIVATE_KEY`` naming a funded testnet account with DEX
access. Execution spends 0.1 USDCx per swap by default and pays network fees
through the delegated prover's fee master. See README.md for the recovery
steps after an interruption.
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

from aleo import Aleo, HTTPProvider, testnet
from aleo.inventory import InventoryTarget, RecordManagementConfig, SqliteRecordInventoryStore
from aleo_shield_swap import Journal, ShieldSwap


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare records and run concurrent Shield Swap trades on testnet.")
    parser.add_argument("--execute", action="store_true", help="prepare records, submit swaps, and claim outputs")
    parser.add_argument("--count", type=int, default=6, help="trades to run at once and records to prepare (2-16)")
    parser.add_argument("--amount", default="0.1", help="input amount per trade in token units")
    args = parser.parse_args()
    token_in, token_out, amount_in, count = "USDCx", "ETH", args.amount, args.count
    if not 2 <= count <= 16:
        raise SystemExit("--count must be from 2 through 16")

    ### STEP 1. Bind the account, a durable reservation journal, and a swap journal to one client. ###
    ### Shield Swap trades spend private records. Record management on the Aleo client reserves each
    ### trade's record before proving so trades submitted together never pick the same one.
    private_key = os.environ.get("SHIELD_SWAP_PRIVATE_KEY")
    if not private_key:
        raise SystemExit("Set SHIELD_SWAP_PRIVATE_KEY to a funded testnet account with DEX access")
    key = testnet.PrivateKey.from_string(private_key)
    directory = Path(".shield-swap") / "concurrent-swaps" / "testnet" / str(key.address)
    directory.mkdir(parents=True, exist_ok=True)

    # The SDK's default journal is in memory and lasts one process. A SQLite journal keeps reservations
    # and proved transactions across restarts and coordinates processes on this machine;
    # JsonFileRecordInventoryStore is the inspectable alternative.
    store = SqliteRecordInventoryStore(directory / "inventory.sqlite")
    aleo = Aleo(HTTPProvider("https://edge.provable.com/api", network="testnet"),
                record_management=RecordManagementConfig(enabled=True, store=store, chain_id="aleo:testnet"))
    account = aleo.account.from_private_key(key)
    aleo.default_account = account
    registration = aleo.records.register(account)      # shares the view key with the scanner
    if not registration.get("ok"):
        raise SystemExit(f"Record scanner registration failed: {registration.get('error')}")
    dex = ShieldSwap(aleo)
    dex.journal = Journal(directory / "swaps.jsonl")   # blinded identities and claim handles
    dex.api.authenticate(str(account.address), lambda message: str(key.sign(message.encode())))

    ### STEP 2. Refuse to start over unresolved work. ###
    ### A transaction the node has not confirmed keeps its reservation; a missing confirmation
    ### is not evidence that it failed.
    pending = [entry for entry in aleo.inventory.reconcile_record_inventory()
               if entry.status in ("reserved", "prepared", "submitted")]
    if pending:
        raise SystemExit(f"Reconcile {len(pending)} unresolved operations in {directory / 'inventory.sqlite'} before another batch")
    if dex.journal.pending_claims():
        raise SystemExit(f"Claim the pending swaps in {directory / 'swaps.jsonl'} (dex.collect_all()) before another batch")

    ### STEP 3. Plan the record layout. ###
    ### One record per trade lets every swap start at once. Balanced lots with zero tolerance give
    ### each trade the same spendable amount. Planning reads; it reserves and signs nothing.
    token = dex.api.get_token(token_in)
    minimum = int(Decimal(amount_in).scaleb(token.decimals))
    plan = dex.plan_inventory(token_in, InventoryTarget(records=count, min_record_amount=minimum,
                                                         distribution="balanced", tolerance_bps=0),
                              max_transactions=24)
    print(f"{count} concurrent {token_in} -> {token_out} swaps of {amount_in} {token_in}; recovery data in {directory}")
    print(f"{len(plan.steps)} maintenance operations; credits deduction {plan.deduction} microcredits")
    for step in plan.steps:
        helper = step.join_program.program if step.join_program else "native"
        print(f"  {step.kind:5} inputs={len(step.inputs):2} outputs={len(step.outputs)} via {helper}")
    preview = dex.quote(token_in=token_in, token_out=token_out, amount_in=amount_in, slippage_bps=50)
    print(f"Per swap: estimated {preview.estimated_amount_out} {token_out}; minimum {preview.minimum_amount_out} {token_out}")
    if not args.execute:
        print("Preview only. Pass --execute to prepare the records, submit the swaps, and claim the outputs.")
        raise SystemExit(0)

    ### STEP 4. Execute the plan. ###
    ### Each step reserves its inputs, proves, broadcasts, waits for confirmation, and waits until the
    ### scanner shows the exact outputs before the next step spends them. A marker survives a crash so
    ### a later run does not repeat an unfinished batch.
    marker = directory / "batch.pending"
    if marker.exists():
        raise SystemExit(f"Resolve the previous batch before removing {marker}")
    marker.write_text("")
    result = aleo.inventory.rebalance_record_inventory(plan, timeout=600.0, poll_interval=3.0)
    if result.status != "complete":
        raise SystemExit(f"Maintenance interrupted: {result.reason}; keep {directory}")

    ### STEP 5. Submit the swaps together, then claim each output. ###
    ### Quote after maintenance so joins and splits do not consume its lifetime. One quote fixes the
    ### output floor for every trade; it reserves no pool liquidity, so a trade can still fail when the
    ### price moves. Each swap selects its own record; a swap whose record another swap took first moves
    ### to the next eligible record before proving. A swap and its claim are two transactions.
    quote = dex.quote(token_in=token_in, token_out=token_out, amount_in=amount_in, slippage_bps=50)

    def trade(index):
        """Swap, then claim. Each outcome is reported on its own so a failed sibling never hides a successful claim."""
        try:
            handle = dex.swap(quote).delegate(wait=False)
        except Exception as error:  # reported per trade, never raised mid-batch
            return index, None, None, error
        try:
            claim = dex.claim_swap_output(handle, timeout=600.0).delegate(wait=True)
            return index, handle, claim, None
        except Exception as error:  # the handle is kept for a later claim
            return index, handle, None, error

    with ThreadPoolExecutor(max_workers=count) as pool:
        outcomes = list(pool.map(trade, range(1, count + 1)))

    failures = 0
    for index, handle, claim, error in outcomes:
        if claim is not None and handle is not None:
            print(f"{index}: {handle.transaction_id} -> {claim.transaction_id}; "
                  f"received {claim.amount_out} base units; refunded {claim.amount_remaining}")
        elif handle is not None:
            failures += 1
            print(f"{index}: swap {handle.transaction_id} submitted; claim failed: {error}", file=sys.stderr)
        else:
            failures += 1
            print(f"{index}: swap failed: {error}", file=sys.stderr)
    if failures:
        raise SystemExit(f"{failures} trades need recovery in {directory}; do not repeat the batch")
    marker.unlink()
