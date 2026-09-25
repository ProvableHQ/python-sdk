"""Return SOL held on Aleo to a Solana address.

Use this transfer to receive native SOL for Solana applications and network fees.
Spend a public Aleo SOL balance; unshield private SOL first. Keep Aleo credits
for transaction and Hyperlane delivery fees. Receiving SOL needs no Solana key.
"""
import os
import sys

from aleo import Aleo, HTTPProvider
from aleo_bridge import Bridge, Solana, FileCheckpointStore, PollingTimeoutError

from _arguments import outbound_parser


def main(argv=None):
    parser = outbound_parser(__doc__, amount="0.01")
    args = parser.parse_args(argv)
    if not args.sender and not args.execute:
        parser.error("Pass --sender for a read-only quote.")
    if args.execute and not os.environ.get("ALEO_PRIVATE_KEY"):
        parser.error("Set ALEO_PRIVATE_KEY before submitting.")

    # Only Aleo signs the withdrawal. The destination connection reads public delivery status.
    aleo = Aleo(HTTPProvider(os.environ.get("ALEO_RPC_URL", "https://edge.provable.com/api"), network="mainnet"))
    if args.execute:
        aleo.default_account = aleo.account.from_private_key(os.environ["ALEO_PRIVATE_KEY"])
    solana = Solana(os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"))
    # A journal retains the submitted work so an interruption does not require a new deposit.
    store = FileCheckpointStore(args.journal) if args.execute else None
    bridge = Bridge(aleo, solana=solana, checkpoints=store)

    # Review the amount arriving on Solana and the assets needed for fees.
    # An estimated fee can change before the transfer is processed.
    quote = bridge.quote(
        source_chain="aleo", source_asset="sol",
        destination_chain="solana", destination_asset="sol",
        amount=args.amount,  # Display units of SOL, not atomic units.
        recipient=args.recipient,
        sender=bridge.aleo_address() if args.execute else args.sender,
    )
    print("Expected received:", quote.amount_out)  # SOL on Solana, in display units.
    # Estimated fees can change before submission; each entry names the asset needed.
    for fee in quote.fees:
        print("Fee:", fee.kind, fee.amount, fee.asset_id, "(estimated)" if fee.estimated else "")
    if not args.execute:
        print("Quote only. Use --execute once after reviewing the fees.")
        return 0

    # Submit once from the public Aleo balance. Delegated proving is the default;
    # add proving="local" to keep transaction contents out of the proving service.
    # Keep the journal ID printed below; it stays fixed through delivery and recovery.
    print("Journal:", args.journal, flush=True)
    progress = bridge.execute(
        quote.plan,
        mode="signer",  # Spend the public balance, without scanning private records.
        on_checkpoint=lambda checkpoint: print("Journal ID:", checkpoint.id, flush=True),
    )
    # The Aleo transaction can confirm before funds arrive on the destination. Waiting only
    # checks progress; it does not sign or send another transaction.
    if progress.next == "wait":
        progress = bridge.wait(progress, timeout_seconds=args.timeout)
    # Only "done" confirms delivery. A pending result or timeout is not a failed deposit.
    print("Next:", progress.next)
    print("Receipt:", progress.receipt.id)
    print("Source transaction:", progress.receipt.source_tx_id)
    if progress.next == "done":
        return 0
    if progress.next == "failed":
        print("Transfer failed. Check transaction history before taking another action.", file=sys.stderr)
        return 1
    if progress.next == "resume":
        print("Recover this journal entry with --action resume to finish the source submission.")
    return 2  # Pending or awaiting another action, not a reason to deposit again.


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PollingTimeoutError as exc:
        if exc.progress is not None:
            print("Last known step:", exc.progress.next)
            print("Source transaction:", exc.progress.receipt.source_tx_id)
        print("Monitoring timed out. Recover the existing transfer; do not submit it again.", file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        print("Interrupted. Recover the journal and check transaction history before submitting again.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        # Raw RPC exceptions can contain credentials or request bodies.
        print(f"{type(exc).__name__}: check inputs, balances and RPC connectivity. "
              "If submission began, recover the journal before retrying. "
              "Check Aleo transaction history if no journal entry was saved.", file=sys.stderr)
        raise SystemExit(1)
