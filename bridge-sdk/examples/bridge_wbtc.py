"""Bridge WBTC from Ethereum to a public balance on Aleo.

Use this transfer to make Ethereum-held WBTC available to Aleo applications.
The sender needs WBTC for the transfer and ETH for fees. Review the quote first,
then use --execute to submit and monitor delivery; the journal allows recovery
if the application closes.
"""
import os
import sys

from aleo import Aleo, HTTPProvider
from aleo_bridge import Bridge, Ethereum, FileCheckpointStore, PollingTimeoutError

from _arguments import transfer_parser


def main(argv=None):
    parser = transfer_parser(__doc__, amount="0.001")
    args = parser.parse_args(argv)
    if not args.sender and not args.execute:
        parser.error("Pass --sender for a read-only quote.")
    if args.execute and not os.environ.get("EVM_PRIVATE_KEY"):
        parser.error("Set EVM_PRIVATE_KEY before submitting.")

    # Connect the networks that report source confirmation and delivery on Aleo.
    # Only --execute loads a sender key; requesting a quote needs no signature.
    aleo = Aleo(HTTPProvider(os.environ.get("ALEO_RPC_URL", "https://edge.provable.com/api"), network="mainnet"))
    ethereum = Ethereum(
        os.environ.get("ETHEREUM_RPC_URL", "https://ethereum-rpc.publicnode.com"),
        private_key=os.environ["EVM_PRIVATE_KEY"] if args.execute else None,
    )
    # A journal retains the submitted work so an interruption does not require a new deposit.
    store = FileCheckpointStore(args.journal) if args.execute else None
    bridge = Bridge(aleo, ethereum=ethereum, checkpoints=store)

    # Review the expected amount received and fees before committing funds. Keep enough ETH
    # for Ethereum fees as well as the asset being bridged.
    quote = bridge.quote(
        source_chain="ethereum", source_asset="wbtc",
        destination_chain="aleo", destination_asset="wbtc",
        amount=args.amount,  # Display units: "0.001" means 0.001 WBTC.
        recipient=args.recipient,
        sender=args.sender or ethereum.address, mint_mode="public",  # A publicly visible Aleo balance.
    )
    print("Expected received:", quote.amount_out)  # WBTC on Aleo, in display units.
    # Estimated fees can change before submission; each entry names the asset needed.
    for fee in quote.fees:
        print("Fee:", fee.kind, fee.amount, fee.asset_id, "(estimated)" if fee.estimated else "")
    if not args.execute:
        print("Quote only. Use --execute once after reviewing the fees.")
        return 0

    # Submit the reviewed plan once. This commits funds and may first approve token spending.
    # Keep the journal ID printed below; it stays fixed through delivery and recovery.
    print("Journal:", args.journal, flush=True)
    progress = bridge.execute(
        quote.plan,
        on_checkpoint=lambda checkpoint: print("Journal ID:", checkpoint.id, flush=True),
    )
    # The source deposit can confirm before the funds arrive on Aleo. Waiting only
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
    elif progress.next == "complete":
        print("Recover with --action complete and the original nonce to claim private USDCx.")
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
              "For shielding, check Aleo transaction history.", file=sys.stderr)
        raise SystemExit(1)
