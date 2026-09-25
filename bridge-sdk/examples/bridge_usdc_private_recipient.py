"""Bridge USDC privately to hide both the Aleo balance and recipient address.

Choose Private Bridge when the Ethereum deposit should conceal the Aleo
recipient. The Ethereum sender and amount remain public. Unlike automatic
private-record delivery, this option requires the recipient to claim USDCx
with the original secret nonce and Aleo account. Save that nonce before sending.
"""
import os
import sys

from aleo import Aleo, HTTPProvider, mainnet
from aleo_bridge import Bridge, Ethereum, FileCheckpointStore, PollingTimeoutError

from _arguments import transfer_parser


def main(argv=None):
    parser = transfer_parser(__doc__, amount="2")
    args = parser.parse_args(argv)
    if not args.sender and not args.execute:
        parser.error("Pass --sender for a read-only quote.")
    if args.execute and not os.environ.get("EVM_PRIVATE_KEY"):
        parser.error("Set EVM_PRIVATE_KEY before submitting.")

    # Connect the networks that report source confirmation and delivery on Aleo.
    # Only --execute loads a sender key; requesting a quote needs no signature.
    aleo = Aleo(HTTPProvider(os.environ.get("ALEO_RPC_URL", "https://edge.provable.com/api"), network="mainnet"))
    if args.execute:
        if not os.environ.get("ALEO_PRIVATE_KEY"):
            parser.error("Set ALEO_PRIVATE_KEY for the recipient's private USDCx claim.")
        aleo.default_account = aleo.account.from_private_key(os.environ["ALEO_PRIVATE_KEY"])
    ethereum = Ethereum(
        os.environ.get("ETHEREUM_RPC_URL", "https://ethereum-rpc.publicnode.com"),
        private_key=os.environ["EVM_PRIVATE_KEY"] if args.execute else None,
    )
    # A journal retains the submitted work so an interruption does not require a new deposit.
    store = FileCheckpointStore(args.journal) if args.execute else None
    bridge = Bridge(aleo, ethereum=ethereum, checkpoints=store)
    # The nonce conceals the Aleo address in the Ethereum deposit and unlocks the later claim.
    # Retain the securely generated value separately BEFORE depositing; the journal omits it.
    # Losing it prevents completion of the private claim through this flow.
    nonce = os.environ.get("BRIDGE_MINT_SECRET_NONCE")
    if not nonce:
        parser.error("Set BRIDGE_MINT_SECRET_NONCE to the securely generated and saved Aleo scalar.")
    if str(mainnet.Scalar.from_string(nonce)) == "0scalar":
        parser.error("The nonce must be a securely generated, nonzero scalar.")
    if args.execute and args.recipient != bridge.aleo_address():
        parser.error("ALEO_PRIVATE_KEY must belong to --recipient for this example's claim.")

    # Review the expected amount received and fees before committing funds. Keep enough ETH
    # for Ethereum fees as well as the asset being bridged.
    quote = bridge.quote(
        source_chain="ethereum", source_asset="usdc",
        destination_chain="aleo", destination_asset="usdcx",
        amount=args.amount,  # Display units: "2" means 2 USDC.
        recipient=args.recipient,
        sender=args.sender or ethereum.address,
        mint_mode="private",  # Conceal the Aleo recipient; return to claim after attestation.
        secret_nonce=nonce,
    )
    print("Expected received:", quote.amount_out)  # USDCx on Aleo, in display units.
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
        secret_nonce=nonce,
        on_checkpoint=lambda checkpoint: print("Journal ID:", checkpoint.id, flush=True),
    )
    # The source deposit can confirm before the funds arrive on Aleo. Waiting only
    # checks progress; it does not sign or send another transaction.
    if progress.next == "wait":
        progress = bridge.wait(progress, timeout_seconds=args.timeout)
    # "complete" means the deposit is ready to claim. The recipient signs an Aleo
    # transaction using the same nonce; retain enough Aleo credits for its fee.
    if progress.next == "complete":
        progress = bridge.complete(
            progress, secret_nonce=nonce,
            on_checkpoint=lambda checkpoint: print("Journal ID:", checkpoint.id, flush=True),
        )
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
