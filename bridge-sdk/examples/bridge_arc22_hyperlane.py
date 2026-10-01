"""Bridge BAT, USDG, or ZEC between Aleo and Ethereum or Solana over Hyperlane.

Pick one of the ten directed routes with --route. A quote needs only public
addresses. Execution spends a public balance on the source chain: unshield
private Aleo records first, and hold SOL or ETH for the fees of a Solana or
Ethereum source. Delivery arrives as a public balance on the destination.
"""
import os
import sys

from aleo import Aleo, HTTPProvider
from aleo_bridge import Bridge, Ethereum, FileCheckpointStore, PollingTimeoutError, Solana

if __package__:
    from ._arguments import route_parser
else:
    from _arguments import route_parser

ROUTES = (
    "hyperlane:ethereum/bat->aleo/bat", "hyperlane:aleo/bat->ethereum/bat",
    "hyperlane:ethereum/usdg->aleo/usdg", "hyperlane:aleo/usdg->ethereum/usdg",
    "hyperlane:solana/bat->aleo/bat", "hyperlane:aleo/bat->solana/bat",
    "hyperlane:solana/usdg->aleo/usdg", "hyperlane:aleo/usdg->solana/usdg",
    "hyperlane:solana/zec->aleo/zec", "hyperlane:aleo/zec->solana/zec",
)
SOURCE_KEYS = {"aleo": "ALEO_PRIVATE_KEY", "ethereum": "EVM_PRIVATE_KEY", "solana": "SOLANA_PRIVATE_KEY"}


def main(argv=None):
    parser = route_parser(__doc__, routes=ROUTES, amount="0.0001")
    args = parser.parse_args(argv)
    source_chain = args.route.split(":")[1].split("/")[0]
    if not args.sender and not args.execute:
        parser.error("Pass --sender for a read-only quote.")
    if args.execute and not os.environ.get(SOURCE_KEYS[source_chain]):
        parser.error(f"Set {SOURCE_KEYS[source_chain]} before submitting.")

    # Only the source chain signs. The other connection reads public delivery status.
    aleo = Aleo(HTTPProvider(os.environ.get("ALEO_RPC_URL", "https://edge.provable.com/api"), network="mainnet"))
    if args.execute and source_chain == "aleo":
        aleo.default_account = aleo.account.from_private_key(os.environ["ALEO_PRIVATE_KEY"])
    connections = {}
    if "ethereum" in args.route:
        connections["ethereum"] = Ethereum(
            os.environ.get("ETHEREUM_RPC_URL", "https://ethereum-rpc.publicnode.com"),
            private_key=os.environ["EVM_PRIVATE_KEY"] if args.execute and source_chain == "ethereum" else None,
        )
    if "solana" in args.route:
        connections["solana"] = Solana(
            os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"),
            private_key=os.environ["SOLANA_PRIVATE_KEY"] if args.execute and source_chain == "solana" else None,
        )
    # A journal retains the submitted work so an interruption does not require a second transfer.
    store = FileCheckpointStore(args.journal) if args.execute else None
    bridge = Bridge(aleo, checkpoints=store, **connections)

    # Review the amount arriving and every fee before committing funds. BAT has 18 decimals on
    # Aleo and Ethereum but 8 on Solana: amounts on a Solana route must fit 8 decimals.
    sender = args.sender
    if args.execute:
        sender = bridge.aleo_address() if source_chain == "aleo" else connections[source_chain].address
    quote = bridge.quote(route=args.route, amount=args.amount, recipient=args.recipient, sender=sender)
    print("Expected received:", quote.amount_out)  # Display units on the destination chain.
    # Estimated fees can change before submission; each entry names the asset needed.
    for fee in quote.fees:
        print("Fee:", fee.kind, fee.amount, fee.asset_id, "(estimated)" if fee.estimated else "")
    if not args.execute:
        print("Quote only. Use --execute once after reviewing the fees.")
        return 0

    # Submit the reviewed plan once. Keep the printed journal ID for recovery.
    print("Journal:", args.journal, flush=True)
    progress = bridge.execute(
        quote.plan,
        mode="signer" if source_chain == "aleo" else None,  # Aleo: spend the public balance directly.
        on_checkpoint=lambda checkpoint: print("Journal ID:", checkpoint.id, flush=True),
    )
    # Waiting only checks the existing dispatch; it does not submit another transfer.
    if progress.next == "wait":
        progress = bridge.wait(progress, timeout_seconds=args.timeout)
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
    return 2  # Pending or awaiting another action, not a reason to send again.


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
              "Check the source chain's transaction history if no journal entry was saved.", file=sys.stderr)
        raise SystemExit(1)
