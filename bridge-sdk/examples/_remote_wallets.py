"""Shared steps of the server-wallet tutorials; each provider example builds its own signer."""
import os
import sys

from aleo_bridge import PollingTimeoutError

DEFAULT_AMOUNTS = {"ethereum": "0.00001", "solana": "0.0001"}
ASSETS = {"ethereum": "eth", "solana": "sol"}
DEFAULT_RPC_URLS = {"ethereum": "https://ethereum-rpc.publicnode.com", "solana": "https://api.mainnet-beta.solana.com"}
RPC_VARIABLES = {"ethereum": "ETHEREUM_RPC_URL", "solana": "SOLANA_RPC_URL"}


def required(name, parser):
    """The environment variable *name*, or a usage error naming only the variable."""
    value = os.environ.get(name, "").strip()
    if not value:
        parser.error(f"Set {name} before running this example.")
    return value


def rpc_url(chain):
    """The public mainnet RPC for *chain* unless its variable names another endpoint."""
    return os.environ.get(RPC_VARIABLES[chain], "").strip() or DEFAULT_RPC_URLS[chain]


def bridge_transfer(args, bridge, *, sender):
    """Quote the transfer for *sender*; with ``--execute`` submit the reviewed plan once and monitor it."""
    asset = ASSETS[args.chain]
    # Review the amount arriving on Aleo and every source-chain fee before committing funds.
    quote = bridge.quote(
        source_chain=args.chain, source_asset=asset,
        destination_chain="aleo", destination_asset=asset,
        amount=args.amount or DEFAULT_AMOUNTS[args.chain],  # Display units: "0.00001" means 0.00001 ETH.
        recipient=args.recipient,
        sender=sender,  # The server wallet pays the fees and signs the deposit.
    )
    print("Sender:", sender)
    print("Expected received:", quote.amount_out)  # Public balance on Aleo, in display units.
    # Estimated fees can change before submission; each entry names the asset needed.
    for fee in quote.fees:
        print("Fee:", fee.kind, fee.amount, fee.asset_id, "(estimated)" if fee.estimated else "")
    if not args.execute:
        print("Quote only. Use --execute once after reviewing the fees.")
        return 0

    # Submit the reviewed plan once. The provider signs remotely; the journal keeps the receipt.
    print("Journal:", args.journal, flush=True)
    progress = bridge.execute(
        quote.plan,
        on_checkpoint=lambda checkpoint: print("Journal ID:", checkpoint.id, flush=True),
    )
    # Waiting only checks the existing deposit; it does not request another signature.
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
    return 2  # Pending or awaiting another action, not a reason to deposit again.


def run(main):
    """Run *main* with the same exit codes and credential-safe error reporting as the other tutorials."""
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
        # Provider and RPC exceptions can contain credentials or request bodies.
        print(f"{type(exc).__name__}: check inputs, balances, provider credentials and RPC connectivity. "
              "If submission began, recover the journal before retrying. "
              "Check the source chain's transaction history if no journal entry was saved.", file=sys.stderr)
        raise SystemExit(1)
