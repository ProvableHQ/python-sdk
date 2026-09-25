"""Recover monitoring of a WBTC transfer when no journal or other files remain.

For a confirmed Ethereum-to-Aleo Hyperlane dispatch, retrieve the original
transaction hash, sender, recipient, and WBTC amount from chain history.
These details restore monitoring without signing or repeating the deposit.
This flow does not recover private USDCx claim secrets or other bridge routes.
"""
import re
import os
import sys
from aleo import Aleo, HTTPProvider
from aleo_bridge import Bridge, Ethereum, PollingTimeoutError

from _arguments import history_parser


def main(argv=None):
    parser = history_parser(__doc__)
    args = parser.parse_args(argv)
    # Locate the confirmed bridge dispatch in Ethereum history. A token approval
    # only permits spending; it is not the transaction that moved funds into the bridge.
    if not re.fullmatch(r'0x[0-9a-fA-F]{64}', args.transaction):
        parser.error('Expected an Ethereum transaction hash: 0x and 64 hexadecimal digits.')
    # No journal, keys, or other persisted files are required.
    aleo = Aleo(HTTPProvider(os.environ.get("ALEO_RPC_URL", "https://edge.provable.com/api"), network="mainnet"))
    ethereum = Ethereum(os.environ.get("ETHEREUM_RPC_URL", "https://ethereum-rpc.publicnode.com"))
    bridge = Bridge(aleo, ethereum=ethereum)
    route = bridge.routes(source_chain='ethereum', source_asset='wbtc',
                          destination_chain='aleo', destination_asset='wbtc', bridge_protocol='hyperlane')[0]
    # Reconstruct the transfer details from the explorer in memory. No checkpoint,
    # journal, or other saved file is loaded or written by this example.
    recovery = {
        'version': 1,
        'intent': {'source': {'chain': 'ethereum', 'asset': 'wbtc'},
                   'destination': {'chain': 'aleo', 'asset': 'wbtc'},
                   'bridgeProtocol': 'hyperlane', 'amount': args.amount,  # e.g. "0.001" WBTC.
                   'sender': args.sender, 'recipient': args.recipient},
        'route': {'id': route.id, 'registryVersion': bridge.registry.version},
        'source': {'transactionId': args.transaction},
    }
    # Ask the network about the original deposit. Recovery does not authorize a new one.
    progress = bridge.recover(recovery)
    # Add --wait to keep checking delivery; leaving this process does not cancel the transfer.
    if args.wait and progress.next == "wait":
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
