"""Compare the cost of sending WBTC from Ethereum to Aleo before committing funds.

The recipient receives a public WBTC balance on Aleo. Request a quote to review
Ethereum fees and the expected WBTC received; this example needs no signing key
and submits no transaction.
"""
import os
import sys
from aleo import Aleo, HTTPProvider
from aleo_bridge import Bridge, Ethereum, PollingTimeoutError

from _arguments import quote_parser


def main(argv=None):
    parser = quote_parser(__doc__)
    args = parser.parse_args(argv)
    # Check the cost for the intended accounts without giving either network a signing key.
    aleo = Aleo(HTTPProvider(os.environ.get("ALEO_RPC_URL", "https://edge.provable.com/api"), network="mainnet"))
    ethereum = Ethereum(os.environ.get("ETHEREUM_RPC_URL", "https://ethereum-rpc.publicnode.com"))
    bridge = Bridge(aleo, ethereum=ethereum)
    # Name both assets to make the expected receipt clear: WBTC on Ethereum becomes
    # WBTC on Aleo. The default amount, "0.001", is WBTC rather than atomic units.
    quote = bridge.quote(source_chain='ethereum', source_asset='wbtc',
                         destination_chain='aleo', destination_asset='wbtc',
                         amount=args.amount, recipient=args.recipient, sender=args.sender)
    print("Expected received:", quote.amount_out)  # WBTC arriving on Aleo, in display units.
    # Keep ETH for the quoted network fees; sending WBTC does not pay those fees in WBTC.
    for fee in quote.fees:
        print("Fee:", fee.kind, fee.amount, fee.asset_id, "(estimated)" if fee.estimated else "")
    # A successful quote does not move funds. Continue with bridge_wbtc.py to submit.
    return 0


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
