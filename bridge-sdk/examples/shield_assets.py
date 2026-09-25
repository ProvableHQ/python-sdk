"""Move an existing public Aleo balance into a private record for private use.

Hyperlane delivers public balances, so shield after bridge delivery if private
funds are needed. xReserve can deliver private USDCx directly, avoiding this
extra conversion. Shielding pays a separate Aleo fee and does not conceal
an earlier public deposit. Preview the amount before adding --execute.
"""
from decimal import Decimal
import os
import sys
from aleo import Aleo, HTTPProvider
from aleo_bridge import Bridge, PollingTimeoutError

from _arguments import shield_parser


def main(argv=None):
    parser = shield_parser(__doc__)
    args = parser.parse_args(argv)
    # Choose only funds already available as a public Aleo balance. Keep display
    # amounts as decimal strings to avoid rounding money through a float.
    amount = Decimal(args.amount)
    if not amount.is_finite() or amount <= 0:
        parser.error('Amount must be finite and positive.')
    # Previewing does not request a signature or pay a fee.
    if not args.execute:
        print(f'Shield {args.amount} {args.asset} on Aleo mainnet. Use --execute to submit.')
        return 0
    if not os.environ.get("ALEO_PRIVATE_KEY"):
        parser.error("Set ALEO_PRIVATE_KEY before shielding.")
    # The Aleo account must hold the asset and enough credits for this separate transaction.
    aleo = Aleo(HTTPProvider(os.environ.get("ALEO_RPC_URL", "https://edge.provable.com/api"), network="mainnet"))
    aleo.default_account = aleo.account.from_private_key(os.environ["ALEO_PRIVATE_KEY"])
    bridge = Bridge(aleo)
    # Delegated proving avoids generating the proof on this machine. The service
    # receives transaction contents, while the account private key stays local.
    receipt = bridge.shield(args.asset, amount=args.amount).delegate()
    # Submission is not confirmation. Keep this ID and check acceptance before
    # spending the resulting record; a lost RPC response is not a reason to repeat shielding.
    print('Submitted Aleo transaction:', receipt.transaction_id, flush=True)
    print('Check confirmation before spending the resulting private record.')
    return 2  # Submitted, not yet confirmed. Shielding is a separate Aleo transaction, not a bridge journal entry.


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
