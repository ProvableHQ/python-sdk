"""Continue an interrupted bridge transfer without depositing again.

A journal stores checkpoints identifying the transfer and submitted work.
List these saved entries, select the intended transfer, then check whether
funds arrived. Status reads send no funds; request resume or complete only
when the existing transfer needs that action.
"""
import os
import sys
from aleo import Aleo, HTTPProvider, mainnet
from aleo_bridge import Bridge, Ethereum, FileCheckpointStore, PollingTimeoutError, Solana

from _arguments import journal_parser


def main(argv=None):
    parser = journal_parser(__doc__)
    args = parser.parse_args(argv)
    # Start with saved entries to identify the transfer by pair, amount, and recipient.
    # Listing only reads local files; an unreadable entry does not prove its transfer failed.
    store = FileCheckpointStore(args.journal)
    if not args.id:
        result = store.load_checkpoints()
        for cp in result.checkpoints:
            print(cp.id, cp.intent['source'], cp.intent['amount'], cp.intent['recipient'])
        for problem in result.errors:
            print('Unreadable checkpoint:', problem.path, problem.error_type)
        return 1 if result.errors else 0
    # Use the filename without .json. A bundled, historical example is
    # 2026-09-25_001_solana-sol_to_aleo-sol_676.2; the date is journal creation, not submission.
    cp = store.load(args.id)
    if cp is None:
        parser.error('No checkpoint with this ID. List the journal to find its current name.')
    # Connect the original networks. Only a requested submission needs a signing key.
    source = cp.intent['source']['chain']
    aleo = Aleo(HTTPProvider(os.environ.get("ALEO_RPC_URL", "https://edge.provable.com/api"), network="mainnet"))
    if args.action == "complete":
        if not os.environ.get("ALEO_PRIVATE_KEY"):
            parser.error("Set ALEO_PRIVATE_KEY to claim private USDCx.")
        aleo.default_account = aleo.account.from_private_key(os.environ["ALEO_PRIVATE_KEY"])
    evm_key = None
    if args.action == "resume" and source == "ethereum":
        evm_key = os.environ.get("EVM_PRIVATE_KEY")
        if not evm_key:
            parser.error("Set EVM_PRIVATE_KEY to resume the Ethereum source submission.")
    ethereum = Ethereum(os.environ.get("ETHEREUM_RPC_URL", "https://ethereum-rpc.publicnode.com"),
                        private_key=evm_key)
    solana = None
    if source == "solana" or cp.intent["destination"]["chain"] == "solana":
        solana = Solana(os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"))
    # Status reads leave the original checkpoint intact, including the bundled example.
    bridge = Bridge(aleo, ethereum=ethereum, solana=solana,
                    checkpoints=store if args.action in ("resume", "complete") else None)
    # Check what already completed before deciding whether another transaction is needed.
    # Recovery reads the existing transfer; it never sends a replacement deposit.
    progress = bridge.recover(cp)
    if args.action in ("resume", "complete") and progress.next == args.action:
        # Private USDCx still needs the original claim secret; the journal cannot replace it.
        nonce = None
        if cp.intent.get("mintMode") == "private":
            nonce = os.environ.get("BRIDGE_MINT_SECRET_NONCE")
            if not nonce:
                parser.error("Set BRIDGE_MINT_SECRET_NONCE to the ORIGINAL saved nonce.")
            if str(mainnet.Scalar.from_string(nonce)) == "0scalar":
                parser.error("The private claim nonce must be nonzero.")
        # Only perform the step the transfer currently requests and the caller selected.
        if args.action == "resume":
            # Finish missing source work without repeating confirmed approvals or deposits.
            progress = bridge.resume(
                progress, secret_nonce=nonce,
                on_checkpoint=lambda checkpoint: print("Journal ID:", checkpoint.id, flush=True),
            )
        elif args.action == "complete":
            # Claim private USDCx with the recipient account and original nonce; pays an Aleo fee.
            progress = bridge.complete(
                progress, secret_nonce=nonce,
                on_checkpoint=lambda checkpoint: print("Journal ID:", checkpoint.id, flush=True),
            )
    # After submission, wait for confirmation or delivery rather than submitting again.
    if args.action != "status" and progress.next == "wait":
        progress = bridge.wait(progress, timeout_seconds=args.timeout)
    print("Next:", progress.next)  # "done" confirms delivery; "wait" means keep monitoring.
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
