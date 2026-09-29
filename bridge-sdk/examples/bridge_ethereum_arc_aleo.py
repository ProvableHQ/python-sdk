"""Preview Ethereum → Arc; after verified delivery, send received USDC to Aleo."""
from pathlib import Path

try:
    from ._arc_workflow import build_bridge, entrypoint, parser, run_leg, spendable
except ImportError:
    from _arc_workflow import build_bridge, entrypoint, parser, run_leg, spendable


def main(argv=None):
    args = parser(__doc__).parse_args(argv)
    bridge = build_bridge(('ethereum','arc'), execute=args.execute)
    root = Path(args.journal).expanduser()
    received = run_leg(bridge, route='cctp:ethereum/usdc->arc/usdc', amount=args.amount,
                recipient=args.sender, sender=args.sender, state_path=root/'ethereum-arc.json',
                execute=args.execute, timeout=args.timeout, manual_mint=args.manual_mint)
    if received is None:
        print('Arc → Aleo will be quoted from the verified receipt, less the Arc gas reserve.')
        return 0
    run_leg(bridge, route='xreserve:arc/usdc->aleo/usdcx', amount=spendable(received,args.arc_gas_reserve),
            recipient=args.recipient, sender=args.sender, state_path=root/'arc-aleo.json',
            execute=args.execute, timeout=args.timeout)
    return 0


if __name__ == '__main__':
    entrypoint(main)
