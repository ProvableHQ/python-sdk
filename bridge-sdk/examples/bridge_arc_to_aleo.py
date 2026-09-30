"""Preview or resume a public Arc USDC → Aleo USDCx transfer."""
from pathlib import Path

try:
    from ._arc_workflow import build_bridge, entrypoint, parser, run_leg
except ImportError:
    from _arc_workflow import build_bridge, entrypoint, parser, run_leg


def main(argv=None):
    args = parser(__doc__).parse_args(argv)
    bridge = build_bridge(('arc',), execute=args.execute)
    run_leg(bridge, route='xreserve:arc/usdc->aleo/usdcx', amount=args.amount,
            recipient=args.recipient, sender=args.sender, state_path=Path(args.journal).expanduser()/'arc-aleo.json',
            execute=args.execute, timeout=args.timeout)
    return 0


if __name__ == '__main__':
    entrypoint(main)
