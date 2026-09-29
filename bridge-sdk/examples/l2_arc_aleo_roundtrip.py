"""Run one explicit public leg of Base/Arbitrum → Arc → Aleo → Arc → L2."""
import json
from pathlib import Path

from aleo_bridge.errors import BridgeError
from aleo_bridge.units import format_decimal_amount

try:
    from ._arc_workflow import build_bridge, entrypoint, parser, run_leg, spendable
except ImportError:
    from _arc_workflow import build_bridge, entrypoint, parser, run_leg, spendable


def main(argv=None):
    p = parser(__doc__)
    p.add_argument('--l2', choices=['base','arbitrum'], required=True)
    p.add_argument('--step', type=int, choices=[1,2,3,4], required=True,
                   help='1 L2→Arc; 2 Arc→Aleo; 3 Aleo→Arc; 4 Arc→L2. Each step needs its own invocation.')
    args = p.parse_args(argv)
    bridge = build_bridge((args.l2,'arc'), execute=args.execute, aleo_signer=args.step == 3)
    root = Path(args.journal).expanduser()/args.l2
    routes = [f'cctp:{args.l2}/usdc->arc/usdc','xreserve:arc/usdc->aleo/usdcx',
              'xreserve:aleo/usdcx->arc/usdc',f'cctp:arc/usdc->{args.l2}/usdc']
    amount = args.amount
    if args.step > 1:
        previous = json.loads((root/f'leg-{args.step-1}.json').read_text())
        if not previous.get('done') or previous.get('request',{}).get('route') != routes[args.step-2]:
            raise BridgeError('Previous leg is not complete; recover it before proceeding')
        received = previous['received_atomic']
        amount = spendable(received,args.arc_gas_reserve) if args.step in (2,4) else format_decimal_amount(received,6)
    run_leg(bridge,route=routes[args.step-1],amount=amount,
            recipient=args.recipient if args.step == 2 else args.sender,
            sender=args.recipient if args.step == 3 else args.sender,
            state_path=root/f'leg-{args.step}.json',execute=args.execute,timeout=args.timeout,
            manual_mint=args.manual_mint)
    return 0


if __name__ == '__main__':
    entrypoint(main)
