"""Command-line options for the bridge tutorials; SDK operations stay in each example."""
import argparse


def transfer_parser(description, *, amount):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--recipient", required=True, help="Aleo address receiving the funds.")
    parser.add_argument("--sender", help="Checksummed Ethereum address for a read-only quote.")
    parser.add_argument("--amount", default=amount, help="Amount in display units.")
    parser.add_argument("--execute", action="store_true", help="Submit a NEW transfer on MAINNET; costs real funds.")
    parser.add_argument("--journal", default="~/.aleo-bridge/checkpoints")
    parser.add_argument("--timeout", type=float, default=120, help="Seconds to monitor delivery.")
    return parser


def quote_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--sender', required=True, help='Checksummed Ethereum sender address.')
    parser.add_argument('--recipient', required=True, help='Aleo recipient address.')
    parser.add_argument('--amount', default='0.001', help='WBTC in display units.')
    return parser


def journal_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--journal', default="~/.aleo-bridge/checkpoints")
    parser.add_argument('--id', help='Filename without .json; omit to list saved transfers locally.')
    parser.add_argument('--action', choices=['status', 'wait', 'resume', 'complete'], default='status',
                        help='Status/wait read only; resume/complete can spend funds.')
    parser.add_argument('--timeout', type=float, default=120)
    return parser


def history_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--transaction', required=True, help='Confirmed bridge dispatch hash, not token approval.')
    parser.add_argument('--sender', required=True, help='Original checksummed Ethereum sender.')
    parser.add_argument('--recipient', required=True, help='Original Aleo recipient.')
    parser.add_argument('--amount', required=True, help='Original amount in WBTC, e.g. 0.001.')
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--timeout', type=float, default=120)
    return parser


def shield_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--asset', default='aleo/sol', help='Aleo asset ID, e.g. aleo/sol or aleo/usdcx.')
    parser.add_argument('--amount', default='0.01', help='Display units to shield.')
    parser.add_argument('--execute', action='store_true', help='Submit a MAINNET Aleo transaction with a fee.')
    return parser


def outbound_parser(description, *, amount):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--recipient', required=True, help='Destination Ethereum or Solana public address.')
    parser.add_argument('--sender', help='Aleo public address for a read-only quote.')
    parser.add_argument('--amount', default=amount, help='Source asset in display units.')
    parser.add_argument('--execute', action='store_true', help='Submit a NEW MAINNET transfer; costs real funds.')
    parser.add_argument('--journal', default='~/.aleo-bridge/checkpoints')
    parser.add_argument('--timeout', type=float, default=120, help='Seconds to monitor delivery.')
    return parser
