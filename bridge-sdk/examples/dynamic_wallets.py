"""Bridge ETH or SOL from an existing Dynamic server wallet to a public balance on Aleo.

The Dynamic MPC wallet signs the deposit remotely, so a backend can bridge
without holding a chain private key. The wallet needs the asset plus fees on
the source chain. Review the quote first, then use --execute to submit and
monitor delivery; the journal allows recovery if the application closes.
"""
import os

from aleo import Aleo, HTTPProvider
from aleo_bridge import Bridge, Ethereum, FileCheckpointStore, Solana
from aleo_bridge.dynamic import DynamicEvmSigner, DynamicSolanaSigner
from dynamic_wallet_sdk import DynamicEvmWalletClient, DynamicSvmWalletClient

if __package__:
    from ._arguments import remote_wallet_parser
    from ._remote_wallets import bridge_transfer, required, rpc_url, run
else:
    from _arguments import remote_wallet_parser
    from _remote_wallets import bridge_transfer, required, rpc_url, run


def main(argv=None):
    parser = remote_wallet_parser(__doc__)
    args = parser.parse_args(argv)

    # The API token authenticates the backend; the wallet password unlocks shares backed up to Dynamic.
    environment_id = required("DYNAMIC_ENVIRONMENT_ID", parser)
    api_token = required("DYNAMIC_API_TOKEN", parser)

    # Only the Dynamic wallet signs. Aleo is used to monitor public delivery.
    aleo = Aleo(HTTPProvider(os.environ.get("ALEO_RPC_URL", "https://edge.provable.com/api"), network="mainnet"))
    # A journal retains the submitted work so an interruption does not require a new deposit.
    store = FileCheckpointStore(args.journal) if args.execute else None
    if args.chain == "ethereum":
        signer = DynamicEvmSigner(
            DynamicEvmWalletClient(environment_id),
            address=required("DYNAMIC_EVM_ADDRESS", parser),
            api_token=api_token,
            password=required("DYNAMIC_EVM_WALLET_PASSWORD", parser),
            wallet_id=os.environ.get("DYNAMIC_EVM_WALLET_ID", "").strip() or None,
        )
        bridge = Bridge(aleo, ethereum=Ethereum(rpc_url("ethereum"), signer=signer), checkpoints=store)
    else:
        signer = DynamicSolanaSigner(
            DynamicSvmWalletClient(environment_id),
            address=required("DYNAMIC_SOLANA_ADDRESS", parser),
            api_token=api_token,
            password=required("DYNAMIC_SOLANA_WALLET_PASSWORD", parser),
            wallet_id=os.environ.get("DYNAMIC_SOLANA_WALLET_ID", "").strip() or None,
        )
        bridge = Bridge(aleo, solana=Solana(rpc_url("solana"), signer=signer), checkpoints=store)

    # Authenticate and resolve the wallet by address now, so a wrong token or address fails before any quote.
    print("Dynamic wallet:", signer.resolve(), signer.address)
    try:
        return bridge_transfer(args, bridge, sender=signer.address)
    finally:
        signer.close()


if __name__ == "__main__":
    run(main)
