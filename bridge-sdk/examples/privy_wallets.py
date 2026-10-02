"""Bridge ETH or SOL from an existing Privy server wallet to a public balance on Aleo.

The Privy wallet signs the deposit remotely, so a backend can bridge without
holding a chain private key. The wallet needs the asset plus fees on the source
chain. Review the quote first, then use --execute to submit and monitor
delivery; the journal allows recovery if the application closes.
"""
import os

from aleo import Aleo, HTTPProvider
from aleo_bridge import Bridge, Ethereum, FileCheckpointStore, Solana
from aleo_bridge.privy import PrivyEvmSigner, PrivySolanaSigner
from privy import PrivyClient

if __package__:
    from ._arguments import remote_wallet_parser
    from ._remote_wallets import bridge_transfer, required, rpc_url, run
else:
    from _arguments import remote_wallet_parser
    from _remote_wallets import bridge_transfer, required, rpc_url, run


def main(argv=None):
    parser = remote_wallet_parser(__doc__)
    args = parser.parse_args(argv)

    # The app secret authenticates the backend; wallet owner keys are optional and per policy.
    privy = PrivyClient(app_id=required("PRIVY_APP_ID", parser), app_secret=required("PRIVY_APP_SECRET", parser))
    owner_key = os.environ.get("PRIVY_AUTHORIZATION_PRIVATE_KEY", "").strip()
    authorization_private_keys = [owner_key] if owner_key else None

    # Only the Privy wallet signs. Aleo is used to monitor public delivery.
    aleo = Aleo(HTTPProvider(os.environ.get("ALEO_RPC_URL", "https://edge.provable.com/api"), network="mainnet"))
    # A journal retains the submitted work so an interruption does not require a new deposit.
    store = FileCheckpointStore(args.journal) if args.execute else None
    if args.chain == "ethereum":
        signer = PrivyEvmSigner(
            privy,
            wallet_id=required("PRIVY_EVM_WALLET_ID", parser),
            address=required("PRIVY_EVM_ADDRESS", parser),
            authorization_private_keys=authorization_private_keys,
        )
        bridge = Bridge(aleo, ethereum=Ethereum(rpc_url("ethereum"), signer=signer), checkpoints=store)
    else:
        signer = PrivySolanaSigner(
            privy,
            wallet_id=required("PRIVY_SOLANA_WALLET_ID", parser),
            address=required("PRIVY_SOLANA_ADDRESS", parser),
            authorization_private_keys=authorization_private_keys,
        )
        bridge = Bridge(aleo, solana=Solana(rpc_url("solana"), signer=signer), checkpoints=store)

    # Confirm the wallet id and address name one Privy wallet before quoting or signing anything.
    print("Privy wallet:", signer.wallet_id, signer.resolve())
    return bridge_transfer(args, bridge, sender=signer.address)


if __name__ == "__main__":
    run(main)
