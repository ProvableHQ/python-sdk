"""Create and fund a testnet account, swap 1.5 USDCx for ETH, and claim the output."""
import os

from aleo import Aleo, HTTPProvider, testnet

from aleo_shield_swap import Journal, Profile, ShieldSwap


ENABLE_JOURNAL = False


if __name__ == "__main__":
    ### STEP 1. Create an Aleo Account ###
    ### Shield Swap runs on the Aleo blockchain to enable private trading of asset pairs.
    ### This step configures an Aleo account to enable private trading.

    # Use a configured private key or create a new key.
    key = os.environ.get("SHIELD_SWAP_PRIVATE_KEY")
    if not key:
        # Load a saved private key from an existing Profile or create a new one.
        profile = Profile.load_or_create(network="testnet")
        if profile.network != "testnet":
            raise RuntimeError("This example requires a testnet profile")
        key = profile.private_key

    # Create an Aleo object capable of talking to the Aleo network and configure an account.
    aleo = Aleo(HTTPProvider("https://edge.provable.com/api", network="testnet"))
    private_key = testnet.PrivateKey.from_string(key)
    account = aleo.account.from_private_key(private_key)
    address = str(account.address)

    # Register the account with the record scanning service to find its records.
    # The SDK decrypts the returned records locally.
    registration = aleo.records.register(account)
    if not registration["ok"]:
        raise RuntimeError(f"Record scanner registration failed: {registration['error']}")

    ### STEP 2. Create a Shield Swap client and fund the account with testnet tokens. ###
    # Use this account and scanner for Shield Swap operations and configure a local journal
    # to keep track of trading activity.
    shield_swap_client = ShieldSwap(aleo)
    if ENABLE_JOURNAL:
        shield_swap_client.journal = Journal(f"testnet-{address}.jsonl")
    # Authenticate with the shield swap API.
    shield_swap_client.api.authenticate(address, lambda message: str(private_key.sign(message.encode())))
    # Use existing private USDCx when one record can cover the swap.
    source = shield_swap_client.api.get_token("USDCx")
    amount_in = "1.5"
    if not shield_swap_client.has_swap_balance(source.address, amount_in):
        # Request testnet tokens and stop if funding fails.
        funding = shield_swap_client.confirm_airdrop()
        if not funding.success:
            raise RuntimeError(funding.error)

    ### STEP 3. Execute a swap between USDCx and ETH. ###
    # Quote the best available route, including intermediate tokens when useful.
    # The quote contains the final output estimate and a 0.5% slippage limit.
    quote = shield_swap_client.quote(
        token_in="USDCx", token_out="ETH", amount_in=amount_in, slippage_bps=50,
    )

    # Execute the quoted route in one swap transaction and wait for confirmation.
    handle = shield_swap_client.swap(quote).delegate(wait=True)

    # Wait for the confirmed swap's output to become readable, then submit one claim.
    claim = shield_swap_client.claim_swap_output(
        handle, timeout=5,
    ).delegate(wait=True)
    if claim.amount_out <= 0:
        raise RuntimeError("The claim returned no ETH")
