"""Create and fund a testnet account, swap 1.5 USDCx for ETH, and claim the output."""
import os

from aleo import Aleo, HTTPProvider, testnet

from aleo_shield_swap import Journal, ShieldSwap


ENABLE_JOURNAL = False


if __name__ == "__main__":
    # Create an in-memory account or use an existing private key.
    key = os.environ.get("SHIELD_SWAP_PRIVATE_KEY")
    private_key = testnet.PrivateKey.from_string(key) if key else testnet.PrivateKey.random()
    aleo = Aleo(HTTPProvider("https://edge.provable.com/api", network="testnet"))
    account = aleo.account.from_private_key(private_key)
    aleo.records.register(account)
    dex = ShieldSwap(aleo)
    address = str(account.address)

    # Optionally retain this account's swap handles in the SDK journal.
    if ENABLE_JOURNAL:
        dex.journal = Journal(f"testnet-{address}.jsonl")

    # Sign the API challenge with the account's private key.
    dex.api.authenticate(address, lambda message: str(private_key.sign(message.encode())))

    # Request testnet tokens and wait for the faucet job to settle.
    funding = dex.api.confirm_airdrop(address)

    # Look up USDCx and ETH, then find a direct pool.
    source = dex.api.get_token("USDCx")
    target = dex.api.get_token("ETH")
    pool = next(pool for pool in dex.api.get_pools()
                if {pool.token0, pool.token1} == {source.id, target.id})

    # Quote the selected pool in token units.
    amount_in = "1.5"
    quote = dex.api.get_route(
        token_in=source.id, token_out=target.id, amount_in=amount_in, pool_key=pool.key,
    )
    if not quote.estimated_amount_out:
        raise RuntimeError("The selected pool returned no quote")

    # Submit one swap and wait for confirmation. Keep the returned handle for the claim.
    handle = dex.swap(
        pool_key=pool.key,
        token_in_id=source.id,
        amount_in=amount_in,
        expected_out=quote.estimated_amount_out,
        slippage_bps=50,
    ).delegate(wait=True)

    # Claim the confirmed swap's output once using its returned handle.
    claim = dex.claim_swap_output(handle).delegate(wait=True)
    if dex.journal is not None:
        dex.journal.record_claim(handle.swap_id, claim.transaction_id, claim.amount_out)
    if claim.amount_out <= 0:
        raise RuntimeError("The claim returned no ETH")
