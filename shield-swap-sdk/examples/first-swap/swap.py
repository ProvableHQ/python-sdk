"""Create and fund a testnet account, swap 1.5 USDCx for ETH, and claim the output."""
import time
from decimal import Decimal

from aleo import testnet

from aleo_shield_swap import ShieldSwap


if __name__ == "__main__":
    # Load the SDK profile or create one. It retains the account and swap journal.
    dex = ShieldSwap.from_profile(network="testnet")
    if dex.profile.network != "testnet":
        raise RuntimeError("This example requires a testnet profile")

    # Sign the API challenge with the account saved in the SDK profile.
    private_key = testnet.PrivateKey.from_string(dex.profile.private_key)
    dex.api.authenticate(dex.profile.address, lambda message: str(private_key.sign(message.encode())))

    # Request testnet tokens, then wait for the faucet job to finish.
    airdrop = dex.api.request_airdrop(dex.profile.address)
    for attempt in range(120):
        funding = dex.api.get_airdrop_job(airdrop.job_id)
        if funding.status == "complete":
            break
        time.sleep(5)
    else:
        raise RuntimeError(f"Airdrop is still pending; inspect job {airdrop.job_id}")

    # Find a direct USDCx/ETH pool and convert 1.5 USDCx to base units.
    tokens = dex.api.get_tokens()
    source = next(token for token in tokens if token.symbol == "USDCx")
    target = next(token for token in tokens if token.symbol == "ETH")
    pool = next(pool for pool in dex.api.get_pools()
                if {pool.token0, pool.token1} == {source.id, target.id})

    # Wait for the scanner to report enough USDCx for the swap.
    amount_in = 15 * 10**source.decimals // 10
    token_program = source.underlying_program or source.amm_token_program
    for attempt in range(40):
        balances = dex.get_private_balances([token_program])
        if balances.get(token_program, 0) >= amount_in:
            break
        time.sleep(15)
    else:
        raise RuntimeError("USDCx is not available; inspect funding.results and the account balance")

    # Quote the selected pool and convert the expected ETH output to base units.
    quote = dex.api.get_route(
        token_in=source.id, token_out=target.id, amount_in="1.5", pool_key=pool.key,
    )
    if not quote.estimated_amount_out:
        raise RuntimeError("The selected pool returned no quote")
    expected_out = int(Decimal(quote.estimated_amount_out) * 10**target.decimals)
    if expected_out <= 0:
        raise RuntimeError("The quote returned no ETH")

    # Submit one swap and wait for confirmation. The SDK journals its handle.
    handle = dex.swap(
        pool_key=pool.key,
        token_in_id=source.id,
        amount_in=amount_in,
        expected_out=expected_out,
        slippage_bps=50,
    ).delegate(wait=True)

    # Claim the confirmed swap's output once, then update the SDK journal.
    claim = dex.claim_swap_output(handle).delegate(wait=True)
    dex.journal.record_claim(handle.swap_id, claim.transaction_id, claim.amount_out)
    if claim.amount_out <= 0:
        raise RuntimeError("The claim returned no ETH")
