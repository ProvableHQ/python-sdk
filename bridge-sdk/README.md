# aleo-bridge-sdk

Python SDK for bridging assets between Aleo, Ethereum and Solana over the reviewed Hyperlane warp
routes and Circle xReserve deployments — a port of veil's `@provablehq/aleo-bridge-sdk` 0.1.0 into
the web3.py-style verb structure of `aleo-sdk`.

## Install

    pip install aleo-bridge-sdk            # Aleo legs only
    pip install 'aleo-bridge-sdk[evm]'     # + Ethereum (web3, eth-account)
    pip install 'aleo-bridge-sdk[solana]'  # + Solana (solders, solana)

## Use (Aleo side)

    from aleo import Aleo, HTTPProvider
    from aleo_bridge import Bridge

    aleo = Aleo(HTTPProvider("https://edge.provable.com/api", network="mainnet"))
    aleo.default_account = aleo.account.from_private_key(key)
    bridge = Bridge(aleo)                       # or Bridge.from_env() / Bridge.from_profile()

    bridge.status()                             # addresses + public balances, read-only
    bridge.hyperlane.quote_gas_payment("aleo/wbtc")
    call = bridge.hyperlane.transfer_remote("aleo/wbtc", "0xRecipient", amount="0.0001", as_signer=True)
    call.simulate()                             # local authorization, nothing sent
    receipt = call.delegate()                   # DPS proves, fee master pays, broadcast
    bridge.xreserve.burn("0xRecipient", amount="2.5")           # private USDCx → USDC
    bridge.shield("aleo/eth", amount="0.01"); bridge.unshield("aleo/usdcx", amount="2.5")

Reads return values; writes return an `AleoCall` with `simulate() / prove() / transact() / delegate()`.
Lifecycle verbs (`quote → execute → wait`, `recover/resume/complete`), Ethereum and Solana origins,
and the agent/MCP surface arrive in the following plans.

## Environment

`BRIDGE_PRIVATE_KEY` (required by `from_env`), `ALEO_ENDPOINT` (default `https://edge.provable.com/api`),
`ALEO_NETWORK` (`mainnet`|`testnet`), `ALEO_API_KEY`/`ALEO_CONSUMER_ID` (legacy hosts),
`EVM_PRIVATE_KEY`+`ETHEREUM_RPC_URL`, `SOLANA_PRIVATE_KEY`(+`SOLANA_RPC_URL`), `BRIDGE_CHECKPOINT_DIR`.
Profiles live at `$ALEO_BRIDGE_HOME` or `~/.aleo-bridge` and hold only the Aleo key (mode 600).

## Tests

    cd bridge-sdk && .venv/bin/python -m pytest -q                          # hermetic
    BRIDGE_LIVE_READS=1 .venv/bin/python -m pytest -m live tests/live -q    # read-only mainnet checks
    BRIDGE_LIVE_READS=1 BRIDGE_LIVE_SIMULATE=1 .venv/bin/python -m pytest -m live tests/live -q

Literals and vectors: `docs/veil-brief.md`.
