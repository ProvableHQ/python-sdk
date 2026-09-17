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
Lifecycle verbs (`quote → execute → wait`, `recover/resume/complete`), Solana origins, and the
agent/MCP surface arrive in the following plans.

## Ethereum

    from web3 import Web3
    from aleo_bridge import Bridge, Ethereum

    bridge = Bridge(aleo, ethereum=Ethereum("https://eth.example/rpc", private_key=evm_key))   # SDK-built transport
    bridge = Bridge(aleo, ethereum=Ethereum(w3=my_w3, signer=my_local_account))               # your Web3 + your signer
    bridge = Bridge(aleo, ethereum=my_w3)        # bare Web3: read-only, or signs via w3.eth.default_account middleware
    bridge = Bridge.from_env()                   # EVM_PRIVATE_KEY + ETHEREUM_RPC_URL (both or neither)

    quote = bridge.eth.quote_transfer_remote("wbtc", aleo_recipient, amount="0.001")
    print(quote.native_fee_atomic, quote.approval_required)

    call = bridge.eth.transfer_remote("wbtc", aleo_recipient, amount="0.001")
    call.build()                                  # unsigned tx dicts: approve(s) then transferRemote
    result = call.send(on_checkpoint=store.save)  # approvals → dispatch; each hash checkpointed before polling
    result.message_id, result.receipt.status      # Hyperlane message id, DELIVERY_PENDING

    deposit = bridge.eth.deposit_usdc(aleo_recipient, amount="2", mint_mode="public").send()
    deposit.message_hash                          # Circle attestation lookup key (receipt id), ATTESTATION_PENDING

    bridge.eth.balance("eth"); bridge.eth.is_delivered(message_id)          # reads
    bridge.eth.source_status(plan, receipt)                                 # one refresh of an approval/confirming receipt
    bridge.eth.recover_source(plan, checkpoint)                             # log-scan recovery, never signs

Routes: ETH (native), WBTC and USDT (collateral; USDT resets a non-zero allowance to 0 first) via Hyperlane;
USDC → USDCx via Circle xReserve (2 USDC minimum, `mint_mode` public/record/private — private deposits go to the
shielded wrapper program and need the same `secret_nonce` at `complete` time; the SDK never stores it).
A receipt timeout returns a pending receipt, never a failure. Live checks: `BRIDGE_LIVE_READS=1 ETHEREUM_RPC_URL=…`
for read-only mainnet quotes; `BRIDGE_LIVE_FUNDS=1 BRIDGE_LIVE_STATE_DIR=… SEPOLIA_RPC_URL=… EVM_PRIVATE_KEY=…
ALEO_E2E_PRIVATE_KEY=…` for the 2 USDC Sepolia leg.

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
