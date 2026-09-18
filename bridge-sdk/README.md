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
    bridge = Bridge.from_env()                   # EVM_PRIVATE_KEY + ETHEREUM_RPC_URL (both or neither);
                                                  # aliases BRIDGE_EVM_PRIVATE_KEY / BRIDGE_LIVE_ETHEREUM_RPC_URL

    quote = bridge.eth.quote_transfer_remote("wbtc", aleo_recipient, amount="0.001")
    print(quote.native_fee_atomic, quote.approval_required)

    call = bridge.eth.transfer_remote("wbtc", aleo_recipient, amount="0.001")
    call.build()                                  # unsigned tx dicts: approve(s) then transferRemote
    result = call.send(on_checkpoint=store.save)  # approvals → dispatch; each hash checkpointed before polling
    result.message_id, result.receipt.status      # Hyperlane message id, DELIVERY_PENDING

    usdc_quote = bridge.eth.quote_deposit_usdc(aleo_recipient, amount="2", mint_mode="public")
    print(usdc_quote.balance_atomic, usdc_quote.approval_required)

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

## Solana (SOL → Aleo over Hyperlane)

Install the extra: `pip install 'aleo-bridge-sdk[solana]'` (solders + solana-py).

    from aleo_bridge import Bridge, Solana

    bridge = Bridge(aleo, solana=Solana(private_key=SOL_KEY))                     # default RPC api.mainnet-beta.solana.com
    bridge = Bridge(aleo, solana=Solana("https://my-rpc", signer=my_keypair))     # solders Keypair or any pubkey()/sign_message() signer
    bridge = Bridge(aleo, solana=my_rpc_client)                                   # bare RPC client (or solana-py AsyncClient) → read-only

    quote = bridge.sol.quote_transfer_remote(aleo_addr, amount="0.01")            # amount + IGP + fee + rent, in lamports
    call = bridge.sol.transfer_remote(aleo_addr, amount="0.01")
    tx = call.build()                                                             # VersionedTransaction, unique-message key signed
    result = call.send(on_checkpoint=store.save)                                  # fee-payer signature, broadcast, poll to confirmed
    result.message_id                                                             # Hyperlane message id from the Mailbox log

`private_key` accepts a base58 secret (Phantom export) or the 64-int JSON array of a solana-cli `id.json`.
`Bridge.from_env()` reads `SOLANA_PRIVATE_KEY` and (optionally) `SOLANA_RPC_URL`. The default transport is the
SDK's own synchronous JSON-RPC client (`aleo_bridge.sol.SolanaRpcClient`, built on `requests`); pass `client=` to
reuse your own — anything with solana-py's read/send methods, or a solana-py `AsyncClient`. Every read uses confirmed
commitment; the transaction sets a 400 000 compute-unit limit; the `SOURCE_CONFIRMING` receipt (signature,
unique-message address, blockhash, last valid block height) is checkpointed before polling, and a polling
timeout returns the pending receipt rather than failing. `on_checkpoint` is optional — passing `checkpoints=store`
to `Bridge(...)` saves every checkpoint automatically, the same channel Ethereum and Aleo calls use. The instruction
encoding and account list are pinned byte-for-byte against a recorded mainnet transfer
(`tests/fixtures/sealevel-transfer-remote.json`). `Solana` supports `close()` and use as a context manager
(`with Solana(...) as solana:`) to release the wrapped client's resources.

Live read-only checks (no key, no funds): `BRIDGE_LIVE_READS=1 .venv/bin/python -m pytest -m live tests/live/test_sol_reads.py -q -s`
decodes the live IGP account and prints a leg-11 quote for a pinned sender; `SOLANA_RPC_URL` overrides the public
default if it rate-limits. The funded round trip runs from `scripts/rehearse.py` (plan 4).

## Environment

`BRIDGE_PRIVATE_KEY` (required by `from_env`), `ALEO_ENDPOINT` (default `https://edge.provable.com/api`),
`ALEO_NETWORK` (`mainnet`|`testnet`), `ALEO_API_KEY`/`ALEO_CONSUMER_ID` (legacy hosts),
`EVM_PRIVATE_KEY`+`ETHEREUM_RPC_URL` (aliases `BRIDGE_EVM_PRIVATE_KEY`+`BRIDGE_LIVE_ETHEREUM_RPC_URL`, used by the
user's live shell/veil config; the primary variable wins when both are set),
`SOLANA_PRIVATE_KEY`(+`SOLANA_RPC_URL`) (aliases `BRIDGE_SOLANA_PRIVATE_KEY`+`BRIDGE_LIVE_SOLANA_RPC_URL`,
same precedence), `BRIDGE_CHECKPOINT_DIR`.
Note that `BRIDGE_LIVE_ETHEREUM_RPC_URL` and `BRIDGE_LIVE_SOLANA_RPC_URL` are not live-test-only: ordinary
`Ethereum.from_env()` / `Solana.from_env()` / `Bridge.from_env()` read them as aliases for
`ETHEREUM_RPC_URL` / `SOLANA_RPC_URL`, so leaving one exported points everyday calls at that endpoint too.
Profiles live at `$ALEO_BRIDGE_HOME` or `~/.aleo-bridge` and hold only the Aleo key (mode 600).

## Tests

    cd bridge-sdk && .venv/bin/python -m pytest -q                          # hermetic
    BRIDGE_LIVE_READS=1 .venv/bin/python -m pytest -m live tests/live -q    # read-only mainnet checks
    BRIDGE_LIVE_READS=1 BRIDGE_LIVE_SIMULATE=1 .venv/bin/python -m pytest -m live tests/live -q

Literals and vectors: `docs/veil-brief.md`.

## Live tests

The funded suite (`tests/live/`, ported from veil's `test/integration/live/`) moves real money. It
is off unless you turn it on, in your own shell, one command at a time.

**Gates** (read only — nothing in this repository sets them; the exact acknowledgement strings are
the constants in `tests/live/config.py`):

| Variable | Effect |
| --- | --- |
| `BRIDGE_LIVE_FUNDS=1` + `BRIDGE_LIVE_STATE_DIR=<dir outside the repo>` | funded cases exist at all |
| `BRIDGE_LIVE_MAINNET_ACK=…` + `BRIDGE_LIVE_MAINNET_CASES=<comma list>` | the named mainnet cases may run |
| `BRIDGE_LIVE_MAINNET_EXECUTE=…` | the wallet may actually submit |

Without the last one every case runs to the quote, prints the route/amount/fee table and returns —
that is the default, and it is how you rehearse. Keys and endpoints come from `Bridge.from_env()`
(`BRIDGE_PRIVATE_KEY`, `EVM_PRIVATE_KEY`/`BRIDGE_EVM_PRIVATE_KEY`, `SOLANA_PRIVATE_KEY`/
`BRIDGE_SOLANA_PRIVATE_KEY`, `ETHEREUM_RPC_URL`/`BRIDGE_LIVE_ETHEREUM_RPC_URL`,
`SOLANA_RPC_URL`/`BRIDGE_LIVE_SOLANA_RPC_URL`); recipients default to your own addresses and can be
overridden with `BRIDGE_LIVE_ALEO_MAINNET_RECIPIENT` / `BRIDGE_LIVE_EVM_RECIPIENT` /
`BRIDGE_LIVE_SOLANA_RECIPIENT`.

**State.** Each case keeps one file at `$BRIDGE_LIVE_STATE_DIR/<environment>/<case>-<route>.json`
(mode 600) holding the checkpoint, the source/destination transaction ids and the message id; the
private-mint secret nonce lives beside it in `<state>.secret` (mode 600, created exclusively) and
never in the state, a log or a checkpoint. Re-running a case resumes from that file — recover
first, then `wait`/`resume`/`complete`; a completed case re-asserts what it recorded and exits.
A timeout is *pending*, not a failure: the checkpoint stays on disk and the run prints the
`--recover` command.

**Cases and routes.** Five cases, each parametrized over every registry route it covers — both
directions are separate cases, and a route with `availability != "active"` is reported as
skipped-by-registry rather than dropped:

| Case | Routes | Amount |
| --- | --- | --- |
| `evm-hyperlane` | ethereum → aleo (ETH, WBTC, USDT) | one atomic unit |
| `aleo-hyperlane` | aleo → ethereum (ETH, WBTC, USDT), aleo → solana (SOL) | one atomic unit, `mode="signer"` |
| `solana-hyperlane` | solana → aleo (SOL) | 1 lamport |
| `evm-xreserve` | ethereum USDC → aleo USDCx | `2` USDC, private mint (needs `complete`) |
| `aleo-xreserve` | aleo USDCx → ethereum USDC | `2.000001` USDCx private burn, delivers 1 atomic unit |

The `metadata-required` mainnet routes (ALEO on ethereum/solana/base/hyperevm, USAD) are
parametrized too and skip with `registry:metadata-required` — `tests/live/test_lifecycle_live.py`
builds its parameters by enumerating `DEFAULT_REGISTRY.routes(...)`, so a route that no case covers
raises at import rather than disappearing. The same two xReserve case functions run the testnet
pair (`xreserve:sepolia/usdc->aleo-testnet/usdcx` at `3` USDC, then
`xreserve:aleo-testnet/usdcx->sepolia/usdc` at `2.000001`); the testnet keys and RPC are
`BRIDGE_LIVE_ALEO_TESTNET_PRIVATE_KEY`/`ALEO_E2E_PRIVATE_KEY`,
`BRIDGE_LIVE_EVM_TESTNET_PRIVATE_KEY`/`EVM_PRIVATE_KEY`/`BRIDGE_EVM_PRIVATE_KEY` and
`SEPOLIA_RPC_URL`/`BRIDGE_LIVE_SEPOLIA_RPC_URL` (public default when unset). Testnet needs no
mainnet acknowledgement — the funds gate alone.

**Recovery is not simulated.** Every funded test runs in two phases: the first quotes, prechecks
and calls `execute` once, then returns; the client that executed is discarded, and a brand-new
`Bridge` over the same `FileCheckpointStore` finishes the transfer from `bridge.pending()` →
`bridge.recover(checkpoint)` → `wait`/`resume`/`complete`. `execute` is never called twice for one
transfer, and `BRIDGE_LIVE_XRESERVE_AMOUNT` (or `BRIDGE_LIVE_<CASE>_AMOUNT`) overrides an amount.

**Funding per run** (from the 2026-09-17 read-only mainnet quote sweep): ETH/WBTC/USDT Hyperlane
deposits cost ≈0.0000838 ETH each in native Hyperlane fees plus L1 gas (USDT also needs one
approval); the Aleo-origin legs cost 8.174147 (ETH), 9.138947 (WBTC), 9.138947 (USDT) and 7.661056
(SOL) credits in IGP payment plus the Aleo transaction fee, and need the asset's public balance on
Aleo; solana → aleo costs 5,647,521 lamports all-in; the xReserve deposit needs 2 USDC plus gas,
and the burn needs an unspent private USDCx record of at least 2.000001. Return legs spend what
the matching inbound leg minted, so run inbound first — an unfunded return leg skips with its
shortfall printed rather than failing.

**Running it.** *Quote only* — prices and prechecks every route and submits nothing, whatever is
acknowledged:

    python scripts/rehearse.py --case evm-hyperlane --quote-only
    python scripts/rehearse.py --case evm-xreserve --report run.json      # submits only if acknowledged

*Testnet* (Sepolia ⇄ aleo-testnet, the funds gate only — no mainnet acknowledgement):

    BRIDGE_LIVE_FUNDS=1 BRIDGE_LIVE_STATE_DIR="$HOME/.bridge-live" \
      pytest -m live -s tests/live/test_lifecycle_live.py::test_testnet_evm_xreserve_deposit
    BRIDGE_LIVE_FUNDS=1 BRIDGE_LIVE_STATE_DIR="$HOME/.bridge-live" \
      pytest -m live -s tests/live/test_lifecycle_live.py::test_testnet_aleo_xreserve_return

*Mainnet, quote only* — the acknowledgement that names the cases, and deliberately no execute
variable, so each route is priced and prechecked and nothing is signed:

    BRIDGE_LIVE_FUNDS=1 BRIDGE_LIVE_STATE_DIR="$HOME/.bridge-live" \
      BRIDGE_LIVE_MAINNET_ACK=<see tests/live/config.py> \
      BRIDGE_LIVE_MAINNET_CASES=evm-hyperlane,evm-xreserve,aleo-hyperlane,aleo-xreserve,solana-hyperlane \
      pytest -m live -s tests/live/test_lifecycle_live.py

*Mainnet, for real* — **you** type this, in your own shell, for one command; both acknowledgement
values are the constants in `tests/live/config.py` and appear nowhere in this repository in a
copy-pasteable form. Nothing in the suite or the CLI ever sets them:

    BRIDGE_LIVE_FUNDS=1 BRIDGE_LIVE_STATE_DIR="$HOME/.bridge-live" \
      BRIDGE_LIVE_MAINNET_ACK=<see tests/live/config.py> \
      BRIDGE_LIVE_MAINNET_CASES=<the one case you mean> \
      BRIDGE_LIVE_MAINNET_EXECUTE=<see tests/live/config.py> \
      pytest -m live -s "tests/live/test_lifecycle_live.py::test_evm_xreserve"

*Resuming* an interrupted transfer (the run prints this line itself):

    python scripts/rehearse.py --recover "$BRIDGE_LIVE_STATE_DIR/mainnet/<case>-<route>.json"

Exit codes: `0` ok, `1` a case failed, `2` something is still pending. The same case functions back
the pytest suite (`-m live`), so the CLI and the tests cannot drift. The harness itself is covered
hermetically by `tests/test_live_helpers.py`.
