# aleo-bridge-sdk

Move assets between **Aleo**, **Ethereum** and **Solana** from Python, over the
reviewed Hyperlane warp routes (ETH, WBTC, USDT, SOL) and Circle xReserve
(USDC ↔ USDCx). Web3.py idioms on top of the `aleo` facade: reads return
values, writes return prepared calls, the lifecycle is `quote → execute →
wait`, and `recover` / `resume` / `complete` pick up wherever a process died.
It is a port of veil's `@provablehq/aleo-bridge-sdk` 0.1.0 (registry
`2026-08-31.solana-deposits.1`) into the web3.py-style verb structure of
`aleo-sdk`.

```sh
pip install aleo-bridge-sdk               # Aleo legs only
pip install 'aleo-bridge-sdk[evm]'        # + Ethereum (web3, eth-account)
pip install 'aleo-bridge-sdk[solana]'     # + Solana (solders, solana)
pip install 'aleo-bridge-sdk[evm,solana]' # everything
```

Import name: `aleo_bridge`.

## Quick start

```python
from aleo_bridge import Bridge

bridge = Bridge.from_env()                     # keys + RPCs from the environment (table below)
print(bridge.status())                         # addresses, balances of every bridge asset, pending transfers

quote = bridge.quote("ethereum/wbtc", "aleo/wbtc", amount="0.001", recipient=bridge.aleo_address())
print(quote.kind, quote.fees, quote.amount_out) # show fees + amount_out before moving anything

progress = bridge.execute(quote.plan)          # approval (if needed) + dispatch; checkpoints saved
progress = bridge.wait(progress)               # stops at resume / complete / done / failed
if progress.next == "resume":   progress = bridge.wait(bridge.resume(progress))
if progress.next == "complete": progress = bridge.wait(bridge.complete(progress, secret_nonce="…"))
assert progress.next == "done", progress.error
```

`Bridge.from_profile()` instead keeps an Aleo key and a checkpoint store under
`~/.aleo-bridge/` (`$ALEO_BRIDGE_HOME`), created on first use — EVM/Solana keys
still come from the arguments or the environment and are never written to disk.

## Connections

```python
from aleo import Aleo, HTTPProvider
from aleo_bridge import Bridge, Ethereum, Solana

aleo = Aleo(HTTPProvider("https://edge.provable.com/api", network="mainnet"))
aleo.default_account = aleo.account.from_private_key(aleo_key)

bridge = Bridge(aleo)                                                             # Aleo legs only
bridge = Bridge(aleo, ethereum=Ethereum(ETH_RPC, private_key=evm_key),
                      solana=Solana(SOL_RPC, private_key=sol_key))                 # raw keys
bridge = Bridge(aleo, ethereum=Ethereum(w3=my_w3, signer=my_local_account),
                      solana=Solana(client=my_client, signer=my_keypair))          # configured signers
bridge = Bridge(aleo, ethereum=my_w3, solana=my_client)                           # bare clients: read-only
```

| `Ethereum(...)` | Transport | Signer |
| --- | --- | --- |
| `rpc_url` + `private_key` / `signer` | `Web3(HTTPProvider(rpc_url))` | the key or `LocalAccount` |
| `w3` + `private_key` / `signer` | your `Web3` (middleware, PoA, retries) | the key or `LocalAccount` |
| `w3` alone | your `Web3` | `w3.eth.default_account` via your signing middleware, else read-only |

| `Solana(...)` | Transport | Signer |
| --- | --- | --- |
| `rpc_url` (default `api.mainnet-beta.solana.com`) + `private_key` / `signer` | the SDK's own synchronous `SolanaRpcClient(rpc_url, commitment="confirmed")` — a `requests`-based JSON-RPC client, since solana-py ≥ 0.36 ships only an async client | base58 (Phantom export) / the 64-int `id.json` array, a solders `Keypair`, or any solana-py `Signer` |
| `client` + `private_key` / `signer` | your client reused: a `SolanaRpcClient`, a solana-py `AsyncClient` (adapted internally through `_AsyncClientAdapter`), or any duck-typed object with the same RPC methods | as above |
| `client` alone | your client | read-only |

`Solana` supports `close()` and use as a context manager (`with Solana(...) as
solana:`) to release the wrapped client's resources; `__exit__` swallows
whatever `close()` raises, so call it directly if you need to see the error.
`Bridge.from_env()` reads `EVM_PRIVATE_KEY` + `ETHEREUM_RPC_URL` and
`SOLANA_PRIVATE_KEY` (+ `SOLANA_RPC_URL`) — see the aliases and precedence
notes in the environment table below.

## Routes

Assets are `"chain/key"` (`"ethereum/usdc"`, `"aleo/usdcx"`); `bridge.registry`
lists everything. Active mainnet routes:

| Route | Protocol | Minimum | Notes |
| --- | --- | --- | --- |
| `xreserve:ethereum/usdc->aleo/usdcx` | Circle xReserve | 2 USDC | `mint_mode` public / record / **private** (you finish with `complete`) |
| `xreserve:aleo/usdcx->ethereum/usdc` | Circle xReserve | > 2 USDCx | 2 USDCx withdrawal fee; private burn (default) needs a record + exclusion proof — computed for you |
| `hyperlane:ethereum/eth->aleo/eth` / reverse | Hyperlane (native) | 1 wei | `msg.value` carries ETH + relayer fee |
| `hyperlane:ethereum/wbtc->aleo/wbtc` / reverse | Hyperlane (collateral) | 1 sat | approval + dispatch |
| `hyperlane:ethereum/usdt->aleo/usdt` / reverse | Hyperlane (collateral) | 1 µUSDT | approval reset to 0 first (USDT) |
| `hyperlane:solana/sol->aleo/sol` / `hyperlane:aleo/sol->solana/sol` | Hyperlane | 1 lamport | IGP + rent quoted live |

Testnet: `xreserve:sepolia/usdc->aleo-testnet/usdcx` and its reverse. ALEO and
USAD routes, plus every `base`/`hyperevm` route, are `metadata-required`:
listed, refused by `quote`/`execute` until their deployments are reviewed
upstream.

Aleo-origin Hyperlane transfers spend PUBLIC balances (`unshield` first);
Hyperlane delivers into public balances (`shield` afterwards if you want).

**Registry fee vs. live fee — a known discrepancy.** The registry's
`xreserve:*usdcx->*usdc` literal is `withdrawalFeeAtomic = 2_000_000` (2
USDCx). The 2026-09-18 testnet round trip (next section) actually delivered a
live xReserve withdrawal fee of **≈1.0035 USDC**, not 2 USDC (2.000001 USDCx
burned delivered 0.996501 USDC) — Circle's fee is evidently dynamic and the
registry literal has not been re-measured against it. This SDK does not change
the registry literal, but it does not advertise it as exact either: the fee on
that route is reported as `estimated=True`, so `amount_out` is a LOWER BOUND —
you may receive more, never less. The burn minimum still uses the literal (it
must stay conservative), and the same literal is used on the mainnet route, so
budget for the same gap there until someone re-measures it live.

## The lifecycle

```python
quote    = bridge.quote(source, destination, amount="…" | amount_atomic=…, recipient=…,
                        sender=None, protocol=None, mint_mode="public", secret_nonce="0scalar")
progress = bridge.execute(quote.plan, on_checkpoint=save, proving="delegate", mode=None,
                          record=None, merkle_proof=None, gas_payment_microcredits=None,
                          secret_nonce=None, poll_seconds=1.0, timeout_seconds=120.0)
progress = bridge.wait(progress, until=None, poll_seconds=15.0, timeout_seconds=1200.0, on_update=None,
                       on_error=None, max_consecutive_errors=5)
receipt  = bridge.get_status(plan, receipt)          # one refresh, no polling
progress = bridge.recover(checkpoint)                # reads only
progress = bridge.resume(progress, on_checkpoint=save)
progress = bridge.complete(progress, secret_nonce="…", on_checkpoint=save)
bridge.pending()                                     # offline, from the checkpoint store
```

`execute` emits a `Checkpoint` at every boundary: after each approval hash,
after proving and **before** broadcast for Aleo legs, and after broadcast.
Aleo legs prove through the delegated prover (`proving="delegate"`) or locally
(`proving="local"`) and are broadcast from the checkpointed bytes, so a crash
between proving and broadcast is resumable without proving twice. Every EVM
and Solana write checkpoints its own hash **before** polling for the receipt,
so a crash mid-poll never re-signs; `send()` is single-use per transfer —
never call it twice for the same plan, recover from the checkpoint instead.

| `progress.next` | Status | Meaning | You do |
| --- | --- | --- | --- |
| `wait` | source confirming / attestation pending / delivery pending | in flight | `wait(progress)` |
| `resume` | `SOURCE_SUBMISSION_PENDING` | approval confirmed or proof built, transfer not submitted | `resume(progress)` |
| `complete` | `DESTINATION_ACTION_REQUIRED` | Circle attested; your private mint needs your signature | `complete(progress, secret_nonce=…)` |
| `done` | `COMPLETED` | delivered (destination Mailbox / nullifier / balance verified) | nothing |
| `failed` | `FAILED` / `EXPIRED` | see `progress.error` | new quote |

A `PollingTimeoutError` from `wait` is **not** a failure: the transfer is still
in flight (`exc.progress`); call `wait` again or `recover` later. This holds
for every leg — Ethereum, Solana and Aleo receipts all return "pending", never
raise, on a polling timeout; a receipt timeout is never treated as a delivery
failure. Once the deposit / dispatch / burn is broadcast, never call `execute`
(or `send()`/`build()` again for the same call) for the same transfer —
recover from the checkpoint.

## Recovery

```python
from aleo_bridge import FileCheckpointStore
bridge = Bridge.from_env(checkpoints=FileCheckpointStore("~/.aleo-bridge/checkpoints"))  # or BRIDGE_CHECKPOINT_DIR
for progress in bridge.pending():                      # after a restart
    if progress.next == "wait":     progress = bridge.wait(progress)
    if progress.next == "resume":   progress = bridge.wait(bridge.resume(progress))
    if progress.next == "complete": progress = bridge.wait(bridge.complete(progress, secret_nonce=my_nonce))
```

Checkpoints are an allowlist (version 1): intent, route id + registry version,
transaction ids, the proved Aleo transaction, the Solana blockhash lifetime and
the delivery baseline. Never keys, record plaintext, `secret_nonce`,
attestations, payloads or hashes. `recover` re-resolves the route from the live
registry and refuses a registry-version mismatch. Idempotent rebroadcast: a
duplicate-transaction response from the node is success. `FileCheckpointStore`
writes mode-600 files via an atomic rename; profile checkpoints (under
`~/.aleo-bridge/`) get the same treatment. `Bridge(checkpoints=store)` also
auto-saves through every lifecycle call, the same channel Ethereum, Solana and
Aleo writes all use — you don't have to pass `on_checkpoint=` yourself unless
you want a second sink.

## Private USDCx mints

`mint_mode="private"` commits `(recipient, secret_nonce)` on Ethereum; only the
recipient's Aleo key can `complete` the mint, with the same `secret_nonce`. The
SDK never stores the nonce (default `0scalar`). Public and record mints are
relayer-driven and finish at `done` when the bridge nullifier confirms delivery.

## shield / unshield

```python
bridge.shield("aleo/eth", amount="0.001").delegate()                  # public balance → private record (ARC-20)
bridge.unshield("aleo/usdcx", amount="5").delegate()                  # ARC-22: record + freeze-list exclusion proof, computed for you
bridge.freezelist.exclusion_proof(address, "usdcx_stablecoin.aleo")   # the `[MerkleProof; 2]` literal itself
```

**Private record selection needs the hosted record scanner — and your VIEW
KEY.** Anything that picks a private USDCx record for you
(`bridge.privacy.select_record`, and therefore `bridge.unshield` and the
default private xReserve burn) reads your records through the hosted record
scanner. The scanner answers nothing for an account it has never been
registered for, and **registering shares that account's view key** with the
scanning service, which can then decrypt every record the account owns. That
is your decision, so the SDK never registers on your behalf — do it
explicitly:

```python
bridge.aleo.records.register(bridge.aleo.default_account)   # shares this account's VIEW KEY
```

Until you do, record selection raises a `ConfigurationError` saying exactly
this. You can skip the scanner entirely by passing `record=` yourself, or by
using a public transfer / burn (`mode="public"`).

## Tier 2 modules

`bridge.hyperlane.transfer_remote / quote_gas_payment / is_delivered`,
`bridge.xreserve.burn / private_mint / get_attestation / is_delivered / hook_data`,
`bridge.eth.transfer_remote / deposit_usdc / quote_transfer_remote / quote_deposit_usdc / balance / is_delivered
/ source_status / recover_source`,
`bridge.sol.transfer_remote / quote_transfer_remote / balance`. Aleo writes return an
`AleoCall` (`simulate() / prove() / delegate_prepared() / submit_prepared() /
transact() / delegate()`); EVM and Solana writes return `EvmCall` / `SolCall`
(`build()` → unsigned, `send()`). `python -m aleo_bridge` prints the full
generated reference (`AGENTS.md`).

### Ethereum, directly

```python
quote = bridge.eth.quote_transfer_remote("wbtc", aleo_recipient, amount="0.001")
print(quote.native_fee_atomic, quote.approval_required)

call = bridge.eth.transfer_remote("wbtc", aleo_recipient, amount="0.001")
call.build()                                  # unsigned tx dicts: approve(s) then transferRemote
result = call.send(on_checkpoint=store.save)  # approvals → dispatch; each hash checkpointed before polling
result.message_id, result.receipt.status      # Hyperlane message id, DELIVERY_PENDING

usdc_quote = bridge.eth.quote_deposit_usdc(aleo_recipient, amount="2", mint_mode="public")
deposit = bridge.eth.deposit_usdc(aleo_recipient, amount="2", mint_mode="public").send()
deposit.message_hash                          # Circle attestation lookup key (receipt id), ATTESTATION_PENDING

bridge.eth.balance("eth"); bridge.eth.is_delivered(message_id)          # reads
bridge.eth.source_status(plan, receipt)                                 # one refresh of an approval/confirming receipt
bridge.eth.recover_source(plan, checkpoint)                             # log-scan recovery, never signs
```

Routes: ETH (native), WBTC and USDT (collateral; USDT resets a non-zero
allowance to 0 first) via Hyperlane; USDC → USDCx via Circle xReserve (2 USDC
minimum, `mint_mode` public/record/private — private deposits go to the
shielded wrapper program and need the same `secret_nonce` at `complete` time;
the SDK never stores it). A receipt timeout returns a pending receipt, never a
failure.

Fees and nonces are filled defensively. Public RPC endpoints answer
`eth_maxPriorityFeePerGas` with 0, and a zero-tip transaction can sit unmined
for hours, so every EIP-1559 tip is raised to a floor of 0.1 gwei
(`aleo_bridge.eth.MIN_PRIORITY_FEE_WEI`); pass
`Ethereum(..., min_priority_fee_wei=…)` or set `BRIDGE_MIN_PRIORITY_FEE_WEI`
(whole wei, read by `from_env`) to raise or lower it (`maxFeePerGas` stays
`2 * baseFeePerGas + tip`). The highest nonce broadcast for a
`(chain, sender)` is also remembered **per process** — not across processes or
machines — and never handed out again, because a load-balanced RPC can stop
reporting our own pending transaction and the next leg (typically a fresh
`Ethereum`) would then replace it. If a source transaction is dropped or
replaced anyway, it no longer waits forever: once two probes agree the hash is
in no block — the node either does not know it or still serves it with a null
`blockNumber`, which is what a load-balanced endpoint does for a replaced
transaction — and the account nonce has moved past the nonce it was sent at,
`source_status` returns `EXPIRED` with a `sourceError` explaining that nothing
moved, and its checkpoint is kept rather than deleted so `recover()` can
re-scan. A transaction that comes back with a block number is mined, however far
behind its receipt read is.
`recover_source` takes that verdict first and then scans source history — a
dispatch that really landed wins — and hands back a resumable
`SOURCE_SUBMISSION_PENDING` when the scan ran, covered the head the verdict was
taken at, and found nothing; `resume()` then supersedes the dropped record with
the new transaction's. If a transfer was finished outside the SDK, its kept
dropped record stays listed in `pending()` until you clear it yourself with
`store.delete(checkpoint_id)` (the id is the source transaction hash).

Live checks: `BRIDGE_LIVE_READS=1 ETHEREUM_RPC_URL=…` for read-only
mainnet quotes (`tests/live/test_eth_reads.py`); `BRIDGE_LIVE_FUNDS=1
BRIDGE_LIVE_STATE_DIR=… SEPOLIA_RPC_URL=… EVM_PRIVATE_KEY=…
ALEO_E2E_PRIVATE_KEY=…` for the 2 USDC Sepolia leg
(`tests/live/test_eth_sepolia_leg1.py`).

### Solana, directly

```python
quote = bridge.sol.quote_transfer_remote(aleo_addr, amount="0.01")            # amount + IGP + fee + rent, in lamports
call = bridge.sol.transfer_remote(aleo_addr, amount="0.01")
tx = call.build()                                                             # VersionedTransaction, unique-message key signed
result = call.send(on_checkpoint=store.save)                                  # fee-payer signature, broadcast, poll to confirmed
result.message_id                                                             # Hyperlane message id from the Mailbox log
```

`private_key` accepts a base58 secret (Phantom export) or the 64-int JSON
array of a solana-cli `id.json`. Every read uses confirmed commitment; the
transaction sets a 400,000 compute-unit limit; the `SOURCE_CONFIRMING` receipt
(signature, unique-message address, blockhash, last valid block height) is
checkpointed before polling, and a polling timeout returns the pending receipt
rather than failing. The instruction encoding and account list are pinned
byte-for-byte against a recorded mainnet transfer
(`tests/fixtures/sealevel-transfer-remote.json`).

Live read-only checks (no key, no funds):
`BRIDGE_LIVE_READS=1 .venv/bin/python -m pytest -m live tests/live/test_sol_reads.py -q -s`
decodes the live IGP account and prints a leg-11 quote for a pinned sender;
`SOLANA_RPC_URL` overrides the public default if it rate-limits. The funded
round trip runs from `scripts/rehearse.py` (see "Live rehearsal" below).

## Agents and MCP

```python
from aleo_bridge import bridge_tools, dispatch_tool
tools = bridge_tools()                 # Claude `tools=` shape; bridge_tools(include_writes=False) for read-only
dispatch_tool(bridge, "bridge_quote", {"source": "ethereum/usdc", "destination": "aleo/usdcx", "amount": "2", "recipient": addr})
```

Reads: `bridge_status`, `bridge_list_assets`, `bridge_list_routes`, `bridge_quote`,
`bridge_get_progress`, `bridge_pending`. Writes (`bridge_execute`, `bridge_resume`,
`bridge_complete`, `bridge_shield`, `bridge_unshield`) require `confirm: true` —
without it they return the quote (or recovered progress / built call) plus
`how_to_confirm`, and move nothing. `bridge_execute` takes the quote inputs and
re-quotes internally, so agents never carry plans.
`python -m aleo_bridge.mcp` serves the same tools over stdio (`[mcp]` extra).

## Live tests and rehearsal

The funded suite (`tests/live/`, ported from veil's `test/integration/live/`)
moves real money. It is off unless you turn it on, in your own shell, one
command at a time. `scripts/rehearse.py` drives the same case functions the
live pytest suite uses, over one route or a whole case, at minimum amounts,
and backs both the CLI and `-m live` so they cannot drift.

**The testnet round trip has run for real, end to end, both directions**
(2026-09-18): Sepolia deposit of 3 USDC
(`0x08cd56e4a10c84d62ee000ceb20d854d2f1b6b9a6102db8f1414abfb871faab4`) into a
private USDCx mint on aleo-testnet
(`at1jdy6wyal4ndwwydy80hjxt5q9eh332vk6t8zrw8jsvgc0pg7jcxqc85day`), then a
return burn
(`at1cufuy7v4lc4dn5ek3vdfa0tpeh8rrqqnahyelv06d2ll5n0kvczsrv9ugq`) delivering a
Sepolia withdrawal
(`0xbacb3ff270f6cee401cf2aa9ce6ed3be3b9d0426b18c04a2674c86dca11e4b54`).
Recovery from the on-disk checkpoint was exercised on both legs — the process
that submitted was discarded and a fresh `Bridge` over the same
`FileCheckpointStore` finished each transfer from `bridge.pending()`. The
observed withdrawal fee (see "Registry fee vs. live fee" above) is the one
concrete correction those runs produced. Every mainnet fund-moving case has
been quoted and prechecked live (funding table below) but still runs
quote-only — mainnet execution needs the operator to fund the bridge wallets
first, which has not happened yet; this README does not carry wallet
addresses or balances.

**Gates** (read only — nothing in this repository sets them; the exact
acknowledgement strings are the constants in `tests/live/config.py`):

| Variable | Effect |
| --- | --- |
| `BRIDGE_LIVE_FUNDS=1` + `BRIDGE_LIVE_STATE_DIR=<dir outside the repo>` | funded cases exist at all |
| `BRIDGE_LIVE_MAINNET_ACK=…` + `BRIDGE_LIVE_MAINNET_CASES=<comma list>` | the named mainnet cases may run |
| `BRIDGE_LIVE_MAINNET_EXECUTE=…` | the wallet may actually submit |

Without the last one every case runs to the quote, prints the route/amount/fee
table and returns — that is the default, and it is how you rehearse. Keys and
endpoints come from `Bridge.from_env()` (`BRIDGE_PRIVATE_KEY`,
`EVM_PRIVATE_KEY`/`BRIDGE_EVM_PRIVATE_KEY`, `SOLANA_PRIVATE_KEY`/
`BRIDGE_SOLANA_PRIVATE_KEY`, `ETHEREUM_RPC_URL`/`BRIDGE_LIVE_ETHEREUM_RPC_URL`,
`SOLANA_RPC_URL`/`BRIDGE_LIVE_SOLANA_RPC_URL`); recipients default to your own
addresses and can be overridden with `BRIDGE_LIVE_ALEO_MAINNET_RECIPIENT` /
`BRIDGE_LIVE_EVM_RECIPIENT` / `BRIDGE_LIVE_SOLANA_RECIPIENT`.

**State.** Each case keeps one file at
`$BRIDGE_LIVE_STATE_DIR/<environment>/<case>-<route>.json` (mode 600) holding
the checkpoint, the source/destination transaction ids and the message id;
the private-mint secret nonce lives beside it in `<state>.secret` (mode 600,
created exclusively) and never in the state, a log or a checkpoint.
Re-running a case resumes from that file — recover first, then
`wait`/`resume`/`complete`; a completed case re-asserts what it recorded and
exits. A timeout is *pending*, not a failure: the checkpoint stays on disk
and the run prints the `--recover` command.

**Cases and routes.** Five cases, each parametrized over every registry route
it covers — both directions are separate cases, and a route with
`availability != "active"` is reported as skipped-by-registry rather than
dropped:

| Case | Routes | Amount |
| --- | --- | --- |
| `evm-hyperlane` | ethereum → aleo (ETH, WBTC, USDT) | one atomic unit |
| `aleo-hyperlane` | aleo → ethereum (ETH, WBTC, USDT), aleo → solana (SOL) | one atomic unit, `mode="signer"` |
| `solana-hyperlane` | solana → aleo (SOL) | 1 lamport |
| `evm-xreserve` | ethereum USDC → aleo USDCx | `2` USDC, private mint (needs `complete`) |
| `aleo-xreserve` | aleo USDCx → ethereum USDC | `2.000001` USDCx private burn, delivers 1 atomic unit |

The `metadata-required` mainnet routes (ALEO on ethereum/solana/base/hyperevm,
USAD) are parametrized too and skip with `registry:metadata-required` —
`tests/live/test_lifecycle_live.py` builds its parameters by enumerating
`DEFAULT_REGISTRY.routes(...)`, so a route that no case covers raises at
import rather than disappearing. The same two xReserve case functions run the
testnet pair (`xreserve:sepolia/usdc->aleo-testnet/usdcx` at `3` USDC, then
`xreserve:aleo-testnet/usdcx->sepolia/usdc` at `2.000001`); the testnet keys
and RPC are `BRIDGE_LIVE_ALEO_TESTNET_PRIVATE_KEY`/`ALEO_E2E_PRIVATE_KEY`,
`BRIDGE_LIVE_EVM_TESTNET_PRIVATE_KEY`/`EVM_PRIVATE_KEY`/`BRIDGE_EVM_PRIVATE_KEY`
and `SEPOLIA_RPC_URL`/`BRIDGE_LIVE_SEPOLIA_RPC_URL` (public default when
unset). Testnet needs no mainnet acknowledgement — the funds gate alone.

**Recovery is not simulated.** Every funded test runs in two phases: the
first quotes, prechecks and calls `execute` once, then returns; the client
that executed is discarded, and a brand-new `Bridge` over the same
`FileCheckpointStore` finishes the transfer from `bridge.pending()` →
`bridge.recover(checkpoint)` → `wait`/`resume`/`complete`. `execute` is never
called twice for one transfer, and `BRIDGE_LIVE_XRESERVE_AMOUNT` (or
`BRIDGE_LIVE_<CASE>_AMOUNT`) overrides an amount.

**Funding per run** (from the 2026-09-17 read-only mainnet quote sweep):
ETH/WBTC/USDT Hyperlane deposits cost ≈0.0000838 ETH each in native Hyperlane
fees plus L1 gas (USDT also needs one approval); the Aleo-origin legs cost
8.174147 (ETH), 9.138947 (WBTC), 9.138947 (USDT) and 7.661056 (SOL) credits in
IGP payment plus the Aleo transaction fee, and need the asset's public
balance on Aleo; solana → aleo costs 5,647,521 lamports all-in; the xReserve
deposit needs 2 USDC plus gas, and the burn needs an unspent private USDCx
record of at least 2.000001. Return legs spend what the matching inbound leg
minted, so run inbound first — an unfunded return leg skips with its
shortfall printed rather than failing.

**Running it.** *Quote only* — prices and prechecks every route and submits
nothing, whatever is acknowledged:

```sh
python scripts/rehearse.py --case evm-hyperlane --quote-only
python scripts/rehearse.py --case evm-xreserve --report run.json      # submits only if acknowledged
```

*Testnet* (Sepolia ⇄ aleo-testnet, the funds gate only — no mainnet
acknowledgement):

```sh
BRIDGE_LIVE_FUNDS=1 BRIDGE_LIVE_STATE_DIR="$HOME/.bridge-live" \
  pytest -m live -s tests/live/test_lifecycle_live.py::test_testnet_evm_xreserve_deposit
BRIDGE_LIVE_FUNDS=1 BRIDGE_LIVE_STATE_DIR="$HOME/.bridge-live" \
  pytest -m live -s tests/live/test_lifecycle_live.py::test_testnet_aleo_xreserve_return
```

*Mainnet, quote only* — the acknowledgement that names the cases, and
deliberately no execute variable, so each route is priced and prechecked and
nothing is signed:

```sh
BRIDGE_LIVE_FUNDS=1 BRIDGE_LIVE_STATE_DIR="$HOME/.bridge-live" \
  BRIDGE_LIVE_MAINNET_ACK=<see tests/live/config.py> \
  BRIDGE_LIVE_MAINNET_CASES=evm-hyperlane,evm-xreserve,aleo-hyperlane,aleo-xreserve,solana-hyperlane \
  pytest -m live -s tests/live/test_lifecycle_live.py
```

*Mainnet, for real* — **you** type this, in your own shell, for one command;
both acknowledgement values are the constants in `tests/live/config.py` and
appear nowhere in this repository in a copy-pasteable form. Nothing in the
suite or the CLI ever sets them:

```sh
BRIDGE_LIVE_FUNDS=1 BRIDGE_LIVE_STATE_DIR="$HOME/.bridge-live" \
  BRIDGE_LIVE_MAINNET_ACK=<see tests/live/config.py> \
  BRIDGE_LIVE_MAINNET_CASES=<the one case you mean> \
  BRIDGE_LIVE_MAINNET_EXECUTE=<see tests/live/config.py> \
  pytest -m live -s "tests/live/test_lifecycle_live.py::test_evm_xreserve"
```

*Resuming* an interrupted transfer (the run prints this line itself):

```sh
python scripts/rehearse.py --recover "$BRIDGE_LIVE_STATE_DIR/mainnet/<case>-<route>.json"
```

Exit codes: `0` ok, `1` a case failed, `2` something is still pending. The
harness itself (route selection, state paths, the exit-code table) is
covered hermetically by `tests/test_live_helpers.py`.

## Environment variables

| Variable | Used by | Meaning |
| --- | --- | --- |
| `BRIDGE_PRIVATE_KEY` | `from_env`, `from_profile` (import), MCP | Aleo private key (`APrivateKey1…`), **required** by `from_env` |
| `ALEO_ENDPOINT` | `from_env` | node API origin (default `https://edge.provable.com/api`) |
| `ALEO_NETWORK` | `from_env` | `mainnet` (default) or `testnet` |
| `ALEO_API_KEY`, `ALEO_CONSUMER_ID` | `from_env` | optional Provable credentials for legacy endpoints |
| `EVM_PRIVATE_KEY`, `ETHEREUM_RPC_URL` | `from_env`, rehearsal | Ethereum signer + RPC (both or neither); aliases `BRIDGE_EVM_PRIVATE_KEY` / `BRIDGE_LIVE_ETHEREUM_RPC_URL` — used by the user's live shell/veil config, and NOT live-test-only: ordinary `Ethereum.from_env()` reads them too, so leaving one exported points everyday calls at that endpoint; the primary variable wins when both are set |
| `SOLANA_PRIVATE_KEY`, `SOLANA_RPC_URL` | `from_env`, rehearsal | Solana signer (base58 or `id.json` array) + RPC (optional); aliases `BRIDGE_SOLANA_PRIVATE_KEY` / `BRIDGE_LIVE_SOLANA_RPC_URL`, same precedence and same everyday-call caveat as the Ethereum pair |
| `BRIDGE_MIN_PRIORITY_FEE_WEI` | `Ethereum.from_env` | override the EIP-1559 tip floor (whole wei, digits only); default 0.1 gwei |
| `BRIDGE_CHECKPOINT_DIR` | `from_env` | bind a `FileCheckpointStore` |
| `ALEO_BRIDGE_HOME` | `from_profile` | profile directory (default `~/.aleo-bridge/`), holds only the Aleo key, mode 600 |
| `ALEO_E2E_PRIVATE_KEY` | live tests / rehearsal, testnet | testnet Aleo key (alias `BRIDGE_LIVE_ALEO_TESTNET_PRIVATE_KEY`) |
| `BRIDGE_LIVE_FUNDS`, `BRIDGE_LIVE_STATE_DIR` | live tests, rehearsal | `1` + a directory outside the repo — funded cases exist at all |
| `BRIDGE_LIVE_MAINNET_ACK`, `BRIDGE_LIVE_MAINNET_CASES` | live tests, rehearsal | `<see tests/live/config.py>` + a comma list of `evm-hyperlane`, `evm-xreserve`, `aleo-hyperlane`, `aleo-xreserve`, `solana-hyperlane` — the named mainnet cases may run |
| `BRIDGE_LIVE_MAINNET_EXECUTE` | live tests, rehearsal | `<see tests/live/config.py>` — without it every case quotes and prechecks only |

The acknowledgement strings live in `tests/live/config.py` and its hermetic
test; nothing in this repository exports them or sets them for a real run.

## Tests

```sh
cd bridge-sdk && .venv/bin/python -m pytest -q                          # hermetic
.venv/bin/python -m pytest -q -m "not live"                             # same set, explicit marker
BRIDGE_LIVE_READS=1 .venv/bin/python -m pytest -m live tests/live -q    # read-only mainnet checks
BRIDGE_LIVE_READS=1 BRIDGE_LIVE_SIMULATE=1 .venv/bin/python -m pytest -m live tests/live -q
```

Literals and vectors: `docs/veil-brief.md`.

## Development

```sh
cd bridge-sdk && python -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q                        # unit + mocked integration
.venv/bin/python codegen/gen_context.py --check      # AGENTS.md is generated from docstrings
```
