# shield-swap-sdk

Typed Python client for the **shield_swap** AMM on Aleo. Sits on top of the
Aleo Python SDK's facade (`aleo.Aleo`): signer, record provider, proving
configuration, and network all come from the client you bind — this package
adds the DEX methods, the typed results, and the off-chain DEX API, nothing
else.

```python
from aleo import Aleo
from aleo_shield_swap import ShieldSwap

aleo = Aleo(Aleo.HTTPProvider("https://edge.provable.com/api"))
aleo.default_account = account
dex = ShieldSwap(aleo)

pools = dex.api.get_pools()                      # requires a live DEX API
handle = dex.swap(pool_key=pools[0].key,
                  token_in_id=pools[0].token0,
                  amount_in=10**9).delegate()    # broadcasts; spends funds
out = dex.claim_swap_output(handle).delegate()   # broadcasts; spends funds
```

Targets the deployed `shield_swap.aleo` stack on testnet. Amounts are raw
native token units (the AMM does no decimal scaling); prices are Q128.128.
Wrapped assets (ALEO/USAD/USDCx) **route automatically** through the swap/LP
routers — fund them with *underlying* records (`credits.aleo` / stablecoin);
you never handle wrapper records. `mint` stores an immutable `withdrawal`
address on the position NFT (defaults to the recipient) and `collect` always
pays it — decide the payout wallet at mint time. The off-chain API is
selected per network — `https://api.testnet.swap.shield.fi` for testnet,
`https://api.swap.shield.fi` for mainnet (override with `SHIELD_SWAP_API_URL`).

## Install

```bash
pip install -e shield-swap-sdk                 # from the repo root
pip install -e "shield-swap-sdk[async]"        # + AsyncShieldSwap (httpx)
pip install -e "shield-swap-sdk[mcp]"          # + the MCP server
```

Requires `aleo-sdk>=0.3` (this repo's SDK; imports as `aleo`) and Python 3.10+.

## Agents

`AGENTS.md` (generated from the SDK's docstrings — always current) is the
one page an agent needs: the five-method lifecycle, the conversation pattern,
and the building-block reference. It ships in the wheel:

```bash
pip install shield-swap-sdk
python -m aleo_shield_swap          # prints the agent guide
```

Claude Code users get it via the `shield-swap` skill (packaged under
`aleo_shield_swap/skills/`, copy into `.claude/skills/`); any MCP client
can run the same lifecycle through `python -m aleo_shield_swap.mcp`.

## How calls work

Every read returns a value immediately. Every write returns a prepared
`DexCall` — nothing touches the network until you invoke a terminal method:

```python
call = dex.swap(pool_key=key, token_in_id=token, amount_in=10**9)

call.simulate()             # runs locally; no broadcast, no fee
call.transact(account)      # proves locally, broadcasts, pays the fee
call.delegate(account)      # proves via the delegated proving service
```

`transact` and `delegate` return the method's *typed result* (a `SwapHandle`,
`MintResult`, `ClaimResult`, …) built from the transaction's root-transition
outputs — not a bare transaction id. Local proving downloads SNARK parameters
on first use and takes minutes for the larger entrypoints; `delegate` is the
practical path and requires DPS credentials on the provider
(`api_key=`, `network_client.consumer_id`).

Chain reads and writes live directly on `ShieldSwap`; the off-chain DEX API
is namespaced under `.api`, so a call site always shows whether a value came
from the chain or the service.

## The method surface

**Chain reads** (node REST API):

| Verb | Returns |
|---|---|
| `get_pool(pool_key)` | The pool struct from the `pools` mapping. |
| `get_slot(pool_key)` | `SlotView` — current tick, sqrt price, liquidity, spacing. |
| `get_swap_output(swap_id)` | The finalized swap outcome; raises `SwapOutputNotFinalizedError` until the finalize lands (and again after the claim consumes it). |
| `get_swap_execution(swap_id)` | The fill receipt — executed height and per-hop amounts, gross/protocol/LP fee, post-trade price. Survives the claim; `None` until finalized. |
| `get_pool_creator(pool_key)` | Who created the pool (`None` for pools that predate creator tracking). |
| `is_pool_initialized(pool_key)` | Whether the pool exists on chain. |
| `get_public_balances(programs)` | Each token program's on-chain `balances` mapping entry for an address (raw base units; absent reads as 0). |
| `get_private_balances(programs)` | Summed unspent record amounts per token program (needs a registered record provider). |
| `get_balances()` | Public + private balances in one shape, joined through the API's token registry. |
| `derive_pool_key(token0, token1, fee)` / `derive_tick_key(pool_key, tick)` | Mapping keys derived locally — no network. |

**Writes** (each returns a `DexCall`):

| Verb | What it does |
|---|---|
| `swap(...)` | Phase one of the two-transaction private swap: locks the input record against a blinded identity. Returns a `SwapHandle` — persist it if the process might die before the claim. |
| `claim_swap_output(handle)` | Phase two: claims the finalized output as a private record. |
| `create_pool(...)` | Initializes a pool for a token pair + fee tier. |
| `mint(...)` | Opens a position in a tick range; returns a `MintResult` with the position's token id. |
| `increase_liquidity(...)` / `decrease_liquidity(...)` | Resizes a position (spends the position NFT record and returns a fresh one). |
| `collect(...)` | Pays out `tokens_owed` as private records. |
| `burn(...)` | Closes an emptied position and removes it from the `positions` mapping. |
| `plan_rebalance(...)` / `rebalance_position(...)` | Close a position and mint its successor range in ONE transaction via `shield_swap_rebalance_router.aleo` (testnet). The plan quotes what comes back, what the new range needs, and funding vs refund per token; every amount is asserted at the execution price, so re-plan and resubmit if the pool moved. |

A swap whose input was fully consumed (`amount_remaining == 0`) claims
through the no-refund entrypoints (`claim_swap_output_no_refund` and the
router's `claim_to_*_no_refund`), which never mint a zero-value refund
record — `claim_swap_output` picks them automatically.

Quote before you swap: pass `expected_out` from `dex.api.get_route(...)` —
without it a spot estimate is used, which ignores fees and price impact.
On busy pools leave slippage headroom: prices move between quote and
finalize, and a too-tight `amount_out_min` rejects safely at finalize.
Amounts are `u128` base units of the token; fees are microcredits.

Two liquidity behaviors worth knowing (both verified live): `mint` walks
the pool's on-chain tick list to compute its insertion hints
(`find_tick_predecessor`) — pass `tick_*_hint=` only if you know better.
And when a pool side is **wrapped**, routed `mint`/`increase` amounts for
that side must be *exactly* what the range consumes (the router burns the
wrapper change record) — single-sided ranges make this deterministic;
in-range wrapped amounts depend on the live price.

**DEX API** (`dex.api`, standalone as `ApiClient`): `get_pools`,
`get_tokens`, `get_route`, `get_ohlcv`, `get_positions`, `get_unclaimed`,
`get_pool_stats_batch`, `get_liquidity_distribution`, the compliance reads
(`get_compliance`, `get_token_compliance`, `get_pair_compliance` — check
before spending on a write), cookie-session management (`get_session`,
`refresh_session`, `list_sessions`, `revoke_session`, `logout`,
`logout_all`, `get_ws_ticket`), and referral issuance (`referral_settings`,
`list_referral_codes`, `generate_referral_codes`).
Route quoting, OHLCV, and the account views are auth-gated — call
`api.authenticate(address, sign)` once (challenge/verify by signature, no
funds required; the session rides as httpOnly cookies + a CSRF header and
is short-lived — mint a durable `ss_…` token via `create_api_token` for
anything long-running). Authentication is the whole gate — no invite is
needed. A **referral code** is optional attribution: `redeem_code` credits
the friend who shared it (once per account), and `my_referral_code` returns
the code this account shares with others.

## Privacy

The swap flow never puts your address on chain next to the output. `swap`
derives a single-use **blinded identity** from the signer's view key
(`derive_blinding_factor` / `derive_blinded_address` are exported for
verification against the TS SDK's vectors); the claim proves knowledge of the
blinding factor instead of revealing the owner.

Two conveniences trade secret material for service:

- `delegate` sends the transaction *authorization* to the proving service —
  it can see the transaction's contents (not your private key). Prove
  locally with `transact` if that is unacceptable.
- The hosted record scanner behind `get_private_balances` /
  `aleo.records.register` shares the account's **view key** with the
  scanning service, which can then see everything the account owns.

## Async

`AsyncShieldSwap` / `AsyncApiClient` mirror the sync surface method-for-method on
`aleo.AsyncAleo` (install the `[async]` extra):

```python
from aleo_shield_swap import AsyncShieldSwap

dex = AsyncShieldSwap(async_aleo)
handle = await (await dex.swap(pool_key=key, token_in_id=token,
                               amount_in=10**9)).delegate()
```

## Agent tools and MCP

`shield_swap_tools()` returns JSON-schema tool definitions for the whole method
surface; `dispatch_tool(dex, name, args)` executes one. For MCP hosts, the
`[mcp]` extra ships a stdio server over the same definitions:

```bash
ALEO_PRIVATE_KEY=APrivateKey1... python -m aleo_shield_swap.mcp
```

Omit `ALEO_PRIVATE_KEY` for a read-only server. See the module docstring for
the full environment (`ALEO_ENDPOINT`, `ALEO_NETWORK`, DPS credentials).

## Generated bindings

The contract surface is pinned, not hand-written: `codegen/` holds the
deployed program source, its ABI, and the DEX API's OpenAPI document.
`_generated.py` (program structs + entrypoints via `aleo.codegen`) and
`_api_models.py` (API response models) are built from those pins. When the
deployed contract or API drifts, rerun `codegen/regen-abi.sh` /
`codegen/regen-openapi.sh` and reconcile.

## Tests

```bash
python -m pytest                        # hermetic tier — no network
python -m pytest -m live               # read-only against the REAL testnet + DEX API
python -m pytest -m "live and slow"    # spends real testnet funds (DPS proving)
ALEO_DEVNODE_UNPROVEN=1 \
python -m pytest -m devnode            # full AMM lifecycle on a local aleo-devnode
```

The devnode tier deploys the vendored `shield_swap.aleo` stack and drives
pool creation, liquidity, swaps, and burn end-to-end, hermetically. It needs
the `aleo-devnode` binary (`ALEO_DEVNODE_BIN` or on `PATH`) and skips
otherwise. Deployments are proofless (dummy verifying keys — the devnode
skips certificate verification); `ALEO_DEVNODE_UNPROVEN=1` extends that to
executions and is the fast path (~5 minutes). Without it, executions are
fully proven locally: expect SNARK parameter downloads and key synthesis on
first use.
